"""IoU of CadQuery predictions against CadQuery ground truth (rich-part test sets from scripts/gen_rich_data.py via
scripts/predict_vllm.py --rich). Same protocol as scripts/cq_iou.py: "gt" fits the GT to a unit box and applies the
same transform to the prediction (position/size errors count); "own" fits each solid to its own box. Invalid = 0.
Also a detail score (mm/detail.py: surface F-score at a tight tolerance, through-hole count, face-type histogram),
which sees missing fillets, chamfers and small holes that barely move volume IoU. Reports overall and per family;
--rows saves per-part results.

  python scripts/rich_iou.py --preds results/<run>/rich_preds.pkl --out results/<run>/rich_iou.json --rows ...
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def _fit(shape, ref=None):
    bb = (ref or shape).BoundingBox()
    c = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    s = 1.0 / max(bb.xlen, bb.ylen, bb.zlen)
    return shape.translate(tuple(-v for v in c)).scale(s)


def score(args):
    gt_code, pred = args
    from mm.cadquery_io import iou, run_cadquery
    gt = run_cadquery(gt_code)
    if gt is None:
        return None
    p = run_cadquery(pred) if pred else None
    if p is None:
        return {"valid": False, "gt": 0.0, "own": 0.0, "detail": 0.0}
    out = {"valid": True}
    for name, (a, b) in {"gt": (_fit(gt), _fit(p, gt)), "own": (_fit(gt), _fit(p))}.items():
        try:
            out[name] = float(min(iou(a, b), 1.0))
        except Exception:  # noqa: BLE001
            out[name] = 0.0
    try:
        from mm.detail import detail
        out.update(detail(gt, p))
    except Exception:  # noqa: BLE001 - e.g. a degenerate face that does not tessellate
        out.update({"surf": 0.0, "topo": 0.0, "faces": 0.0, "detail": 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=None)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    fam = np.array(P["family"])
    report, all_rows = {"n": len(P["uids"])}, {"uids": P["uids"], "family": list(fam), "rows": {}}
    for m in a.methods or list(P["preds"]):
        rows = []
        with ProcessPool(max_workers=a.workers) as pool:
            futs = [pool.schedule(score, args=((g, c),), timeout=a.timeout) for g, c in zip(P["gt_code"], P["preds"][m])]
            for fu in futs:
                try:
                    rows.append(fu.result())
                except Exception:  # noqa: BLE001
                    rows.append({"valid": False, "gt": 0.0, "own": 0.0, "failed": True})

        def summ(mask):
            r = [x for x, k in zip(rows, mask) if k and x is not None]
            return {"n": len(r), "IR": round(100 * float(np.mean([not x["valid"] for x in r])), 2),
                    "IoU_gt": round(100 * float(np.mean([x["gt"] for x in r])), 2),
                    "IoU_own": round(100 * float(np.mean([x["own"] for x in r])), 2),
                    "detail": round(100 * float(np.mean([x.get("detail", 0.0) for x in r])), 2)}
        res = {"all": summ(np.ones(len(rows), bool)), "families": {f: summ(fam == f) for f in sorted(set(fam))},
               "timeout_or_crash": sum(bool(x and x.get("failed")) for x in rows)}
        report[m] = res
        all_rows["rows"][m] = rows
        print(f"{m:20s} all: IoU gt/own={res['all']['IoU_gt']:.1f}/{res['all']['IoU_own']:.1f} "
              f"detail={res['all']['detail']:.1f} IR={res['all']['IR']:.1f} "
              f"(n={res['all']['n']}) timeout/crash={res['timeout_or_crash']}", flush=True)
        for f, v in res["families"].items():
            print(f"    {f:18s} IoU gt {v['IoU_gt']:5.1f} own {v['IoU_own']:5.1f} detail {v['detail']:5.1f} IR {v['IR']:4.1f} "
                  f"(n={v['n']})", flush=True)
    json.dump(report, open(a.out, "w"), indent=1)
    if a.rows:
        pickle.dump(all_rows, open(a.rows, "wb"))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
