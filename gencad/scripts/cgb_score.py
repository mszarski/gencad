"""Score our predictions on rich/real parts with CADGenBench's own evaluator (shape, interface, topology, validity,
cad_score), using auto-generated interface regions (scripts/interface_regions.py).

Two steps, because CADGenBench needs build123d / OCP 8 and our CadQuery stack OCP 7.9 (keep them in separate envs):
  1. prepare (our env): GT fixture dirs (ground_truth.step + jig_*.step) and candidate dirs (output.step), each
     candidate written twice: `raw` (as the program builds it) and `fit` (uniformly scaled to the GT's bounding-box
     diagonal and centred on it; our render-trained model never sees absolute size, CADGenBench does not rescale).
       python scripts/cgb_score.py prepare --preds results/gbon_real_0/preds.pkl --method g --work <dir>
  2. evaluate (cadgenbench env): runs cadgenbench.eval.evaluate.evaluate_result per part, collects scores.
       <cgb-venv>/bin/python scripts/cgb_score.py evaluate --work <dir> --out <dir>/scores.json
"""
import argparse
import json
import os
import pickle
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _prep_one(args):
    i, name, gt_code, code, work = args
    import cadquery as cq
    from mm.cadquery_io import run_cadquery
    from scripts.interface_regions import write_regions
    gt = run_cadquery(gt_code)
    if gt is None:
        return name, "gt fails"
    gdir = os.path.join(work, "gt", name)
    os.makedirs(gdir, exist_ok=True)
    cq.exporters.export(cq.Workplane().add(gt), os.path.join(gdir, "ground_truth.step"))
    n_reg = write_regions(gt, gdir)
    p = run_cadquery(code) if code else None
    for var in ("raw", "fit"):
        rdir = os.path.join(work, var, name)
        os.makedirs(rdir, exist_ok=True)
        json.dump({}, open(os.path.join(rdir, "result.json"), "w"))
        if p is None:
            continue
        s = p
        if var == "fit":
            bg, bp = gt.BoundingBox(), p.BoundingBox()
            k = bg.DiagonalLength / max(bp.DiagonalLength, 1e-9)
            s = p.translate(-bp.center).scale(k).translate(bg.center)
        cq.exporters.export(cq.Workplane().add(s), os.path.join(rdir, "output.step"))
    return name, n_reg


def prepare(a):
    from pebble import ProcessPool
    from mm.sysutil import usable_cpus
    P = pickle.load(open(a.preds, "rb"))
    codes = P["preds"][a.method]
    jobs = [(i, f"p{i:04d}", P["gt_code"][i], codes[i], a.work) for i in range(len(codes))][:a.limit]
    out = {}
    with ProcessPool(max_workers=a.workers or usable_cpus()) as pool:
        futs = [(j[1], pool.schedule(_prep_one, args=(j,), timeout=300)) for j in jobs]
        for name, fu in futs:
            try:
                out[name] = fu.result()[1]
            except Exception as e:  # noqa: BLE001
                out[name] = f"error {e}"[:80]
    json.dump({"uids": [P["uids"][int(k[1:])] for k in out], "regions": out}, open(os.path.join(a.work, "prep.json"), "w"))
    print(f"[cgb] prepared {len(out)} parts; with interface regions: "
          f"{sum(isinstance(v, int) and v > 0 for v in out.values())}", flush=True)


def _eval_one(args):
    work, var, name = args
    from cadgenbench.eval.evaluate import evaluate_result
    rdir, gdir = os.path.join(work, var, name), os.path.join(work, "gt", name)
    try:
        evaluate_result(rdir, gdir, candidate_step=(os.path.join(rdir, "output.step")
                                                    if os.path.exists(os.path.join(rdir, "output.step")) else None))
    except Exception as e:  # noqa: BLE001
        return var, name, {"error": f"{type(e).__name__}: {e}"[:200]}
    return var, name, json.load(open(os.path.join(rdir, "result.json")))


def evaluate(a):
    import concurrent.futures as cf
    names = sorted(os.listdir(os.path.join(a.work, "gt")))[:a.limit]
    jobs = [(a.work, var, n) for var in ("raw", "fit") for n in names]
    res = {"raw": {}, "fit": {}}
    with cf.ProcessPoolExecutor(max_workers=a.workers or 4) as ex:
        for var, name, r in ex.map(_eval_one, jobs):
            res[var][name] = r
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"))
    import numpy as np
    for var in ("raw", "fit"):
        R = [r for r in res[var].values() if "error" not in r]
        cs = [r.get("cad_score", 0) or 0 for r in R]
        print(f"[cgb] {var}: n {len(R)} (errors {len(res[var]) - len(R)}), mean cad_score {np.mean(cs):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "evaluate"])
    ap.add_argument("--preds")
    ap.add_argument("--method", default="g")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else evaluate(a)


if __name__ == "__main__":
    main()
