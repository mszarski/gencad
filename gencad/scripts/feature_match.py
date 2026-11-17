"""Feature-level agreement between predicted and GT solids, with quiddity (https://github.com/pzfreo/quiddity,
Apache-2.0; `pip install quiddity`): a rule-based recogniser that lists manufacturing features (holes, fillets,
chamfers, pockets, slots, bosses, steps, ...) of a STEP B-rep with sizes and positions.

Both solids are fitted to their own box (longest side 100, centred; as for own IoU), exported to STEP and recognised.
Features with an axis (holes, turned features) are located by their axis line (position along the axis
dropped: a through hole is reported at its entry side). Features of each family are matched one-to-one (greedy by distance): a match needs the feature location within
--loc_tol (in the 100-unit frame) and its main size (diameter / radius / chamfer leg / width) within --size_tol
relative. Per part and family: tp, n_gt, n_pred; summary = per-family precision / recall / F1 over all parts, and
the share of parts whose GT feature counts are reproduced exactly.

  python scripts/feature_match.py --preds results/gself_real/rich_preds.pkl --method vlm_cq:view4 --out results/feat/gself_v4.pkl
"""
import argparse
import json
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

SIZE_KEYS = ("diameter", "radius", "leg1", "width", "depth", "height")
LOC_KEYS = ("location", "at", "centre", "center", "origin", "position")


def _num3(v):
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return np.array(v, float)
    return None


def features(shape):
    """-> {family: [(loc or None, size or None), ...]} for a CadQuery shape (already fitted)."""
    import cadquery as cq
    import quiddity
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "p.step")
        cq.exporters.export(cq.Workplane().add(shape), f)
        doc = json.loads(quiddity.recognise(f))
    out = {}
    for fam, items in doc.items():
        if not isinstance(items, list) or not items or not isinstance(items[0], dict):
            continue
        L = []
        for it in items:
            loc = next((_num3(it.get(k)) for k in LOC_KEYS if _num3(it.get(k)) is not None), None)
            ax = it.get("axis")
            ax = _num3(ax) if not isinstance(ax, str) else np.eye(3)["xyz".index(ax)] if ax in ("x", "y", "z") else None
            if loc is not None and ax is not None and np.linalg.norm(ax) > 0:   # position along the axis is ambiguous
                ax = ax / np.linalg.norm(ax)                                    # (entry side): compare axis lines
                loc = loc - loc.dot(ax) * ax
            size = next((float(it[k]) for k in SIZE_KEYS if isinstance(it.get(k), (int, float))), None)
            L.append((loc, size))
        out[fam] = L
    return out


def match(G, P, loc_tol, size_tol):
    tp = 0
    used = set()
    for g in G:
        best, bd = None, 1e9
        for j, p in enumerate(P):
            if j in used:
                continue
            if g[1] is not None and p[1] is not None and abs(p[1] - g[1]) > size_tol * max(g[1], 1e-6):
                continue
            d = np.linalg.norm(g[0] - p[0]) if g[0] is not None and p[0] is not None else 0.0
            if d <= loc_tol and d < bd:
                best, bd = j, d
        if best is not None:
            used.add(best)
            tp += 1
    return tp


def _fit100(shape):
    bb = shape.BoundingBox()
    c = ((bb.xmin + bb.xmax) / 2, (bb.ymin + bb.ymax) / 2, (bb.zmin + bb.zmax) / 2)
    return shape.translate(tuple(-v for v in c)).scale(100.0 / max(bb.xlen, bb.ylen, bb.zlen))


def work(args):
    gt_code, pred, loc_tol, size_tol = args
    from mm.cadquery_io import run_cadquery
    gt = run_cadquery(gt_code)
    if gt is None:
        return None
    try:
        FG = features(_fit100(gt))
    except Exception:  # noqa: BLE001
        return None
    p = run_cadquery(pred) if pred else None
    try:
        FP = features(_fit100(p)) if p is not None else {}
    except Exception:  # noqa: BLE001
        FP = {}
    fams = set(FG) | set(FP)
    return {"valid": p is not None,
            "fam": {f: {"tp": match(FG.get(f, []), FP.get(f, []), loc_tol, size_tol), "n_gt": len(FG.get(f, [])),
                        "n_pred": len(FP.get(f, [])), "gt_sizes": [s for _, s in FG.get(f, [])],
                        "pred_sizes": [s for _, s in FP.get(f, [])]} for f in fams}}


def summarise(rows):
    r = [x for x in rows if x is not None]
    fams = sorted({f for x in r for f in x["fam"]})
    out = {"n": len(r)}
    for f in fams:
        tp = sum(x["fam"].get(f, {}).get("tp", 0) for x in r)
        ng = sum(x["fam"].get(f, {}).get("n_gt", 0) for x in r)
        npr = sum(x["fam"].get(f, {}).get("n_pred", 0) for x in r)
        if ng + npr == 0:
            continue
        pr, rc = tp / max(npr, 1), tp / max(ng, 1)
        cnt = np.mean([x["fam"].get(f, {}).get("n_gt", 0) == x["fam"].get(f, {}).get("n_pred", 0) for x in r])
        out[f] = {"n_gt": ng, "n_pred": npr, "precision": round(pr, 3), "recall": round(rc, 3),
                  "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3), "count_exact": round(float(cnt), 3)}
    tp = sum(v.get("tp", 0) for x in r for v in x["fam"].values())
    ng = sum(v.get("n_gt", 0) for x in r for v in x["fam"].values())
    npr = sum(v.get("n_pred", 0) for x in r for v in x["fam"].values())
    pr, rc = tp / max(npr, 1), tp / max(ng, 1)
    out["all"] = {"n_gt": ng, "n_pred": npr, "precision": round(pr, 3), "recall": round(rc, 3),
                  "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--loc_tol", type=float, default=5.0, help="in the 100-unit fitted frame")
    ap.add_argument("--size_tol", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    gts, preds = P["gt_code"][:a.limit], P["preds"][a.method][:a.limit]
    rows = []
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(work, args=((g, c, a.loc_tol, a.size_tol),), timeout=180) for g, c in zip(gts, preds)]
        for fu in futs:
            try:
                rows.append(fu.result())
            except Exception:  # noqa: BLE001
                rows.append(None)
    S = summarise(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump({"uids": P["uids"][:a.limit], "family": P["family"][:a.limit], "rows": rows, "summary": S},
                open(a.out, "wb"))
    print(f"[feat] {a.preds} {a.method}: {json.dumps(S)}", flush=True)


if __name__ == "__main__":
    main()


KEY_FAMILIES = ("holes", "fillets", "chamfers", "slots", "pockets", "prismatic_pockets", "bosses", "step_levels",
                "turned_steps", "grooves", "hole_patterns", "section_recesses")


def feat_f1(row):
    """Feature F1 over KEY_FAMILIES from a work() row (1.0 when neither solid has any of them)."""
    if row is None:
        return 0.0
    tp = sum(row["fam"].get(f, {}).get("tp", 0) for f in KEY_FAMILIES)
    ng = sum(row["fam"].get(f, {}).get("n_gt", 0) for f in KEY_FAMILIES)
    npr = sum(row["fam"].get(f, {}).get("n_pred", 0) for f in KEY_FAMILIES)
    if ng + npr == 0:
        return 1.0
    return 2 * tp / (ng + npr)


def rich_score_feat(args):
    """scripts/rich_iou.score + "feat" (feature F1 vs GT): the reward pieces for RL on rich/real parts."""
    gt_code, pred = args
    from scripts.rich_iou import score
    r = score((gt_code, pred))
    if r is not None and r.get("valid"):
        try:
            r["feat"] = feat_f1(work((gt_code, pred, 5.0, 0.25)))
        except Exception:  # noqa: BLE001
            r["feat"] = 0.0
    return r


def attempt_state(args):
    """Build + render a program with the input cameras and summarise its features (the edit loop's observation).
    -> (error or None, render stem or None, feature summary or None)."""
    code, rid, scratch = args
    from scripts.build_revision import work as render_attempt
    os.makedirs(os.path.join(scratch, "attempts"), exist_ok=True)
    _, err, stem = render_attempt((rid, code, scratch))
    if err or not stem:
        return err or "render failed", None, None
    try:
        from mm.cadquery_io import run_cadquery
        from mm.features import recognise, summary
        R = recognise(run_cadquery(code))
        return None, stem, (summary(R[1]) if R else None)
    except Exception:  # noqa: BLE001
        return None, stem, None
