"""Rotation-aligned own IoU: how much of a prediction's error is only orientation?

For each part, both solids are fitted to their own unit box (as for own IoU); points sampled on their surfaces; the
prediction is rigidly aligned to the GT by ICP (rotation + small translation), started from each of the 24
axis-aligned rotations, keeping the best fit; then the own IoU of the aligned prediction is computed. Comparing the
drop of plain vs aligned IoU between two input conditions (e.g. normal vs rotated cameras) separates "right part,
turned wrong" from "wrong part". Symmetric parts can align a little better than the true pose even on normal renders,
which is why both conditions are scored the same way.

  python scripts/aligned_iou.py --preds results/<run>/rich_preds.pkl --method vlm_cq:view4 --out results/<run>/aligned.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def axis_rotations():
    mats = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for signs in np.ndindex(2, 2, 2):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = -1 if s else 1
            if np.linalg.det(m) > 0:
                mats.append(m)
    return mats


def icp(src, tree, dst, R0, iters=25):
    """Rigid ICP of points src onto the cloud dst (KD-tree `tree`), from rotation R0. -> (R, t, mean distance)."""
    R, t = R0.copy(), np.zeros(3)
    for _ in range(iters):
        p = src @ R.T + t
        d, j = tree.query(p)
        q = dst[j]
        mp, mq = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - mp).T @ (q - mq))
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        dR = Vt.T @ D @ U.T
        R, t = dR @ R, dR @ (t - mp) + mq
    d, _ = tree.query(src @ R.T + t)
    return R, t, float(d.mean())


def work(args):
    gt_code, pred = args
    import cadquery as cq
    import trimesh
    from scipy.spatial import cKDTree
    from mm.cadquery_io import iou, run_cadquery
    from mm.detail import _fit, mesh
    gt, p = run_cadquery(gt_code), run_cadquery(pred) if pred else None
    if gt is None:
        return None
    if p is None:
        return {"valid": False, "own": 0.0, "aligned": 0.0, "angle": 0.0}
    a, b = _fit(gt), _fit(p)
    own = float(min(iou(a, b), 1.0))
    pa = trimesh.sample.sample_surface(mesh(a), 1500, seed=0)[0]
    pb = trimesh.sample.sample_surface(mesh(b), 1500, seed=1)[0]
    tree = cKDTree(pa)
    best = min((icp(pb, tree, pa, R0) for R0 in axis_rotations()), key=lambda r: r[2])
    R, t, _ = best
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    try:
        b2 = _fit(b.transformGeometry(cq.Matrix([list(r) for r in M[:3]])))
        aligned = float(min(iou(a, b2), 1.0))
    except Exception:  # noqa: BLE001
        aligned = own
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return {"valid": True, "own": own, "aligned": max(aligned, own), "angle": angle}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--method", default="vlm_cq:view4")
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    P = pickle.load(open(a.preds, "rb"))
    rows = []
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(work, args=((g, c),), timeout=300) for g, c in zip(P["gt_code"], P["preds"][a.method])]
        for fu in futs:
            try:
                rows.append(fu.result())
            except Exception:  # noqa: BLE001
                rows.append({"valid": False, "own": 0.0, "aligned": 0.0, "angle": 0.0, "failed": True})
    pickle.dump({"uids": P["uids"], "rows": rows}, open(a.out, "wb"))
    r = [x for x in rows if x is not None]
    print(f"[aligned] {a.preds} {a.method}: n {len(r)}, own {100 * np.mean([x['own'] for x in r]):.1f}, aligned "
          f"{100 * np.mean([x['aligned'] for x in r]):.1f}, median ICP angle "
          f"{np.median([x['angle'] for x in r if x['valid']]):.1f} deg", flush=True)


if __name__ == "__main__":
    main()
