"""Best-of-N at inference: pick one program per part among greedy + samples (scripts/sample_real.py layout, methods
`g`, `s1..sN`) by render agreement with the input views (revise_guard.agreement, pix score; no GT). Reports own IoU
and detail of greedy, of the picks, and of the oracle (best by GT), from the rich_iou.py rows of the same preds file.

  python scripts/pick_render.py --preds results/gbon_real_0/preds.pkl --rows results/gbon_real_0/rich_rows.pkl \
      --out results/gbon_real_0/pick.json
Also writes <out>_picks.pkl: per part the picked method and its row, for paired comparisons.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from pebble import ProcessPool
    from scripts.revise_guard import agreement
    P = pickle.load(open(a.preds, "rb"))
    R = pickle.load(open(a.rows, "rb"))
    R = R["rows"] if "rows" in R else R
    methods = ["g"] + sorted((m for m in P["preds"] if m != "g"), key=lambda m: int(m[1:]))
    views = [0, 1, 2, 3] if P.get("mode", "view4") == "view4" else [0]
    n = len(P["uids"])
    with ProcessPool(max_workers=a.workers) as pool:
        futs = {m: [pool.schedule(agreement, args=((c, s, views, a.size),), timeout=120)
                    for c, s in zip(P["preds"][m], P["stems"])] for m in methods}
        agr = {}
        for m, fs in futs.items():
            agr[m] = []
            for fu in fs:
                try:
                    agr[m].append(fu.result())
                except Exception:  # noqa: BLE001
                    agr[m].append(None)
    ok = [i for i in range(n) if R["g"][i] is not None]
    val = lambda m, i, k: R[m][i][k] if R[m][i] and R[m][i]["valid"] else 0.0  # noqa: E731
    picks, res = {}, {}
    for K in (1, 3, 5, len(methods)):
        cand = methods[:K]
        pk = {}
        for i in ok:
            sc = [(agr[m][i] if agr[m][i] is not None else -1e9, m == "g") for m in cand]
            pk[i] = cand[max(range(K), key=lambda j: sc[j])]
        picks[K] = pk
        res[f"pick{K}"] = {k: 100 * float(np.mean([val(pk[i], i, k) for i in ok])) for k in ("own", "detail")}
        res[f"pick{K}"]["valid"] = 100 * float(np.mean([bool(R[pk[i]][i] and R[pk[i]][i]["valid"]) for i in ok]))
    res["oracle_own"] = 100 * float(np.mean([max(val(m, i, "own") for m in methods) for i in ok]))
    res["mean_sample_own"] = 100 * float(np.mean([val(m, i, "own") for m in methods[1:] for i in ok]))
    res["picked_greedy_share"] = float(np.mean([picks[len(methods)][i] == "g" for i in ok]))
    gd = np.array([val("g", i, "own") for i in ok])
    pd_ = np.array([val(picks[len(methods)][i], i, "own") for i in ok])
    d = pd_ - gd
    res["paired_vs_greedy_own"] = {"d": 100 * float(d.mean()), "se": 100 * float(d.std() / len(d) ** .5),
                                   "better>0.05": int((d > 0.05).sum()), "worse>0.05": int((d < -0.05).sum())}
    res["n"] = len(ok)
    json.dump(res, open(a.out, "w"), indent=1)
    pickle.dump({"uids": P["uids"], "picks": picks, "agr": agr}, open(os.path.splitext(a.out)[0] + "_picks.pkl", "wb"))
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
