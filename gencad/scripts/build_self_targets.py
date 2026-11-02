"""Fine-tuning targets from our own model's scored attempts (scripts/sample_real.py + scripts/rich_iou.py --rows per
shard): per part, the best program by own IoU (ties -> greedy, then shortest), kept if >= --thr. Writes a parquet in
the `train_vlm_cq.py --teacher` format (stem, code), plus stats on how much the samples beat greedy.

  python scripts/build_self_targets.py --runs results/self_real_{0,1,2,3} --out data/self_real/targets.parquet
"""
import argparse
import os
import pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dirs with preds.pkl + rich_rows.pkl")
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    rows, gstat = [], []
    for run in a.runs:
        P = pickle.load(open(os.path.join(run, "preds.pkl"), "rb"))
        R = pickle.load(open(os.path.join(run, "rich_rows.pkl"), "rb"))["rows"]
        methods = ["g"] + sorted((m for m in P["preds"] if m != "g"), key=lambda m: int(m[1:]))
        for i, u in enumerate(P["uids"]):
            sc = [(R[m][i]["own"] if R[m][i] and R[m][i]["valid"] else -1.0) for m in methods]
            codes = [P["preds"][m][i] for m in methods]
            if max(sc) < 0:
                gstat.append((-1.0, -1.0))
                continue
            best = max(range(len(methods)), key=lambda k: (round(sc[k], 3), k == 0, -len(codes[k] or "")))
            gstat.append((sc[0], sc[best]))
            if sc[best] >= a.thr and codes[best]:
                rows.append({"uid": u, "family": P["family"][i], "stem": P["stems"][i], "code": codes[best],
                             "own": sc[best], "greedy_own": sc[0], "source": methods[best]})
    T = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_parquet(a.out)
    g = np.array(gstat)
    gv, bv = np.clip(g[:, 0], 0, None), np.clip(g[:, 1], 0, None)
    print(f"[self] {len(g)} parts: greedy own {100 * gv.mean():.1f}, best-of-{len(methods)} {100 * bv.mean():.1f}; "
          f"best beats greedy by > 0.02 on {100 * np.mean(bv - gv > 0.02):.1f} %, by > 0.10 on "
          f"{100 * np.mean(bv - gv > 0.10):.1f} %", flush=True)
    print(f"[self] wrote {len(T)} targets >= {a.thr} (mean own {T.own.mean():.3f}; from samples "
          f"{(T.source != 'g').mean() * 100:.0f} %; improves greedy by > 0.02 on "
          f"{(T.own - T.greedy_own > 0.02).mean() * 100:.0f} %) to {a.out}", flush=True)


if __name__ == "__main__":
    main()
