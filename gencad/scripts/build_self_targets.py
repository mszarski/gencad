"""Fine-tuning targets from our own model's scored attempts (scripts/sample_real.py + scripts/rich_iou.py --rows per
shard): per part, the best program by score = (1 - detail_w) * own IoU + detail_w * detail (mm/detail.py; ties ->
greedy, then shortest), kept if >= --thr. Writes a parquet in the `train_vlm_cq.py --teacher` format (stem, code), plus
stats on how much the samples beat greedy. --aug adds one copy per target for each randomised render set
(scripts/render_variants.py rand_*), same program, so the model learns that the look of the input does not change it.

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
    ap.add_argument("--detail_w", type=float, default=0.4)
    ap.add_argument("--aug", nargs="*", default=[], help="randomised render dirs, e.g. data/rich/real_rand/rand_a")
    ap.add_argument("--src_root", default="data/rich/real_train", help="render root the stems start with")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    rows, gstat = [], []
    for run in a.runs:
        P = pickle.load(open(os.path.join(run, "preds.pkl"), "rb"))
        R = pickle.load(open(os.path.join(run, "rich_rows.pkl"), "rb"))["rows"]
        methods = ["g"] + sorted((m for m in P["preds"] if m != "g"), key=lambda m: int(m[1:]))
        for i, u in enumerate(P["uids"]):
            sc = [((1 - a.detail_w) * R[m][i]["own"] + a.detail_w * R[m][i].get("detail", 0.0)
                   if R[m][i] and R[m][i]["valid"] else -1.0) for m in methods]
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
    n_base = len(T)
    if a.aug:
        extra = []
        for d in a.aug:
            A = T.copy()
            A["stem"] = [st.replace(a.src_root, d, 1) for st in A.stem]
            A = A[[os.path.exists(f"{st}_0.png") for st in A.stem]]
            extra.append(A.assign(source=A.source + "@" + os.path.basename(d)))
        T = pd.concat([T] + extra, ignore_index=True)
        print(f"[self] + {len(T) - n_base} randomised-render copies from {a.aug}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    T.to_parquet(a.out)
    g = np.array(gstat)
    gv, bv = np.clip(g[:, 0], 0, None), np.clip(g[:, 1], 0, None)
    print(f"[self] {len(g)} parts: greedy score {100 * gv.mean():.1f}, best-of-{len(methods)} {100 * bv.mean():.1f}; "
          f"best beats greedy by > 0.02 on {100 * np.mean(bv - gv > 0.02):.1f} %, by > 0.10 on "
          f"{100 * np.mean(bv - gv > 0.10):.1f} %", flush=True)
    B = T.head(n_base)
    print(f"[self] wrote {len(T)} rows ({n_base} targets with score >= {a.thr}, mean {B.own.mean():.3f}; from samples "
          f"{(B.source != 'g').mean() * 100:.0f} %; beat greedy by > 0.02 on "
          f"{(B.own - B.greedy_own > 0.02).mean() * 100:.0f} %) to {a.out}", flush=True)


if __name__ == "__main__":
    main()
