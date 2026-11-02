"""Training data from Claude's programs on real training parts (`scripts/claude_baseline.py submit --rich ... --sample`
and `retry`): per part, the better of Claude's first attempt and its guided second attempt, kept if its own IoU
against the real STEP is >= --thr (SFT targets for `train_vlm_cq.py --teacher`); and, where the second attempt fixed
the first, a revision-turn example (first attempt + its render or build error -> second attempt) in the
`train_vlm_cq.py --revision` format.

  python scripts/build_teacher.py --runs results/claude_teacher_real{,2,3} --retries results/claude_retry_real{,2,3} \
      --out data/claude_teacher
Writes <out>/teacher.parquet (uid, family, stem, code, own, source), <out>/revision.parquet and <out>/attempts/*.png.
Render stems are relative to gencad/ (data/rich/real_train/...): unpack the real training shards there.
"""
import argparse
import json
import os
import pickle
import shutil


def load(run):
    P = pickle.load(open(os.path.join(run, "preds.pkl"), "rb"))
    R = pickle.load(open(os.path.join(run, "rich_rows.pkl"), "rb"))["rows"]["claude:view4"]
    return P, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--retries", nargs="*", default=[])
    ap.add_argument("--thr", type=float, default=0.8)
    ap.add_argument("--rev_gain", type=float, default=0.05, help="second attempt must beat the first by this much")
    ap.add_argument("--max_chars", type=int, default=3000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    retry_of = {}
    for rd in a.retries:
        if not os.path.exists(os.path.join(rd, "rich_rows.pkl")):
            print(f"[teacher] {rd}: not scored yet, skipped", flush=True)
            continue
        meta = json.load(open(os.path.join(rd, "batch.json")))
        retry_of[os.path.normpath(meta["src"])] = (rd, meta)
    os.makedirs(os.path.join(a.out, "attempts"), exist_ok=True)
    teach, rev = [], []
    for run in a.runs:
        P, R = load(run)
        codes = P["preds"]["claude:view4"]
        rd, meta = retry_of.get(os.path.normpath(run), (None, None))
        if rd:
            P2, R2 = load(rd)
            codes2 = P2["preds"]["claude:view4"]
        tag = os.path.basename(os.path.normpath(run))
        for i, u in enumerate(P["uids"]):
            first = R[i]["own"] if R[i] and R[i]["valid"] else -1.0
            second = R2[i]["own"] if rd and str(i) in meta["attempt"] and R2[i] and R2[i]["valid"] else -1.0
            best, code, src = (second, codes2[i], "retry") if second > first else (first, codes[i], "first")
            if best >= a.thr and code and len(code) <= a.max_chars:
                teach.append({"uid": u, "family": P["family"][i], "stem": P["stems"][i], "code": code, "own": best,
                              "source": src, "run": tag})
            if second >= a.thr and second >= first + a.rev_gain and codes2[i] and len(codes2[i]) <= a.max_chars:
                rid = f"{tag}_{i}"
                st = meta["attempt_stem"].get(str(i))
                rel = None
                if st and all(os.path.exists(f"{st}_{k}.png") for k in range(4)):
                    for k in range(4):
                        shutil.copy(f"{st}_{k}.png", os.path.join(a.out, "attempts", f"{rid}_{k}.png"))
                    rel = os.path.join("attempts", rid)
                err = meta["attempt_error"].get(str(i))
                if rel is None and not err:
                    continue                           # rendered attempt whose images are gone: skip
                rev.append({"rid": rid, "uid": u, "mode": "view4", "views": json.dumps([0, 1, 2, 3]), "prompt": "",
                            "attempt": meta["attempt"][str(i)] or "", "attempt_iou": first, "target": codes2[i],
                            "kind": "fix_claude", "error": err if rel is None else None, "attempt_stem": rel,
                            "input_stem": P["stems"][i]})
        print(f"[teacher] {run}: {len(P['uids'])} parts, retry {'yes' if rd else 'no'}; totals so far: "
              f"{len(teach)} targets, {len(rev)} revision examples", flush=True)
    T = pd.DataFrame(teach)
    T.to_parquet(os.path.join(a.out, "teacher.parquet"))
    pd.DataFrame(rev).to_parquet(os.path.join(a.out, "revision.parquet"))
    print(f"[teacher] wrote {len(T)} targets (mean own {T.own.mean():.3f}; {T.source.value_counts().to_dict()}; "
          f"{T.family.value_counts().to_dict()}) and {len(rev)} revision examples to {a.out}", flush=True)


if __name__ == "__main__":
    main()
