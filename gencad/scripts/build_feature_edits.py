"""Feature-edit training data: "here is a program and what it builds; add the features it is missing".

For each target program A (our own verified programs on real parts, data/self_real/targets.parquet; generator programs
of rich parts), we delete feature operations from the code to get a program B that builds a part without some of its
features: `.fillet(...)` / `.chamfer(...)` (with the edge/face selectors before them), `.hole(...)` / `.cboreHole` /
`.cskHole` / `.cutBlind` / `.cutThruAll`, `.cut(...)` / `.union(...)` of a sub-shape, and sketch cut-outs
(`.push([...]).circle(..., mode='s')`, `.rect(..., mode='s')`). B must build one valid solid and quiddity (mm/features.py)
must see a feature difference (or the volume must change by > 0.5 %), and the volume may change by at most 40 %
(deleting a union can otherwise remove the main body: a rebuild, not a feature edit). Each row stores B, B's render with the input
cameras, quiddity's summary of B ("2 x hole d6 through; 1 x chamfer 1"), and target A; plus "keep" rows where the
attempt is A itself (target A). Layout = scripts/sample_rich_rev.py (`train_vlm_cq.py --revision <out>`), with the extra
column `features` (shown in the revision prompt by mm.vlm_cad.revision_text).

  python scripts/build_feature_edits.py --targets data/self_real/targets.parquet --rich data/rich/train \
      --max_rich 10000 --out data/feat_edits
"""
import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402

FEATURE_OPS = ("fillet", "chamfer", "hole", "cboreHole", "cskHole", "cutBlind", "cutThruAll", "cut", "union")
SELECTORS = ("edges", "faces", "vertices", "workplane", "pushPoints", "center", "rarray", "polarArray", "tag",
             "transformed")


def calls(code):
    """All `.name(args)` spans at any nesting depth: [(name, start, end)] (end exclusive, after the ')')."""
    out = []
    for m in re.finditer(r"\.([A-Za-z_]\w*)\(", code):
        depth, j = 0, m.end() - 1
        while j < len(code):
            c = code[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                depth -= 1
                if depth == 0:
                    break
            elif c in "'\"":
                j = code.index(c, j + 1)
            j += 1
        out.append((m.group(1), m.start(), j + 1))
    return out


def candidates(code):
    """Deletable spans [(kind, start, end)], each removing one feature operation."""
    cs = calls(code)
    by_end = {e: (n, s, e) for n, s, e in cs}
    out = []
    for n, s, e in cs:
        args = code[s:e]
        if n in ("fillet", "chamfer", "hole", "cboreHole", "cskHole", "cutBlind", "cutThruAll"):
            start = s                                   # extend back over the selectors chained right before it
            while start in by_end and by_end[start][0] in SELECTORS:
                start = by_end[start][1]
            out.append((n if n in ("fillet", "chamfer") else "hole", start, e))
        elif n in ("cut", "union") and len(args) > 40:  # a sub-shape boolean
            out.append((n, s, e))
        elif n in ("circle", "rect", "polygon", "slot2D") and "mode='s'" in args.replace('"', "'"):
            start = s
            if start in by_end and by_end[start][0] == "push":
                start = by_end[start][1]
            out.append(("sketch_cut", start, e))
    return out


def work(args):
    """-> dict for one target program: A's summary and up to n verified ablations [(kind, code_B, summary_B)]."""
    code, n_abl, seed, rid, out = args
    a_max_dv = 0.4
    import cadquery as cq  # noqa: F401
    from mm.cadquery_io import run_cadquery
    from mm.features import recognise, summary
    from scripts.build_revision import work as render_attempt
    rng = random.Random(seed)
    A = run_cadquery(code)
    if A is None:
        return None
    RA = recognise(A)
    if RA is None:
        return None
    sumA = summary(RA[1])
    famA = sorted(f["family"] for f in RA[1])
    vA = A.Volume()
    cands = candidates(code)
    rng.shuffle(cands)
    abls = []
    for kind, s, e in cands:
        if len(abls) >= n_abl:
            break
        k = 1 if rng.random() < 0.7 else 2          # sometimes delete two operations
        spans = [(s, e)] + [(s2, e2) for _, s2, e2 in cands if (s2, e2) != (s, e) and (e2 <= s or s2 >= e)][:k - 1]
        b = code
        for s2, e2 in sorted(spans, reverse=True):
            b = b[:s2] + b[e2:]
        B = run_cadquery(b)
        if B is None:
            continue
        RB = recognise(B)
        if RB is None:
            continue
        famB = sorted(f["family"] for f in RB[1])
        dv = abs(B.Volume() - vA)
        if (famB == famA and dv < 0.005 * vA) or dv > a_max_dv * vA:   # no visible change / not a feature edit
            continue
        r_id = f"{rid}_a{len(abls)}"
        _, err, stem = render_attempt((r_id, b, out))
        if err or not stem:
            continue
        abls.append({"kind": kind, "attempt": b, "features": summary(RB[1]), "attempt_stem": os.path.relpath(stem, out),
                     "n_removed": len(spans)})
    keep = None
    if rng.random() < 0.35:
        _, err, stem = render_attempt((f"{rid}_k", code, out))
        if not err and stem:
            keep = {"kind": "keep", "attempt": code, "features": sumA, "attempt_stem": os.path.relpath(stem, out),
                    "n_removed": 0}
    return {"sumA": sumA, "abls": abls, "keep": keep, "n_cands": len(cands)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=[], help="parquets with stem, code (e.g. data/self_real/targets.parquet)")
    ap.add_argument("--rich", nargs="*", default=[], help="rich part dirs (<dir>/<shard>/parts.parquet)")
    ap.add_argument("--max_rich", type=int, default=10000)
    ap.add_argument("--max_chars", type=int, default=2500)
    ap.add_argument("--n_abl", type=int, default=2)
    ap.add_argument("--view4_p", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=usable_cpus())
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    rng = random.Random(a.seed)
    srcs = []
    for t in a.targets:
        T = pd.read_parquet(t)
        if "source" in T:
            T = T[~T.source.astype(str).str.contains("@")]   # base renders only (not the randomised copies)
        srcs.append(T.assign(src=os.path.basename(os.path.dirname(t)))[["uid", "family", "stem", "code", "src"]])
    for d in a.rich:
        dfs = []
        for sh in sorted(os.listdir(d)):
            pq = os.path.join(d, sh, "parts.parquet")
            if os.path.exists(pq):
                dfs.append(pd.read_parquet(pq).assign(stem=lambda x, sh=sh: [os.path.join(d, sh, "renders", u)
                                                                            for u in x.uid]))
        R = pd.concat(dfs, ignore_index=True).sample(frac=1.0, random_state=a.seed).head(a.max_rich)
        srcs.append(R.assign(src=os.path.basename(d))[["uid", "family", "stem", "code", "src"]])
    df = pd.concat(srcs, ignore_index=True)
    df = df[df.code.str.len() <= a.max_chars]
    df = df[[os.path.exists(f"{s}_3.png") for s in df.stem]].reset_index(drop=True)
    if a.limit:
        df = df.sample(frac=1.0, random_state=a.seed).head(a.limit).reset_index(drop=True)
    print(f"[feat-edit] {len(df)} target programs: {df.src.value_counts().to_dict()}", flush=True)
    os.makedirs(os.path.join(a.out, "attempts"), exist_ok=True)
    rows, t0, stats = [], time.time(), {"none": 0}
    with ProcessPool(max_workers=a.workers) as pool:
        futs = [pool.schedule(work, args=((c, a.n_abl, a.seed + i, f"{s}_{i}".replace("/", "_"), a.out),),
                              timeout=a.timeout) for i, (c, s) in enumerate(zip(df.code, df.src))]
        for i, fu in enumerate(futs):
            try:
                r = fu.result()
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                stats["none"] += 1
                continue
            views = [0, 1, 2, 3] if rng.random() < a.view4_p else [rng.randrange(4)]
            for ab in r["abls"] + ([r["keep"]] if r["keep"] else []):
                stats[ab["kind"]] = stats.get(ab["kind"], 0) + 1
                rows.append({"rid": os.path.basename(ab["attempt_stem"]), "uid": df.uid[i], "family": df.family[i],
                             "src": df.src[i], "mode": "view4" if len(views) == 4 else "view1",
                             "views": json.dumps(views), "prompt": "", "attempt": ab["attempt"],
                             "attempt_iou": 1.0 if ab["kind"] == "keep" else 0.0, "target": df.code[i],
                             "kind": "keep" if ab["kind"] == "keep" else "feat_" + ab["kind"], "error": None,
                             "attempt_stem": ab["attempt_stem"], "input_stem": df.stem[i],
                             "features": ab["features"], "features_target": r["sumA"], "n_removed": ab["n_removed"]})
            if i % 1000 == 0:
                print(f"[feat-edit] {i}/{len(futs)} programs, {len(rows)} rows, {stats}, {time.time() - t0:.0f}s",
                      flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(os.path.join(a.out, "revision.parquet"))
    print(f"[feat-edit] wrote {len(out)} rows to {a.out}: {out.kind.value_counts().to_dict()}; by source "
          f"{out.src.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
