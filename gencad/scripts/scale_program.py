"""Uniformly rescale a CadQuery program by multiplying its length literals by k (angles, counts and directions kept).

Used to turn our verified real-part programs (written at an arbitrary scale, since renders do not show size; longest side
~200) into programs at varied sizes in mm for dimensioned drawings. The Fusion 360 Gallery STEP files are themselves
normalised (longest side 20), so the real size is unknown: each program gets a factor from --factors (default 0.1 0.2
0.25 0.5 1). For drawings of any source, --size lo hi (default 10 500) rescales each program to a random longest side
(log-uniform; half snap to a round factor), so a drawing's printed numbers are the only way to know the part's size. Python's tokenizer walks the
source keeping a stack of (call name, argument index, keyword); a number is left alone when it is an angle or a count
(e.g. the angle of `.rotate`, `revolve`, `polarArray`, `rect`/`slot2D`/`trapezoid` angles, `rarray`/`polarArray`
counts, `polygon` sides, `transformed(rotate=...)`) or a direction vector (`xDir`, `normal`, `dir`, `axis*`).
`scale_and_check` verifies the result: it must build, and its bounding box must be k times the original's (1 %).

  python scripts/scale_program.py --targets <parquet uid, code> --size 10 500 --out x.parquet
"""
import io
import sys
import tokenize

import numpy as np

# (call, positional index) whose number is not a length; "*" = every positional argument
KEEP_POS = {("rotate", 2), ("revolve", 0), ("polarArray", 1), ("polarArray", 2), ("polarArray", 3),
            ("rarray", 2), ("rarray", 3), ("polygon", 0), ("rect", 2), ("slot2D", 2), ("trapezoid", 2),
            ("trapezoid", 3), ("cskHole", 2), ("sagittaArc", -1), ("Location", 1), ("rotateAboutCenter", 1),
            ("parray", 1), ("parray", 2), ("parray", 3), ("regularPolygon", 1), ("regularPolygon", 2),
            ("slot", 2)}
# positional entries apply to numbers written directly in that argument, not inside nested lists / tuples (so
# Sketch.polygon([(x, y), ...]) is scaled while Workplane.polygon(nSides, d) keeps nSides); keywords apply throughout
KEEP_KW = {"angle", "angleDegrees", "rotate", "xDir", "normal", "dir", "axisStart_dir", "nSides", "count", "angle1",
           "angle2", "startAngle", "endAngle", "taper", "mode", "tol"}
DIRECTION_CALLS = {"Vector"}       # Vector(...) as an argument of Plane(..., xDir=Vector(...)) is covered by KEEP_KW


def scale_code(code, k):
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    out = []
    stack = []                      # entries: [name, arg_index, current_keyword, keep_whole_call]
    prev_name = None
    i = 0
    while i < len(toks):
        t = toks[i]
        s = t.string
        if t.type == tokenize.OP and s in "([{":
            kw = stack[-1][2] if stack else None
            parent_keep = bool(stack and (stack[-1][3] or kw in KEEP_KW))
            stack.append([prev_name if s == "(" else None, 0, None, parent_keep])
        elif t.type == tokenize.OP and s in ")]}":
            if stack:
                stack.pop()
        elif t.type == tokenize.OP and s == "," and stack:
            stack[-1][1] += 1
            stack[-1][2] = None
        elif t.type == tokenize.OP and s == "=" and stack and toks[i - 1].type == tokenize.NAME:
            stack[-1][2] = toks[i - 1].string
        elif t.type == tokenize.NUMBER and stack:
            top = stack[-1]
            keep = (top[3] or top[2] in KEEP_KW or (top[0], top[1]) in KEEP_POS
                    or (i > 0 and toks[i - 1].type == tokenize.OP and toks[i - 1].string in ("/", "//", "**")))
            if not keep and not ("j" in s.lower()):
                v = float(s) * k
                s = f"{v:.4g}" if abs(v) >= 1 else f"{v:.3g}"
                t = t._replace(string=s)
        prev_name = s if t.type == tokenize.NAME else (prev_name if t.type == tokenize.OP and s == "." else None)
        if t.type == tokenize.NAME:
            prev_name = s
        out.append(t)
        i += 1
    return tokenize.untokenize(out)


def scale_and_check(code, k, tol=0.01):
    """-> scaled code, or None when it does not build or its box is not k times the original's."""
    from mm.cadquery_io import run_cadquery
    a = run_cadquery(code)
    if a is None:
        return None
    try:
        b_code = scale_code(code, k)
    except (tokenize.TokenError, IndentationError):
        return None
    b = run_cadquery(b_code)
    if b is None:
        return None
    ba, bb = a.BoundingBox(), b.BoundingBox()
    for x, y in ((ba.xlen, bb.xlen), (ba.ylen, bb.ylen), (ba.zlen, bb.zlen)):
        if abs(y - k * x) > tol * k * max(ba.xlen, ba.ylen, ba.zlen):
            return None
    if abs(b.Volume() - k ** 3 * a.Volume()) > 3 * tol * k ** 3 * a.Volume():
        return None
    return b_code


def _job(args):
    code, k = args
    return scale_and_check(code, k)


def _job_size(args):
    """Rescale to a target longest side: (code, target mm, nice factors or None) -> (code, k) or None."""
    code, target, nice = args
    from mm.cadquery_io import run_cadquery
    s = run_cadquery(code)
    if s is None:
        return None
    b = s.BoundingBox()
    L = max(b.xlen, b.ylen, b.zlen)
    if L <= 1e-6:
        return None
    k = target / L
    if nice:                                         # a round factor near it keeps printed values short
        k = min(nice, key=lambda f: abs(np.log(f / k)))
    c = scale_and_check(code, k)
    return (c, k) if c else None


def main():
    import argparse
    import os
    import random
    import pandas as pd
    from pebble import ProcessPool
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mm.sysutil import usable_cpus
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="parquet with uid, code (e.g. scripts/build_self_targets.py; "
                    "rows whose source has '@' are generated parts and skipped)")
    ap.add_argument("--factors", type=float, nargs="+", default=None,
                    help="fixed factors, one drawn per program (default: --size)")
    ap.add_argument("--size", type=float, nargs=2, default=[10.0, 500.0],
                    help="longest side drawn log-uniform in [lo, hi] mm; half the parts snap to a round factor")
    ap.add_argument("--prefix", default="", help="prepended to uids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = pd.read_parquet(a.targets)
    if "source" in T:
        T = T[~T.source.astype(str).str.contains("@")].reset_index(drop=True)   # real parts only
    rng = random.Random(a.seed)
    nice = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5]
    if a.factors:
        jobs = [(_job, (c, rng.choice(a.factors))) for c in T.code]
    else:
        lo, hi = np.log(a.size[0]), np.log(a.size[1])
        jobs = [(_job_size, (c, float(np.exp(rng.uniform(lo, hi))), nice if rng.random() < 0.5 else None))
                for c in T.code]
    rows = []
    with ProcessPool(max_workers=usable_cpus(), max_tasks=200) as pool:
        futs = [pool.schedule(fn, args=(x,), timeout=120) for fn, x in jobs]
        for u, (fn, x), f in zip(T.uid, jobs, futs):
            try:
                r = f.result()
            except Exception:  # noqa: BLE001
                r = None
            if r is None:
                continue
            c, k = (r, x[1]) if fn is _job else r
            rows.append({"uid": a.prefix + str(u).replace("/", "_"), "code": c, "factor": k})
    pd.DataFrame(rows).to_parquet(a.out)
    print(f"[scale] {len(rows)} of {len(T)} programs rescaled", flush=True)


if __name__ == "__main__":
    main()
