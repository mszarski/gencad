"""Best-of-N on engineering drawings with a picker that sees only the candidates (no ground truth).

Per drawing: the greedy answer plus N-1 samples (vLLM). Each candidate is built; the picker keeps the valid candidate
that agrees most with the others (mean centred volume IoU in mm, i.e. a consensus / minimum-Bayes-risk pick; ties go
to the greedy answer). Reported against the GT program: greedy, pick, oracle (best candidate), all as abs IoU (both
solids centred, no rescale), plus validity. Programs for --extra drawings (e.g. CADGenBench sheets) are picked the same
way and saved.

  python scripts/bon_drawings.py --model results/<run>/model --drawings data/draw_val --n 8 --out results/<run>/bon.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def _centred(s):
    return s.translate(-s.BoundingBox().center)


def pick(args):
    """(gt_code or None, [codes]) -> dict with per-candidate validity, pairwise agreement, GT IoUs and the pick."""
    gt_code, codes = args
    from mm.cadquery_io import iou, run_cadquery
    sols = []
    for c in codes:
        try:
            s = run_cadquery(c) if c else None
            sols.append(_centred(s) if s is not None else None)
        except Exception:  # noqa: BLE001
            sols.append(None)
    n = len(codes)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if sols[i] is not None and sols[j] is not None:
                try:
                    A[i, j] = A[j, i] = min(iou(sols[i], sols[j]), 1.0)
                except Exception:  # noqa: BLE001
                    pass
    valid = [s is not None for s in sols]
    agree = [A[i].sum() / max(n - 1, 1) if valid[i] else -1.0 for i in range(n)]
    best = int(np.argmax(agree)) if any(valid) else 0
    if valid[0] and agree[0] >= agree[best] - 1e-9:
        best = 0
    out = {"valid": valid, "agree": [float(x) for x in agree], "pick": best}
    if gt_code:
        g = run_cadquery(gt_code)
        if g is None:
            return None
        g = _centred(g)
        gi = []
        for s in sols:
            try:
                gi.append(float(min(iou(g, s), 1.0)) if s is not None else 0.0)
            except Exception:  # noqa: BLE001
                gi.append(0.0)
        out["gt_iou"] = gi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=1536)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import pandas as pd
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import DRAWING_PROMPT, configure, load_drawing, _messages
    import mm.vlm_cad as V
    configure(a.model)
    side = V.DRAWING_SIDE or 2048
    D = pd.read_parquet(os.path.join(a.drawings, "drawings.parquet")).head(a.limit)
    items = [(u, os.path.join(a.drawings, f), c) for u, f, c in zip(D.uid, D.png, D.code)]
    if a.extra:
        items += [("x_" + os.path.splitext(f)[0], os.path.join(a.extra, f), None) for f in sorted(os.listdir(a.extra))
                  if f.endswith(".png")]
    proc = AutoProcessor.from_pretrained(a.model)
    text = proc.apply_chat_template(_messages(object(), DRAWING_PROMPT), tokenize=False, add_generation_prompt=True)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"max_pixels": int(side * side * 0.72)}, gpu_memory_utilization=0.85)
    reqs = [{"prompt": text, "multi_modal_data": {"image": load_drawing(p, side)}} for _, p, _ in items]
    greedy = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens), use_tqdm=False)
    samp = llm.generate(reqs, SamplingParams(temperature=a.temperature, n=a.n - 1, max_tokens=a.max_new_tokens),
                        use_tqdm=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cands = [[g.outputs[0].text.strip()] + [o.text.strip() for o in s.outputs] for g, s in zip(greedy, samp)]
    pickle.dump({"items": [(u, p) for u, p, _ in items], "cands": cands},
                open(os.path.splitext(a.out)[0] + "_cands.pkl", "wb"))
    from pebble import ProcessPool
    res = {}
    with ProcessPool(max_workers=usable_cpus(), max_tasks=50) as pool:
        fs = [(u, pool.schedule(pick, args=((g, c),), timeout=900)) for (u, _, g), c in zip(items, cands)]
        for u, f in fs:
            try:
                res[u] = f.result()
            except Exception:  # noqa: BLE001
                res[u] = None
    R = [r for u, r in res.items() if r and "gt_iou" in r]
    summ = {"n_parts": len(R), "n": a.n, "temperature": a.temperature,
            "greedy": round(100 * float(np.mean([r["gt_iou"][0] for r in R])), 1),
            "pick": round(100 * float(np.mean([r["gt_iou"][r["pick"]] for r in R])), 1),
            "oracle": round(100 * float(np.mean([max(r["gt_iou"]) for r in R])), 1),
            "greedy_valid": round(100 * float(np.mean([r["valid"][0] for r in R])), 1),
            "pick_valid": round(100 * float(np.mean([r["valid"][r["pick"]] for r in R])), 1)}
    d = np.array([r["gt_iou"][r["pick"]] - r["gt_iou"][0] for r in R])
    summ["pick_minus_greedy"] = f"{100 * d.mean():+.1f} +- {100 * d.std() / np.sqrt(len(d)):.1f}"
    by = {}
    for u, r in res.items():
        if r and "gt_iou" in r:
            k = u.split("_")[0] if u.startswith(("g0", "gt", "g20", "real")) else "recode"
            by.setdefault(k, []).append((r["gt_iou"][0], r["gt_iou"][r["pick"]]))
    summ["by_source_greedy_pick"] = {k: [round(100 * float(np.mean([x[i] for x in v])), 1) for i in (0, 1)]
                                     for k, v in by.items()}
    X = {u: r for u, r in res.items() if u.startswith("x_")}
    if X:
        summ["extra_valid_greedy_pick"] = [sum(bool(r and r["valid"][0]) for r in X.values()),
                                           sum(bool(r and r["valid"][r["pick"]]) for r in X.values()), len(X)]
    json.dump({"summary": summ, "rows": res}, open(a.out, "w"))
    print(f"[bon] {a.model}: {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
