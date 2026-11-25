"""Drawing -> CadQuery evaluation: generate programs from engineering drawings with vLLM and score them in absolute
millimetres.

For drawings made by scripts/make_drawing.py (drawings.parquet holds the GT program): per part
  abs_iou   volume IoU after centring both solids on their bounding-box centres, no rescaling (what CADGenBench's
            rigid alignment rewards: size in mm counts)
  own_iou   IoU after fitting each solid to its own box (shape only, as for our render-trained models)
  size_err  mean relative error of the three bounding-box extents
  valid     the program builds one solid
Optionally also writes programs for other drawings (e.g. the 49 public CADGenBench sheets) with build validity only.

  python scripts/eval_drawings.py --model results/<run>/model --drawings data/draw_val --extra data/cgb_sheets \
      --out results/<run>/draweval.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def score(args):
    gt_code, code = args
    from mm.cadquery_io import iou, run_cadquery
    from mm.detail import _fit
    g = run_cadquery(gt_code)
    if g is None:
        return None
    p = run_cadquery(code) if code else None
    if p is None:
        return {"valid": False, "abs_iou": 0.0, "own_iou": 0.0, "size_err": 1.0}
    bg, bp = g.BoundingBox(), p.BoundingBox()
    try:
        a = float(min(iou(g.translate(-bg.center), p.translate(-bp.center)), 1.0))
    except Exception:  # noqa: BLE001
        a = 0.0
    try:
        o = float(min(iou(_fit(g), _fit(p)), 1.0))
    except Exception:  # noqa: BLE001
        o = 0.0
    eg, ep = np.array([bg.xlen, bg.ylen, bg.zlen]), np.array([bp.xlen, bp.ylen, bp.zlen])
    return {"valid": True, "abs_iou": a, "own_iou": o, "gt_size": float(max(eg)), "pred_size": float(max(ep)),
            "size_err": float(np.mean(np.abs(np.sort(ep) - np.sort(eg)) / np.maximum(np.sort(eg), 1e-6)))}


def builds(code):
    from mm.cadquery_io import run_cadquery
    return run_cadquery(code) is not None if code else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drawings", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--drawing_side", type=int, default=None, help="default: the model's gencad_vlm.json, else 2048")
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
    side = a.drawing_side or V.DRAWING_SIDE or 2048
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
    outs = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens), use_tqdm=False)
    codes = [o.outputs[0].text.strip() or None for o in outs]
    from pebble import ProcessPool
    rows, extra = {}, {}
    with ProcessPool(max_workers=usable_cpus()) as pool:
        fs = [(u, pool.schedule(score, args=((g, c),), timeout=120)) for (u, _, g), c in zip(items, codes) if g]
        fx = [(u, pool.schedule(builds, args=(c,), timeout=120)) for (u, _, g), c in zip(items, codes) if not g]
        for u, f in fs:
            try:
                rows[u] = f.result()
            except Exception:  # noqa: BLE001
                rows[u] = {"valid": False, "abs_iou": 0.0, "own_iou": 0.0, "size_err": 1.0}
        for u, f in fx:
            try:
                extra[u] = bool(f.result())
            except Exception:  # noqa: BLE001
                extra[u] = False
    R = [r for r in rows.values() if r is not None]
    summ = {k: round(100 * float(np.mean([r[k] for r in R])), 1) for k in ("valid", "abs_iou", "own_iou")}
    summ["size_err_median"] = round(float(np.median([r["size_err"] for r in R if r["valid"]] or [1.0])), 3)
    summ["n"] = len(R)
    V = [r for r in R if r["valid"] and r.get("pred_size", 0) > 0]
    if len(V) > 2:                                # does the model read size? (CAD-Recode alone is all ~200 mm)
        summ["size_corr_log"] = round(float(np.corrcoef(np.log([r["gt_size"] for r in V]),
                                                        np.log([r["pred_size"] for r in V]))[0, 1]), 3)
        summ["size_within_5pct"] = round(100 * float(np.mean([abs(r["pred_size"] / r["gt_size"] - 1) <= 0.05
                                                               for r in V])), 1)
    by = {}
    for u, r in rows.items():
        if r:
            by.setdefault(u.split("_")[0] if u.startswith(("g0", "gt")) else "recode", []).append(r["abs_iou"])
    summ["abs_iou_by_source"] = {k: round(100 * float(np.mean(v)), 1) for k, v in by.items()}
    if extra:
        summ["extra_valid"] = round(100 * float(np.mean(list(extra.values()))), 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "rows": rows, "extra_valid": extra}, open(a.out, "w"))
    pickle.dump({"items": [(u, p) for u, p, _ in items], "codes": codes},
                open(os.path.splitext(a.out)[0] + "_codes.pkl", "wb"))
    print(f"[draweval] {a.model}: {json.dumps(summ)}", flush=True)


if __name__ == "__main__":
    main()
