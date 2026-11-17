"""Build-inspect-fix at inference: turn 1 writes a program (greedy), it is built and rendered with the input cameras
(or its build error is captured), and turn 2 (mm.vlm_cad.revision_text, two images) writes a revision. Repeats for
--rounds. A revision that fails to build is dropped in favour of the previous program.

Same test inputs as scripts/predict_vllm.py (DeepCAD test via --uids_from, or rich test via --rich); writes a preds
file with keys vlm_cq:<mode> (turn 1) and vlm_cq:<mode>@r<k> (after round k), scored by cq_iou.py / rich_iou.py.

  python scripts/revise_vllm.py --model results/<run>/model --uids_from results/baselines_vlm_final/preds.pkl \
      --modes view1 view4 --rounds 2 --out results/<run>_test/revise.pkl
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mm.sysutil import usable_cpus  # noqa: E402


def attempt_feedback(args):
    """Build `code`; -> (error text or None, render stem or None[, feature summary]). Renders all 4 cameras like the
    input renders; with a third arg True also returns quiddity's feature summary (mm/features.py)."""
    code, stem = args[:2]
    feats = len(args) > 2 and args[2]
    from scripts.build_revision import error_of
    from scripts.render_views import DIRS, render, tessellate_cq
    if not code:
        return "no program was produced", None
    shape, err = error_of(code)
    if err:
        return err, None
    tris, fid = tessellate_cq(shape)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for j, d in enumerate(DIRS):
        render(tris, fid, d).save(f"{stem}_{j}.png")
    if feats:
        try:
            from mm.features import recognise, summary
            R = recognise(shape)
            return None, stem, (summary(R[1]) if R else None)
        except Exception:  # noqa: BLE001
            return None, stem, None
    return None, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--uids_from", default=None)
    ap.add_argument("--rich", default=None)
    ap.add_argument("--modes", nargs="*", default=["view1", "view4", "view1+expert"])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--revise_model", default=None, help="model for the revision turns (default: --model)")
    ap.add_argument("--features", action="store_true", help="add quiddity's feature summary of the attempt to the "
                    "revision prompt (models trained with feature-edit data / the RL edit turn)")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=max(1, usable_cpus() - 2))
    ap.add_argument("--scratch", default="/tmp/revise_renders")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from pebble import ProcessPool
    pool = ProcessPool(max_workers=a.workers)            # before CUDA
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import _messages, compose_views, configure, revision_text
    from train_vlm_cad import load_views
    print(f"[vlm] view grid {configure(a.model)} px", flush=True)

    text_index = {}
    if a.rich:
        import pandas as pd
        dfs = [pd.read_parquet(os.path.join(a.rich, sh, "parts.parquet")).assign(shard=sh)
               for sh in sorted(os.listdir(a.rich)) if os.path.exists(os.path.join(a.rich, sh, "parts.parquet"))]
        df = pd.concat(dfs, ignore_index=True)
        P = {"uids": list(df.uid), "dup": [False] * len(df), "family": list(df.family), "gt_code": list(df.code),
             "stems": [os.path.join(a.rich, sh, "renders", u) for sh, u in zip(df.shard, df.uid)]}
        a.modes = [m for m in a.modes if "expert" not in m]
    else:
        P = pickle.load(open(a.uids_from, "rb"))
        P = {k: P[k] for k in ("uids", "dup")}
        text_index = json.load(open(os.path.join(a.data, "text_index.json")))
    stems = P.get("stems") or [os.path.join(a.data, "renders", u) for u in P["uids"]]
    processor = AutoProcessor.from_pretrained(a.model)
    mem = 0.42 if a.revise_model else 0.85
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 2},
              gpu_memory_utilization=mem, enable_prefix_caching=True, generation_config="vllm", seed=0)
    llm_rev = llm if not a.revise_model else LLM(
        model=a.revise_model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 2},
        gpu_memory_utilization=mem, enable_prefix_caching=True, generation_config="vllm", seed=0)
    stop = [processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id]
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop)

    def request(images, text):
        msg = processor.apply_chat_template(_messages(images if images else None, text), tokenize=False,
                                            add_generation_prompt=True)
        r = {"prompt": msg}
        if images:
            r["multi_modal_data"] = {"image": images if len(images) > 1 else images[0]}
        return r

    preds, stats = {}, {}
    for mode in a.modes:
        t0 = time.time()
        views = [0, 1, 2, 3] if mode.startswith("view4") else [0]
        idx, inputs, prompts = [], [], []
        for i, u in enumerate(P["uids"]):
            if not os.path.exists(f"{stems[i]}_0.png"):
                continue
            p = text_index.get(u, {}).get("expert") if "expert" in mode else None
            if "expert" in mode and not p:
                continue
            idx.append(i)
            inputs.append(compose_views(load_views("", stems[i], views)))
            prompts.append(p)
        outs = llm.generate([request([im], p) for im, p in zip(inputs, prompts)], sp, use_tqdm=False)
        cur = [o.outputs[0].text.strip() or None for o in outs]
        rows = [None] * len(P["uids"])
        for i, c in zip(idx, cur):
            rows[i] = c
        preds[f"vlm_cq:{mode}"] = rows
        for k in range(1, a.rounds + 1):
            futs = [pool.schedule(attempt_feedback, args=((c, os.path.join(a.scratch, mode, f"r{k}", str(i)),
                                                            a.features),), timeout=120) for i, c in zip(idx, cur)]
            fb = []
            for fu in futs:
                try:
                    r = fu.result()
                    fb.append((r[0], r[1], r[2] if len(r) > 2 else None))
                except Exception as e:  # noqa: BLE001
                    fb.append((f"building it timed out ({type(e).__name__})", None, None))
            reqs = []
            for im, p, c, (err, st, ft) in zip(inputs, prompts, cur, fb):
                imgs = [im] + ([compose_views(load_views("", st, views))] if st else [])
                reqs.append(request(imgs, revision_text(p, c or "", err, ft)))
            new = [o.outputs[0].text.strip() or None for o in llm_rev.generate(reqs, sp, use_tqdm=False)]
            chk = [pool.schedule(attempt_feedback, args=((c, os.path.join(a.scratch, mode, f"c{k}", str(i))),),
                                 timeout=120) for i, c in zip(idx, new)]
            nxt, changed = [], 0
            for c_old, c_new, fu in zip(cur, new, chk):
                try:
                    err = fu.result()[0]
                except Exception:  # noqa: BLE001
                    err = "timeout"
                keep_new = err is None or c_old is None
                nxt.append(c_new if keep_new else c_old)
                changed += keep_new and c_new != c_old
            cur = nxt
            rows = [None] * len(P["uids"])
            for i, c in zip(idx, cur):
                rows[i] = c
            preds[f"vlm_cq:{mode}@r{k}"] = rows
            stats[f"{mode}@r{k}"] = {"changed": changed, "turn1_failed": sum(f[0] is not None for f in fb)}
            print(f"[revise] {mode} round {k}: changed {changed}/{len(idx)}; turn-1 failed to build "
                  f"{stats[f'{mode}@r{k}']['turn1_failed']}", flush=True)
        print(f"[revise] {mode} done in {time.time() - t0:.0f}s", flush=True)
    pool.close()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump({**P, "kind": "cadquery", "preds": preds, "revise_stats": stats}, open(a.out, "wb"))
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
