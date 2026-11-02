"""Self-improvement data on rich/real training parts (rejection sampling with our own model): for each part, one greedy
program and --n_samples sampled ones (vLLM), saved as one preds file per shard in the rich layout with methods
`g` (greedy) and `s1..sN`, so `scripts/rich_iou.py` (job `richiou`) scores them all against the GT (STEP for real
parts). `scripts/build_self_targets.py` then keeps the best program per part.

  python scripts/sample_real.py --model results/vlm_grpo_real2/model --rich data/rich/real_train --mode view4 \
      --n_samples 8 --out results/self_real
Writes <out>_<shard>/preds.pkl (one run dir per shard, so each can be scored by its own `richiou` job).
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rich", required=True)
    ap.add_argument("--mode", default="view4", choices=["view4", "view1"])
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--max_parts", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import pandas as pd
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from mm.vlm_cad import _messages, compose_views, configure
    from train_vlm_cad import load_views
    print(f"[vlm] view grid {configure(a.model)} px", flush=True)
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.85, enable_prefix_caching=True, generation_config="vllm", seed=0)
    stop = [processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id]
    views = [0, 1, 2, 3] if a.mode == "view4" else [0]
    for sh in sorted(os.listdir(a.rich)):
        pq = os.path.join(a.rich, sh, "parts.parquet")
        if not os.path.exists(pq):
            continue
        df = pd.read_parquet(pq)
        df = df.head(a.max_parts) if a.max_parts else df
        stems = [os.path.join(a.rich, sh, "renders", u) for u in df.uid]
        reqs = []
        for st in stems:
            im = compose_views(load_views("", st, views))
            reqs.append({"prompt": processor.apply_chat_template(_messages([im], None), tokenize=False,
                                                                 add_generation_prompt=True),
                         "multi_modal_data": {"image": im}})
        t0 = time.time()
        g = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop),
                         use_tqdm=False)
        s = llm.generate(reqs, SamplingParams(temperature=a.temp, n=a.n_samples, max_tokens=a.max_new_tokens,
                                              stop_token_ids=stop, seed=0), use_tqdm=False)
        preds = {"g": [o.outputs[0].text.strip() or None for o in g]}
        for k in range(a.n_samples):
            preds[f"s{k + 1}"] = [o.outputs[k].text.strip() or None for o in s]
        od = f"{a.out}_{sh}"
        os.makedirs(od, exist_ok=True)
        pickle.dump({"uids": list(df.uid), "dup": [False] * len(df), "family": list(df.family), "gt_code": list(df.code),
                     "stems": stems, "kind": "cadquery", "mode": a.mode, "preds": preds},
                    open(os.path.join(od, "preds.pkl"), "wb"))
        print(f"[sample] shard {sh}: {len(df)} parts x {1 + a.n_samples} programs in {time.time() - t0:.0f}s", flush=True)
    print(f"[done] {a.out}", flush=True)


if __name__ == "__main__":
    main()
