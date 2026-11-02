"""Online RL (GRPO) for the CadQuery writer: vLLM samples from the current policy, the HF model takes the gradient
step, and the new weights are loaded into the in-process vLLM engine after every step.

Each step: --prompts DeepCAD train parts (input drawn like scripts/sample_rft.py: view1 with a random camera / view4 /
view1+expert), --group programs each at --temp, reward = GT IoU (scripts/cq_iou.score, GT normalisation) or
--invalid_reward, advantage = (r - group mean) / (group std + eps); groups without spread are dropped. One optimizer
step per batch on exactly the sampled tokens, so the policy is on-policy (importance ratio 1, no clipping needed).
Loss = -mean over sequences of A * mean token log-prob. No KL term.
Validation = the 256 val parts of train_vlm_cq.py, greedy, through vLLM (view1 = render 0, view4, view1+expert).
A weight-sync check compares vLLM's and the HF model's log-prob of one fixed program before and after step 1.

  python train_vlm_grpo.py --data data --init results/vlm_rft_v1/model --name vlm_grpo_v1
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from mm.sysutil import usable_cpus  # noqa: E402

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")      # FlashInfer JIT-compiles its sampler: needs nvcc
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # engine in this process, so weights can be loaded
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from mm.vlm_cad import VLMCad, _messages, compose_views, configure, encode_batch, save_config  # noqa: E402
from train_vlm_cad import load_views  # noqa: E402
from train_vlm_cq import gt_vec  # noqa: E402


def vllm_model(llm):
    """The (unwrapped) torch model inside an in-process vLLM engine; attribute paths differ between versions."""
    paths = ["llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner",
             "llm_engine.model_executor.driver_worker.worker.model_runner",
             "llm_engine.model_executor.driver_worker.model_runner",
             "llm_engine.engine_core.model_executor.driver_worker.worker.model_runner"]
    for p in paths:
        obj = llm
        try:
            for attr in p.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        model = obj.get_model() if hasattr(obj, "get_model") else obj.model
        print(f"[vllm] model at {p}: {type(model).__name__}", flush=True)
        return model
    raise RuntimeError("cannot find the vLLM model runner; is VLLM_ENABLE_V1_MULTIPROCESSING=0?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--init", default="results/vlm_rft_v1/model")
    ap.add_argument("--name", default="vlm_grpo")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--deepcad_cq", default="data/deepcad_cq.parquet")
    ap.add_argument("--modes", default="view1=0.4,view4=0.4,view1+expert=0.2")
    ap.add_argument("--rich", nargs="*", default=None,
                    help="part-shard dirs (scripts/gen_rich_data.py, scripts/prep_real.py) to mix into the prompts")
    ap.add_argument("--rich_w", type=float, nargs="*", default=None, help="relative weight of each --rich dir")
    ap.add_argument("--rich_p", type=float, default=0.5, help="share of prompts drawn from --rich")
    ap.add_argument("--rich_max", type=int, default=300000, help="rich parts to sample prompts from")
    ap.add_argument("--rich_val", type=int, default=128, help="rich parts held out for validation")
    ap.add_argument("--prompts", type=int, default=32)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--invalid_reward", type=float, default=-0.5)
    ap.add_argument("--detail_w", type=float, default=0.0,
                    help="rich/real parts: reward = (1 - w) * own IoU + w * detail score (mm/detail.py)")
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--eval_n", type=int, default=256)
    ap.add_argument("--gpu_mem", type=float, default=0.3, help="vLLM share of GPU memory; the rest trains")
    ap.add_argument("--workers", type=int, default=max(1, usable_cpus() - 3))
    ap.add_argument("--score_timeout", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    from pebble import ProcessPool
    from scripts.cq_iou import score
    from scripts.rich_iou import score as rich_score
    pool = ProcessPool(max_workers=a.workers)            # before CUDA: workers fork clean
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    out = os.path.join(a.out_dir, a.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)
    log = open(os.path.join(out, "log.jsonl"), "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    render_dir = os.path.join(a.data, "renders")
    with open(os.path.join(a.data, "text_index.json")) as f:
        text_index = json.load(f)
    expert = {}
    dc = pd.read_parquet(a.deepcad_cq)
    with open(os.path.join(render_dir, "status.json")) as f:
        rendered = {u for u, st in json.load(f).items() if st in ("ok", "skip")}
    dc = dc[dc.ok & dc.uid.isin(rendered)]
    has = lambda u: os.path.exists(os.path.join(a.data, "cad_vec", u + ".h5")) and \
        os.path.exists(os.path.join(render_dir, f"{u}_3.png"))  # noqa: E731
    train = [u for u in dc[dc.split == "train"].drop_duplicates("code").uid if has(u)]
    val = dc[(dc.split == "validation") & (dc.iou >= 0.95) & (dc.iou <= 1.001) & (dc.code.str.len() <= 2500)]
    val = [u for u in val.uid if text_index.get(u, {}).get("expert") and has(u)][: a.eval_n]
    for u in train + val:
        if text_index.get(u, {}).get("expert"):
            expert[u] = text_index[u]["expert"]
    del text_index
    rich, rich_train, rich_valk = {}, [], []          # "rich:<j>:<shard>:<uid>" -> (render stem, GT CadQuery code)
    rich_by_dir, val_by_dir = [], []
    for j, rdir in enumerate(a.rich or []):
        part = {}
        for sh in sorted(os.listdir(rdir)):
            pq = os.path.join(rdir, sh, "parts.parquet")
            if os.path.exists(pq):
                d = pd.read_parquet(pq)
                d = d[d.code.str.len() <= 2400]
                for u, c in zip(d.uid, d.code):
                    part[f"rich:{j}:{sh}:{u}"] = (os.path.join(rdir, sh, "renders", u), c)
        keys = sorted(part)
        random.Random(a.seed + 1 + j).shuffle(keys)
        vk, tk = keys[: a.rich_val], keys[a.rich_val: a.rich_val + a.rich_max]
        rich.update({k: part[k] for k in vk + tk})
        rich_by_dir.append(tk)
        val_by_dir.append((os.path.basename(rdir.rstrip("/")), vk))
        rich_train += tk
        rich_valk += vk
        print(f"[data] rich {rdir}: {len(tk)} prompt parts, {len(vk)} val", flush=True)
    rich_w = np.array(a.rich_w if a.rich_w else [1.0] * len(rich_by_dir), float)
    rich_w = rich_w / rich_w.sum() if len(rich_w) else rich_w
    if rich_by_dir:
        print(f"[data] rich share {a.rich_p}, dir weights {rich_w.round(3).tolist()}", flush=True)
    gts = {}

    def gt(u):
        if u not in gts:
            gts[u] = gt_vec(a.data, u)
        return gts[u]

    modes = {k: float(v) for k, v in (kv.split("=") for kv in a.modes.split(","))}
    names, mp = list(modes), np.cumsum(np.array(list(modes.values())) / sum(modes.values()))
    print(f"[data] {len(train)} train parts, {len(val)} val; modes {modes}; grid {configure(a.init)} px", flush=True)

    # --- models: vLLM for sampling (bf16, in-process), HF policy for gradients (fp32 master weights)
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(a.init)
    llm = LLM(model=a.init, dtype="bfloat16", max_model_len=a.max_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, generation_config="vllm", seed=a.seed)
    vmodel = vllm_model(llm)
    tok = processor.tokenizer
    stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
    max_prompt = a.max_len - a.max_new_tokens - 1000
    vlm = VLMCad.from_pretrained(a.init, dtype=torch.float32)
    vlm.model.to("cuda")
    params = [p for p in vlm.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))

    def inputs(u, mode, val_mode=False):
        if u.startswith("rich:"):                     # rich parts: images only
            mode = "view4" if mode == "view4" else "view1"
            views = [0, 1, 2, 3] if mode == "view4" else ([0] if val_mode else [rng.randrange(4)])
            return compose_views(load_views("", rich[u][0], views)), None
        prompt = expert.get(u) if "expert" in mode else None
        if prompt and len(tok(prompt)["input_ids"]) > max_prompt:
            prompt = None
        if "expert" in mode and not prompt:
            mode = "view1"
        views = [0, 1, 2, 3] if mode == "view4" else ([0] if val_mode else [rng.randrange(4)])
        image = compose_views(load_views("", os.path.join(render_dir, u), views))
        return image, prompt

    def request(image, prompt, suffix=""):
        text = processor.apply_chat_template(_messages(image, prompt), tokenize=False, add_generation_prompt=True)
        return {"prompt": text + suffix, "multi_modal_data": {"image": image}}

    def scores(codes, uids):
        futs = [pool.schedule(rich_score, args=((rich[u][1], c),), timeout=a.score_timeout) if u.startswith("rich:")
                else pool.schedule(score, args=((gt(u), c, "cadquery"),), timeout=a.score_timeout)
                for c, u in zip(codes, uids)]
        res = []
        for fu in futs:
            try:
                res.append(fu.result())
            except Exception:  # noqa: BLE001 - timeout / crash -> invalid
                res.append({"valid": False, "gt": 0.0, "own": 0.0, "failed": True})
        return res

    def sync():
        vmodel.load_weights(((k, v.detach().to(torch.bfloat16)) for k, v in vlm.model.state_dict().items()))
        llm.reset_prefix_cache()

    def sync_check(tag, u, code):
        """Sum log-prob of `code` given part u's view4 input, by vLLM (prompt_logprobs) and by the HF policy."""
        image, _ = inputs(u, "view4", True)
        o = llm.generate([request(image, None, code + "<|im_end|>")],
                         SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)[0]
        n = len(tok(code + "<|im_end|>", add_special_tokens=False)["input_ids"])
        ids = o.prompt_token_ids[-n:]
        lv = sum(d[t].logprob for d, t in zip(o.prompt_logprobs[-n:], ids))
        enc = encode_batch(vlm.processor, [image], [None], [code]).to("cuda")
        labels = enc.pop("labels")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = vlm.model(**enc).logits[:, :-1].float()
        tgt = labels[:, 1:]
        lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0)[..., None]).squeeze(-1)
        lh = float((lp * (tgt != -100)).sum())
        emit({"sync_check": tag, "vllm_logp": round(lv, 3), "hf_logp": round(lh, 3)})

    def evaluate(step):
        t0 = time.time()
        rec = {"step": step}
        greedy = SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop)
        for mode in ("view1", "view4", "view1+expert"):
            reqs = [request(*inputs(u, mode, True)) for u in val]
            codes = [o.outputs[0].text.strip() or None for o in llm.generate(reqs, greedy, use_tqdm=False)]
            rs = [r for r in scores(codes, val) if r is not None]
            rec[mode] = {"iou_gt": round(100 * float(np.mean([r["gt"] for r in rs])), 2),
                         "invalid": round(float(np.mean([not r["valid"] for r in rs])), 4)}
        for name, vk in val_by_dir:                   # held-out parts of each --rich dir, 4 views, own IoU
            reqs = [request(*inputs(u, "view4", True)) for u in vk]
            codes = [o.outputs[0].text.strip() or None for o in llm.generate(reqs, greedy, use_tqdm=False)]
            rs = [r for r in scores(codes, vk) if r is not None]
            key = "rich_view4" if name == "train" else f"{name}_view4"
            rec[key] = {"iou_own": round(100 * float(np.mean([r["own"] for r in rs])), 2),
                        "detail": round(100 * float(np.mean([r.get("detail", 0.0) for r in rs])), 2),
                        "invalid": round(float(np.mean([not r["valid"] for r in rs])), 4)}
        rec["eval_sec"] = round(time.time() - t0)
        emit(rec)

    def save(step):
        d = os.path.join(out, "model")
        sd = {k: v.to(torch.bfloat16) for k, v in vlm.model.state_dict().items()}
        vlm.model.save_pretrained(d, state_dict=sd, safe_serialization=True)
        vlm.processor.save_pretrained(d)
        save_config(d)
        with open(os.path.join(d, "step.json"), "w") as f:
            json.dump({"step": step, "args": vars(a)}, f)

    check_uid = val[0]
    check_code = llm.generate([request(*inputs(check_uid, "view4", True))],
                              SamplingParams(temperature=0.0, max_tokens=a.max_new_tokens, stop_token_ids=stop),
                              use_tqdm=False)[0].outputs[0].text
    sync_check("before training (should match)", check_uid, check_code)
    evaluate(0)
    sp = SamplingParams(n=a.group, temperature=a.temp, top_p=1.0, max_tokens=a.max_new_tokens, stop_token_ids=stop)
    for step in range(1, a.steps + 1):
        t0 = time.time()
        n_rich = sum(rng.random() < a.rich_p for _ in range(a.prompts)) if rich_train else 0
        dirs = rng.choices(range(len(rich_by_dir)), weights=rich_w, k=n_rich) if n_rich else []
        uids = rng.sample(train, a.prompts - n_rich) + [rng.choice(rich_by_dir[j]) for j in dirs]
        items = [inputs(u, names[int(np.searchsorted(mp, rng.random()))]) for u in uids]
        outs = llm.generate([request(im, p) for im, p in items], sp, use_tqdm=False)
        seqs = [list(c.token_ids) for o in outs for c in o.outputs]
        codes = [c.text.strip() or None for o in outs for c in o.outputs]
        t_gen = time.time() - t0
        rs = scores(codes, [u for u in uids for _ in range(a.group)])
        t_score = time.time() - t0 - t_gen
        rew_uids = [u for u in uids for _ in range(a.group)]
        R = np.array([np.nan if r is None else (((1 - a.detail_w) * r["own"] + a.detail_w * r.get("detail", 0.0)
                                                  if u.startswith("rich:") else r["gt"]) if r["valid"]
                      else a.invalid_reward) for r, u in zip(rs, rew_uids)])   # rich: size unobservable -> own IoU
        R = R.reshape(a.prompts, a.group)
        std = np.nanstd(R, axis=1, keepdims=True)
        adv = (R - np.nanmean(R, axis=1, keepdims=True)) / (std + 1e-4)
        keep = [i for i in range(len(seqs)) if np.isfinite(adv.flat[i]) and std.flat[i // a.group] > 1e-3 and seqs[i]]

        vlm.model.train()
        vlm.model.config.use_cache = False
        vlm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        loss_sum = 0.0
        for s in range(0, len(keep), a.micro):
            idx = keep[s: s + a.micro]
            enc = encode_batch(vlm.processor, [items[i // a.group][0] for i in idx],
                               [items[i // a.group][1] for i in idx])
            P = enc["input_ids"].shape[1]
            L = max(len(seqs[i]) for i in idx)
            gen_ids = torch.full((len(idx), L), tok.pad_token_id, dtype=torch.long)
            gen_mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                gen_ids[j, : len(seqs[i])] = torch.tensor(seqs[i])
                gen_mask[j, : len(seqs[i])] = 1
            for k, v in list(enc.items()):            # per-token tensors (mm_token_type_ids): text = 0
                if torch.is_tensor(v) and v.dim() == 2 and v.shape == (len(idx), P) and k != "input_ids":
                    enc[k] = torch.cat([v, gen_mask if k == "attention_mask" else torch.zeros_like(gen_ids)], 1)
            enc["input_ids"] = torch.cat([enc["input_ids"], gen_ids], 1)
            enc = enc.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = vlm.model(**enc).logits[:, P - 1: -1].float()
            lp = torch.log_softmax(logits, -1).gather(-1, enc["input_ids"][:, P:, None]).squeeze(-1)
            m = gen_mask.to(lp.device).float()
            A = torch.tensor([adv.flat[i] for i in idx], dtype=torch.float32, device=lp.device)
            loss = -(A * (lp * m).sum(1) / m.sum(1)).sum() / max(len(keep), 1)
            loss.backward()
            loss_sum += loss.item()
            del logits, lp
        vlm.model.gradient_checkpointing_disable()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if keep else 0.0
        if keep:
            opt.step()
            sched.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        sync()
        if step == 1:
            sync_check("after step 1 (should match, and differ from before)", check_uid, check_code)
        valid = [r for r in rs if r is not None]
        emit({"step": step, "reward": round(float(np.nanmean(R)), 4),
              "iou_valid": round(float(np.mean([r["gt"] for r in valid if r["valid"]] or [0])), 4),
              "invalid": round(float(np.mean([not r["valid"] for r in valid])), 4), "kept": len(keep),
              "len": round(float(np.mean([len(s) for s in seqs])), 1), "loss": round(loss_sum, 5),
              "gnorm": round(gnorm, 3), "lr": sched.get_last_lr()[0], "t_gen": round(t_gen, 1),
              "t_score": round(t_score, 1), "sec": round(time.time() - t0, 1)})
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
        if step % a.save_every == 0 or step == a.steps:
            save(step)
    pool.close()
    print(f"[done] {out}/model", flush=True)


if __name__ == "__main__":
    sys.exit(main())
