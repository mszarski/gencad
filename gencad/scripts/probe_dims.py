"""Zero-shot dimension-reading probe: can a base VLM read the numbers on an engineering drawing?

Each model (vLLM) is asked to list every dimension on each drawing of --data (made by scripts/make_drawing.py, whose
dims.json holds the true values) at several image sizes; recall = share of true dimension values found in the answer
(+-0.05), precision = share of the numbers it lists that are true dimension values. Drawings in --extra (e.g.
CADGenBench sheets) get their answers saved for inspection only.

  python scripts/probe_dims.py --data data/dimprobe --extra data/cgb_sheets --models Qwen/Qwen3-VL-2B-Instruct \
      --sides 3100 1600 --out results/<run>/probe.json
"""
import argparse
import gc
import json
import os
import re

import numpy as np

PROMPT = ("This is a mechanical engineering drawing. List every dimension written on it, one per line, as "
          "'<value> : <what it measures>' (for example '12.5 : overall width', 'Ø6.6 : hole diameter', "
          "'R2 : fillet radius'). Include every number that is a dimension; do not list title-block text.")
NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)")


def numbers(text):
    out = []
    for line in text.splitlines():
        head = line.split(":")[0]
        out += [float(x) for x in NUM.findall(head)]
    return out


def score(pred, true, tol=0.051):
    T, P = list(true), list(pred)
    hit = 0
    for t in T:
        j = next((j for j, p in enumerate(P) if abs(p - t) <= tol), None)
        if j is not None:
            hit += 1
            P.pop(j)
    return hit, len(T), len(pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sides", type=int, nargs="+", default=[3100, 1600], help="longest image side in px")
    ap.add_argument("--max_tokens", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    dims = json.load(open(os.path.join(a.data, "dims.json")))
    items = [(n, os.path.join(a.data, f"{n}.png"), [round(d["value"], 1) for d in v]) for n, v in dims.items()]
    extra = [(os.path.splitext(f)[0], os.path.join(a.extra, f), None) for f in sorted(os.listdir(a.extra))] \
        if a.extra else []
    res = {}
    for m in a.models:
        big = max(a.sides)
        llm = LLM(model=m, max_model_len=12288, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85,
                  mm_processor_kwargs={"max_pixels": big * big}, trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
        proc = AutoProcessor.from_pretrained(m)
        text = proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                                        tokenize=False, add_generation_prompt=True)
        for side in a.sides:
            reqs, keys = [], []
            for name, path, true in items + extra:
                im = Image.open(path).convert("RGB")
                s = side / max(im.size)
                if s < 1:
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                reqs.append({"prompt": text, "multi_modal_data": {"image": im}})
                keys.append((name, true))
            outs = llm.generate(reqs, sp, use_tqdm=False)
            hits = tot = npred = 0
            per = {}
            for (name, true), o in zip(keys, outs):
                txt = o.outputs[0].text
                per[name] = txt
                if true is None:
                    continue
                h, t, p = score(numbers(txt), true)
                hits, tot, npred = hits + h, tot + t, npred + p
            r = {"recall": round(hits / max(tot, 1), 3), "precision": round(hits / max(npred, 1), 3),
                 "n_true": tot, "n_pred": npred, "answers": per}
            res[f"{m}@{side}"] = r
            print(f"[probe] {m} @{side}px: recall {r['recall']} precision {r['precision']} ({tot} true dims)",
                  flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
