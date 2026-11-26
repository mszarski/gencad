"""Blend a fine-tuned model's weights with its base model's: w = (1 - alpha) * ours + alpha * base, per tensor.

Used to bring back skills fine-tuning erased (our render-to-code line lost the base Qwen3-VL's ability to read text
on drawings: 70 % -> 2 % of dimensions). Writes a full model dir with our config / processor / gencad_vlm.json.
`--parts` limits the blend to tensors whose name contains one of the given substrings (e.g. `visual` for the vision
tower only); the rest stay ours.

  python scripts/blend_weights.py --ours results/vlm_draw3/model --base Qwen/Qwen3-VL-2B-Instruct --alpha 0.3 \
      --out results/blend_a30/model
"""
import argparse
import glob
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--base", required=True, help="model dir or Hub id")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--parts", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from safetensors.torch import load_file, save_file
    base = a.base
    if not os.path.isdir(base):
        from huggingface_hub import snapshot_download
        base = snapshot_download(base, allow_patterns=["*.safetensors", "*.json"])
    B = {}
    for f in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        B.update(load_file(f))
    os.makedirs(a.out, exist_ok=True)
    n_mix = n_keep = n_missing = 0
    for f in sorted(glob.glob(os.path.join(a.ours, "*.safetensors"))):
        W = load_file(f)
        for k, v in W.items():
            kb = k if k in B else next((c for c in (k.replace("model.language_model.", "model."),
                                                    k.replace("model.visual.", "visual.")) if c in B), None)
            if kb is None or B[kb].shape != v.shape:
                n_missing += 1
                continue
            if a.parts and not any(p in k for p in a.parts):
                n_keep += 1
                continue
            W[k] = ((1 - a.alpha) * v.float() + a.alpha * B[kb].float()).to(v.dtype)
            n_mix += 1
        save_file(W, os.path.join(a.out, os.path.basename(f)), metadata={"format": "pt"})
    for f in os.listdir(a.ours):
        if not f.endswith(".safetensors") and not f.startswith(".tmp") and os.path.isfile(os.path.join(a.ours, f)):
            shutil.copy(os.path.join(a.ours, f), a.out)
    if n_mix == 0:
        raise SystemExit(f"[blend] no tensor matched the base model ({n_missing} unmatched): check the key names")
    print(f"[blend] alpha {a.alpha}: {n_mix} tensors blended, {n_keep} kept, {n_missing} without a base match",
          flush=True)
    del torch


if __name__ == "__main__":
    main()
