"""Image(s) and/or text -> CAD program with a pretrained vision-language model, fine-tuned end to end.

Follows cadrille (arXiv 2505.22914): instead of a small decoder trained from scratch (which mostly memorised the
duplicated DeepCAD training set, see CLAUDE.md), a pretrained VLM (Qwen3-VL-2B-Instruct) reads the render(s) and the
prompt natively and writes the program as plain text, one command per line, every number a 0-255 quantised value:

  S                          start of a sketch loop
  L x y                      line to (x, y)
  A x y alpha f              arc to (x, y), sweep angle alpha, flag f
  R x y r                    circle, centre (x, y), radius r
  E theta phi gamma px py pz s e1 e2 b u   extrude the sketch so far (plane, origin, scale, extents, op, type)

Inputs: one view, or up to four views pasted into a 2x2 grid (each 224 px, so the grid is 448 px like a single
view), and an optional text prompt. Nothing about the CAD format is built into the model; the text grammar above is
parsed back into DeepCAD's (L, 17) vectors by `text_to_program`.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from cadlib.macro import (ARC_IDX, CIRCLE_IDX, CMD_ARGS_MASK, EOS_IDX, EXT_IDX, LINE_IDX, N_ARGS, PAD_VAL,
                          SOL_IDX)

LETTER = {LINE_IDX: "L", ARC_IDX: "A", CIRCLE_IDX: "R", SOL_IDX: "S", EXT_IDX: "E"}
CMD = {v: k for k, v in LETTER.items()}
USED = {c: np.nonzero(CMD_ARGS_MASK[c])[0] for c in LETTER}          # which of the 16 args each command uses
SIDE = 448                                                           # model input image side (14 x 32 px patches)
INSTRUCTION = "Write the CAD program for this part."
CFG_FILE = "gencad_vlm.json"          # per-model input settings, saved next to the weights
GRID_SIDE = SIDE                      # side of the 2-4 view grid; 2 * SIDE keeps every view at full resolution


def configure(path=None, grid_side=None):
    """Set the multi-view grid side: explicitly, or from <path>/gencad_vlm.json (absent = SIDE, the original
    layout). Every loader (HF or vLLM) calls this so a model is always fed the layout it was trained on."""
    global GRID_SIDE
    if grid_side is None and path is not None:
        import json
        import os
        f = os.path.join(path, CFG_FILE)
        grid_side = json.load(open(f)).get("grid_side", SIDE) if os.path.exists(f) else SIDE
    if grid_side is not None:
        GRID_SIDE = int(grid_side)
    return GRID_SIDE


def save_config(path):
    import json
    import os
    with open(os.path.join(path, CFG_FILE), "w") as f:
        json.dump({"grid_side": GRID_SIDE}, f)


def program_to_text(vec) -> str:
    lines = []
    for row in np.asarray(vec):
        c = int(row[0])
        if c == EOS_IDX:
            break
        args = row[1:][USED[c]]
        lines.append(" ".join([LETTER[c], *(str(int(x)) for x in args)]))
    return "\n".join(lines)


def text_to_program(text: str):
    """-> (L, 17) int array (no EOS row, like the other decoders' output), or None if the text is malformed."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] not in CMD:
            return None
        c = CMD[parts[0]]
        try:
            args = [int(x) for x in parts[1:]]
        except ValueError:
            return None
        if len(args) != len(USED[c]) or any(not 0 <= x <= 255 for x in args):
            return None
        row = np.full(1 + N_ARGS, PAD_VAL, dtype=np.int64)
        row[0] = c
        row[1 + USED[c]] = args
        rows.append(row)
    if not rows:
        return None
    return np.stack(rows)


def compose_views(images, side=SIDE, grid_side=None):
    """List of PIL renders -> one RGB image: a single view resized to side x side, or 2-4 views in a 2x2 grid of
    grid_side (default: the configured GRID_SIDE) with unused cells white."""
    images = [im.convert("RGB") for im in images]
    if len(images) == 1:
        return images[0].resize((side, side), Image.BICUBIC)
    gs = grid_side or GRID_SIDE
    cell = gs // 2
    grid = Image.new("RGB", (gs, gs), (255, 255, 255))
    for k, im in enumerate(images[:4]):
        grid.paste(im.resize((cell, cell), Image.BICUBIC), ((k % 2) * cell, (k // 2) * cell))
    return grid


def _messages(image, prompt):
    """image: None, one PIL image, or a list of them (input first; a revision turn adds the attempt's render)."""
    images = [] if image is None else (image if isinstance(image, list) else [image])
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt if prompt else INSTRUCTION})
    return [{"role": "user", "content": content}]


def revision_text(prompt, attempt, error=None, features=None):
    """Prompt of a revision turn (build-inspect-fix): the task, the previous program, and what happened when it was
    built: the build error, or (error None) its render, given as the second image with the same cameras as the input,
    plus optionally a feature summary of what it builds (mm.features.summary: holes, fillets, chamfers, ...)."""
    head = (prompt + "\n\n") if prompt else INSTRUCTION + "\n\n"
    if error:
        result = f"It failed to build: {error}"
    else:
        result = "The second image shows what it builds, rendered with the same cameras as the first image."
        if features:
            result += f"\nFeature check of what it builds: {features}."
    return (f"{head}A previous attempt:\n{attempt}\n\n{result}\n"
            "Write the corrected program (or repeat it if it is already right).")


def encode_batch(processor, images, prompts, targets=None):
    """images: list of PIL or None; prompts: list of str or None; targets: list of program texts (training) or None
    (generation). Returns processor tensors (left padded), with `labels` masking everything but the target.
    A plain function so DataLoader workers can run it."""
    tok = processor.tokenizer
    texts, ims = [], []
    for k, (im, p) in enumerate(zip(images, prompts)):
        s = processor.apply_chat_template(_messages(im, p), tokenize=False, add_generation_prompt=True)
        if targets is not None:
            s += targets[k] + tok.eos_token             # "<|im_end|>" closes the answer
        texts.append(s)
        if im is not None:
            ims.extend(im if isinstance(im, list) else [im])
    enc = processor(text=texts, images=ims or None, return_tensors="pt", padding=True)
    if targets is not None:
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, t in enumerate(targets):                  # left padding: the answer is the last n tokens of row i
            n = len(tok(t + tok.eos_token, add_special_tokens=False)["input_ids"])
            labels[i, : labels.shape[1] - n] = -100
        enc["labels"] = labels
    return enc


def set_pixel_range(processor):
    """Images are fed at their own size: 448 px single views (196 visual tokens) and GRID_SIDE grids."""
    processor.image_processor.size = {"shortest_edge": SIDE * SIDE, "longest_edge": max(SIDE, GRID_SIDE) ** 2}


class VLMCad:
    """Thin wrapper: chat formatting, loss masking and generation around a Qwen-VL model + processor."""

    def __init__(self, model, processor):
        self.model, self.processor = model, processor
        self.tok = processor.tokenizer

    @classmethod
    def from_pretrained(cls, path, dtype=torch.float32, **kw):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(path)
        configure(path)
        set_pixel_range(processor)
        processor.tokenizer.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, **kw)
        return cls(model, processor)

    def set_grid_side(self, grid_side):
        configure(grid_side=grid_side)
        set_pixel_range(self.processor)

    def save(self, path):
        self.model.save_pretrained(path, safe_serialization=True)
        self.processor.save_pretrained(path)
        save_config(path)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def batch(self, images, prompts, targets=None):
        return encode_batch(self.processor, images, prompts, targets)

    def loss(self, images, prompts, targets):
        enc = self.batch(images, prompts, targets).to(self.device)
        return self.model(**enc).loss

    @torch.no_grad()
    def generate(self, images, prompts, max_new_tokens=1024, **kw):
        """-> list of (L, 17) programs or None (malformed text)."""
        enc = self.batch(images, prompts).to(self.device)
        kw.setdefault("do_sample", False)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id, **kw)
        texts = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [text_to_program(t) for t in texts], texts
