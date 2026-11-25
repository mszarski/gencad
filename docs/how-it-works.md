# How it works

## Model

`Qwen/Qwen3-VL-2B-Instruct`, fully fine-tuned. Inputs (`gencad/mm/vlm_cad.py`):
- **Renders**: one view (448 px) or a 2x2 grid of four fixed cameras at full resolution (896 px grid; iso (1,1,1),
  iso (-1,-1,-1), front (0,-1,0), top (0,0,1); `scripts/render_views.py`), optionally with a text description.
- **Drawings**: one engineering drawing sheet, fed at up to 2048 px (~2,900 image tokens; the processor's pixel
  budget is raised to allow it).

Output: a CadQuery program (`r = ...`). Programs are executed with `mm.cadquery_io.run_cadquery`, which adds a
`near()` edge selector (CadQuery's own nearest-point selector compares edge centres, so circular edges could not be
picked).

## Training data

| source | size | what it teaches |
|---|---|---|
| DeepCAD, converted to CadQuery (`scripts/convert_deepcad.py`) | ~100k programs | sketch-and-extrude parts |
| CAD-Recode v1.5 | ~1M programs | reading geometry from renders (the biggest single gain) |
| `datagen/` procedural families | ~1.1M parts, 21 families | fillets, chamfers, holes, revolves, shells, patterns |
| Fusion 360 Gallery real parts (`scripts/prep_real.py`) | 14k train / 600 test | real design habits |
| Self-improvement targets (`scripts/build_self_targets.py`) | 7k | the model's own best verified programs on real parts |
| Synthetic drawings (`scripts/make_drawing.py`) | ~83k (style v3) | drawing → program in mm |

## Training recipe

1. **SFT** (`train_vlm_cq.py`): image (or drawing) → program, mixed sources.
2. **Online RL** (`train_vlm_grpo.py`): GRPO with vLLM in the same process (weights synced after every step), 8
   samples per part at temperature 1, reward = IoU (plus a detail score on real parts; invalid code -0.5). Online RL
   beat rejection-sampling fine-tuning, DPO and HF-generate RL.
3. Supervised training after RL erodes the RL gains, so every SFT round is followed by an RL top-up.

## Inference

- **One answer**: greedy decoding with vLLM (`scripts/predict_vllm.py`).
- **Best-of-N** (`scripts/pick_render.py`): greedy + 8 samples, keep the one whose render best matches the input
  views (no ground truth). +5 own IoU on real parts.
- **Full pipeline**: answer → orientation search over 24 right-angle rotations (`scripts/orient_fix.py`) → one
  revision turn that sees the render of its own program or the build error (`scripts/revise_vllm.py`) → orient again
  → keep the revision only if its render matches the input better (`scripts/revise_guard.py`).
- **Drawings**: greedy, or best-of-8 with a consensus picker (`scripts/bon_drawings.py`: the candidate that agrees
  most with the others).

## Compute

Training and evaluation run on Hugging Face Jobs (`jobs/launch.sh`, `jobs/entry.sh`; H200 for training, A100 for
evaluation, CPU jobs for data and scoring). See [getting-started.md](getting-started.md).
