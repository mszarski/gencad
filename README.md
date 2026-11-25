# gencad: pictures and drawings of a part → CadQuery code, with a local 2B model

A small vision-language model (Qwen3-VL-2B, fine-tuned) that looks at **renders** or an **engineering drawing** of a
mechanical part and writes an executable **[CadQuery](https://github.com/CadQuery/cadquery) program** that builds the
part as a solid. It runs on one GPU; nothing at inference time depends on a cloud API.

The project started as a replication of [GenCAD](https://arxiv.org/abs/2409.16294) (image → CAD latent → program).
That design plateaued at ~11 % volume IoU on unseen parts, so we switched to a model that writes code directly,
trained on ~1M procedural programs and then post-trained with online reinforcement learning on IoU.

## Where it stands (Nov 2026)

| task | input | result | reference |
|---|---|---|---|
| DeepCAD test parts | 4 renders | **91.4 IoU** (88.4 on unseen parts), one answer | GenCAD ~22; Claude Opus 5.5 zero-shot 83.4; cadrille + RL 92.2 |
| Real designs (Fusion 360 Gallery, 600 parts) | 4 renders | **71.9 own IoU** with best-of-9 picking (66.5 one answer) | Claude Opus 5.5 zero-shot 66.5 |
| Procedural parts with real features (fillets, holes, revolves…) | 4 renders | **~82-84 own IoU** | 55 before training on them |
| Engineering drawings → part in millimetres | 1 drawing sheet | shapes right (~75 own IoU) but **absolute size not read yet**; fix in training now | [CADGenBench](https://github.com/huggingface/cadgenbench) is the target |

IoU = volume intersection over union with the true part, in %, invalid code = 0. "own" fits each solid to its own
box (size ignored). Details, caveats and paired comparisons: [docs/results.md](docs/results.md).

## How it works, briefly

1. **Data:** DeepCAD (~100k programs), CAD-Recode (~1M procedural CadQuery programs), 21 of our own procedural part
   families (~1.1M parts with fillets, chamfers, holes, revolves, shells, patterns), 14k real Fusion 360 Gallery
   parts, and ~83k synthetic dimensioned drawings made with [SolidCraft](https://github.com/tangency-labs/solidcraft).
2. **Training:** supervised fine-tuning on (image → program), then online GRPO with vLLM in the loop and IoU as the
   reward; a "look at what you built and fix it" revision turn.
3. **Inference:** one greedy answer, or best-of-N with a picker that compares each candidate's render with the input
   (no ground truth needed); optional orientation search and a guarded revision turn.

More: [docs/how-it-works.md](docs/how-it-works.md). Drawings and CADGenBench: [docs/drawings.md](docs/drawings.md).

## Quick start

```bash
cd gencad
pip install torch torchvision -r jobs/requirements-job.txt   # + cadquery pebble h5py trimesh pyarrow
./setup.sh                                                   # fetches upstream GenCAD (not redistributed here)
python -m pytest tests/test_mm_smoke.py -q                   # CPU smoke tests, ~1-2 min
```

Running models, evaluating, and launching training on Hugging Face Jobs: [docs/getting-started.md](docs/getting-started.md).

## Repository layout

| path | what |
|---|---|
| `gencad/mm/` | model wrapper (`vlm_cad.py`: prompts, view grid, drawings), CadQuery execution and IoU (`cadquery_io.py`), scoring helpers |
| `gencad/train_vlm_cq.py`, `gencad/train_vlm_grpo.py` | supervised fine-tuning; online RL (GRPO) |
| `gencad/scripts/` | data building, prediction (vLLM), scoring, best-of-N picking, drawings, galleries |
| `gencad/datagen/` | procedural part families and their checks ([spec](gencad/datagen/README.md)) |
| `gencad/jobs/` | Hugging Face Jobs launcher and in-job stages |
| `gencad/CLAUDE.md` | the lab notebook: every experiment with commands and numbers |
| `gencad/REPORT.md` | narrative report; `gencad/PLAN_drawings.md`: the drawings plan |
| `HANDOFF.md`, `CLAUDE.md` | current state and working rules for AI coding sessions |

## Data, weights and licences

- Model weights and datasets live in a private Hugging Face bucket, not in this repo.
- Upstream GenCAD has no licence, so it is never committed: `setup.sh` clones it at a pinned commit and applies our
  patches locally.
- CAD-Recode is CC BY-NC 4.0 and Text2CAD is CC BY-NC-SA 4.0 (only its prompt CSVs are used, never committed): models
  trained on them inherit non-commercial terms.
- The Fusion 360 Gallery data has its own licence terms; we use it for training and evaluation only.
