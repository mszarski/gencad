# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`gencad/`: GenCAD replication + text/image conditioning for CAD generation (image/text -> CAD program).
Self-contained; run `gencad/setup.sh` first. It started as a replication of GenCAD
(https://github.com/ferdous-alam/GenCAD) and moved to a fine-tuned VLM (Qwen3-VL-2B) that writes **CadQuery code**
from renders and/or prompts, scored by volume IoU against the ground-truth solid.

Read these before non-trivial work (all in `gencad/`):
- `CLAUDE.md`: the detailed lab notebook: every experiment, results tables, current best model, the latest
  **Handoff** section, next ideas and a **Gotchas** list. Long (~800 lines); grep for the section you need.
- `REPORT.md`: narrative project report. `MULTIMODAL.md`: original design and plan.
- `datagen/README.md`: rules for procedural part families (emitted code must be literal CadQuery).

`trajectory_classifier/` at the repo root is unrelated leftover from the monorepo this was split from; ignore it.

## Setup and tests (run everything from `gencad/`)

```bash
cd gencad
pip install torch torchvision -r jobs/requirements-job.txt   # cadquery pebble h5py trimesh pyarrow for local scoring
./setup.sh                                  # clones upstream GenCAD at a pinned commit, applies patches/, symlinks it in
python -m pytest tests/test_mm_smoke.py -q  # CPU smoke tests (~1-2 min); MM_TEST_ENCODERS=1 also downloads real encoders
```

Upstream GenCAD has no license, so it is **never committed**: `setup.sh` puts it in `gencad/upstream/` and symlinks
`cadlib/ model/ config/ utils/ trainer/ data/ results/ train_gencad.py ...` into `gencad/` (all gitignored). To change
upstream code, edit it in `upstream/`, then `git -C upstream diff > patches/<name>.patch` and make `setup.sh` apply it.
Our own code: `mm/` (models, data, metrics, CadQuery I/O), `scripts/` (eval, scoring, baselines, data prep),
`datagen/`, `jobs/`, `tests/`, and the top-level `train_*.py`, `eval_mm.py`, `infer_mm.py`.

## Compute: Hugging Face Jobs + bucket

GPU work runs as HF Jobs, not locally:
```bash
export HF_TOKEN=...                         # needs Jobs + write access
RUN=<name> jobs/launch.sh <stage> [args]    # e.g. vllmpred, vlmcq, vlmgrpo, cqiou; DRY=1 prints the command
hf jobs logs <id> --follow                  # or: hf jobs inspect <id> --format json
```
- `jobs/launch.sh` defines the stage list and per-stage hardware/timeouts; `jobs/entry.sh` is the in-job dispatcher
  (one `case` arm per stage). A new stage needs an arm in both.
- Data and results live in the bucket `mszarski/gencad-work`, mounted at `/work` in jobs
  (`results/<run>/...`, `data/...`). Pull with `hf buckets cp hf://buckets/mszarski/gencad-work/<path> <local>`.
- The job gets a dereferenced snapshot of `gencad/` (symlinks followed; `upstream/`, `results/`, big `data/` dirs
  excluded). Anything else under local `data/` (e.g. downloaded `.tar`s) gets uploaded too: move it aside, but leave a
  placeholder file, since an empty `data/` breaks `entry.sh`.
- Env knobs are forwarded only if listed in the `ENVS` loop in `launch.sh` (RUN, INIT, STEPS, PRIOR, VLLM_NIGHTLY, ...).

## Evaluation conventions
- Test predictions: `jobs/launch.sh vllmpred` (~5 min), then score **locally** with `scripts/cq_iou.py --rows`
  (`scripts/rich_iou.py` for rich/real parts). Cloud `cqiou` jobs lost parts to worker timeouts; if `cq_iou.py`
  reports `timeout/crash` not ~0, rerun with fewer `--workers`.
- Report dup vs novel test splits (38 % of DeepCAD test programs duplicate training ones); novel is the headline.
  Compare models with paired differences on the same parts (rows files).
- "gt" vs "own" IoU normalisation: prefer gt when they disagree ("own" can collapse for a correct solid).
- GenCAD's 5 renders per uid are not 5 views; `image_ids.json` is shuffled and only file `_0` shows the GT program.
  Use our fixed-camera renders (`scripts/render_views.py`) or `--true_view`.

## Gotchas
- Claude API key is stored as `GENCAD_API_KEY`; run `ANTHROPIC_API_KEY=$GENCAD_API_KEY python scripts/claude_baseline.py ...`.
- Text2CAD is gated (CC BY-NC-SA 4.0): use only `text2cad_v1.1/*.csv`, never commit its data. CAD-Recode is CC BY-NC 4.0.
- Job image needs torch>=2.6; upstream training code hard-codes CUDA and its `configAE.py` defaults are debug-sized.
- Upstream `utils/image_processor.process_image` naming is inverted (`img_type="image"` applies Canny).
- More in the Gotchas section of `gencad/CLAUDE.md`.

## Working conventions
- Commit messages are prefixed `gencad: ...` and usually name the result or feature.
- When an experiment finishes, record it in `gencad/CLAUDE.md` (results table + readings) and update its Handoff
  section; keep `REPORT.md` in step for headline changes. Note job cost against the budget in the Handoff.
- `.gitignore` drops `*.png` repo-wide; galleries and grids go to the bucket, not git.
