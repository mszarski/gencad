# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `HANDOFF.md` first** for current state, best model and latest numbers, jobs in flight, spend so far, history,
next steps and open questions. This file holds only the durable rules.

## Project

`gencad/`: images (optionally + text) of a mechanical part -> executable **CadQuery** program (valid B-rep solid),
running **locally on one GPU**. Started as a GenCAD replication (arXiv 2409.16294,
https://github.com/ferdous-alam/GenCAD); now a fine-tuned Qwen3-VL-2B (`Qwen/Qwen3-VL-2B-Instruct`) trained with
SFT then online RL (GRPO), scored by volume IoU.

Docs in `gencad/`: `CLAUDE.md` (detailed lab notebook, ~800 lines: every experiment, commands, numbers, more
Gotchas; grep for the section you need), `REPORT.md` (narrative summary), `MULTIMODAL.md` (original plan),
`datagen/README.md` (spec for procedural part families).

## Working with the user

- **Plain language, bottom line first.** They ask "so are we better overall?", "why is this worse?", "status".
  Answer with a short honest verdict and a small table against the *previous best* (paired, same parts), then the
  caveats. Don't oversell: if a gain on generated parts doesn't transfer to real parts, say so plainly.
- **Show, don't just tell.** They want to see outputs "with my eyes": make galleries (`scripts/gallery.py`,
  `scripts/gallery_rich.py`) and send them as images.
- **Budget is theirs to set.** They give explicit budgets. Give cost estimates up front, track spend, and **ask
  (options + a recommendation, via AskUserQuestion) before exceeding a budget or starting a large spend.** A few
  dollars of clearly in-scope scoring is fine without asking. Say so explicitly when an estimate was wrong.
- **They like ambition and volume**: "make ridiculous amounts of data and use yourself (Claude) to check it",
  "keep going until you've got something significant". Use sub-agents to author/audit generators, automated checks
  for volume.
- **Local model is the goal.** "I want to make the model good, not have a cloud connection unnecessarily." Cloud APIs
  (Claude etc.) are for data, teaching and reference, not a runtime dependency; confirm with them before proposing a
  Claude-in-the-loop product.
- **External links** (articles, model cards, repos) get a straight "would this make a difference to us?": read the
  primary source, separate claims from evidence, tie it to our bottleneck (data and feedback, not model size).
- **Status pings**: reply with where each job is, latest numbers, ETA, spend. If a watcher died or a job failed, say
  so and what you did about it.
- Process: commit and push work as you go; commit messages end with the harness's attribution lines; never put model
  identifiers in commits/PRs beyond those lines; **don't create PRs unless asked**.

## Setup and tests (run everything from `gencad/`)

```bash
cd gencad
pip install torch torchvision -r jobs/requirements-job.txt   # cadquery pebble h5py trimesh pyarrow for local scoring
./setup.sh                                  # clones upstream GenCAD at a pinned commit, applies patches/, symlinks it in
python -m pytest tests/test_mm_smoke.py -q  # CPU smoke tests (~1-2 min); MM_TEST_ENCODERS=1 also downloads real encoders
```

`.claude/hooks/session-start.sh` does the install + `setup.sh` automatically in remote (web) sessions.
Upstream GenCAD has no license, so it is **never committed**: `setup.sh` puts it in `gencad/upstream/` and symlinks
`cadlib/ model/ config/ utils/ trainer/ data/ results/ train_gencad.py ...` into `gencad/` (all gitignored). To change
upstream code, edit it in `upstream/`, then `git -C upstream diff > patches/<name>.patch` and make `setup.sh` apply it.
Our code: `mm/` (`vlm_cad.py` prompts/view grid, `cadquery_io.py` `run_cadquery`/`iou`, `cad_helpers.py`
`near`/`namespace`, `sysutil.py` `usable_cpus`), `scripts/`, `datagen/`, `jobs/`, `tests/`, top-level `train_*.py`
(current: `train_vlm_cq.py` SFT, `train_vlm_grpo.py` RL; `train_vlm_rl.py` superseded).

## Inference pipeline ("full pipeline")

Greedy answer -> `scripts/orient_fix.py` (try 24 right-angle rotations, keep the best render match) -> one revision
turn (`scripts/revise_vllm.py --rounds 1`: model sees input + render of its program or the build error) -> orient
again -> `scripts/revise_guard.py` (keep the revision only if its render matches the input better). Uses only the
input images, never ground truth. `scripts/final_chain.sh` runs score -> orient merge -> guard -> paired comparisons.

## Key decisions (rules, with why)

- **VLM writing CadQuery, not GenCAD's latent pipeline**: the latent prior plateaued ~11 IoU on novel parts whatever
  we changed; CadQuery + CAD-Recode's ~1M programs jumped to ~89. Don't revisit latent-prior variants.
- **Stay at 2B unless the user funds a full recipe**: a 4B with a compressed recipe projected below the 2B; a fair test
  needs the full ~30k-step SFT + RL (~$300+). Data and feedback win, not size.
- **Four fixed cameras, 2x2 grid, full resolution** (896 px grid, 448/view; `scripts/render_views.py`: iso (1,1,1),
  iso (-1,-1,-1), front (0,-1,0), top (0,0,1)). Downscaled grids lose detail.
- **Online GRPO is the main post-training** (on-policy vLLM, in-process weight sync, reward = IoU, invalid code -0.5).
  It beat HF-sampled RL, RFT and DPO. **SFT after RL erodes RL gains**, so always follow SFT with an RL top-up.
- **Real Fusion 360 Gallery parts are the yardstick** (600-part test set, own IoU, 4 views); the generated test set
  overstates progress. RL mixes real / generated / DeepCAD; too much real (45/15/40) dented generated parts, 36/24/40
  was fine.
- **Always guard the revision turn**: unguarded it rewrote 87 % of generated-part programs (77.1 -> 62.1).
- **Orientation search at inference instead of retraining**: most bracket/sheet failures were the right part turned
  the wrong way (generators' final `.rotate` habit).
- **Claude is a teacher/reference, not a runtime dependency**: distil its good programs on real parts into the local
  model.
- Rejected (don't re-propose without new evidence): fine-tuning huge models (DeepSeek V4.1 Flash, Kimi K3.1, Mistral
  Large 4); switching to Ministral; SolidCraft / usd-brep as kernels (no data; maybe front end / export later); a
  separate revision-only model as default.

## Dead ends (don't repeat)

DPO on rejection-sampling pairs (likelihood displacement, then no gain); small on-policy RL with HF generation
(`train_vlm_rl.py`); untrained revision turn; more rounds of the same real-part RL (diminishing); self-hosting Mistral
Large 4 (vLLM 0.32 / nightly can't parse the NVFP4 config; FP8 needs 8xH200 that never scheduled); trusting
`scripts/rot_iou.py` (undercounts; use `orient_fix.py` + `rich_iou.py`). Details in `HANDOFF.md` section 4.

## Compute: Hugging Face Jobs + bucket

```bash
export HF_TOKEN=...                         # needs Jobs + write access
RUN=<name> jobs/launch.sh <stage> [args]    # e.g. vllmpred, vlmcq, vlmgrpo, revise, orient, richiou; DRY=1 prints
hf jobs logs <id> --follow                  # or: hf jobs inspect <id> --format json
```
- `jobs/launch.sh` defines stages and per-stage hardware/timeouts (`FLAVOR`, `TIMEOUT` override); `jobs/entry.sh` is
  the in-job dispatcher. A new stage needs an arm in both. Env knobs reach the job only if listed in the `ENVS` loop.
- Bucket `hf://buckets/mszarski/gencad-work`, mounted at `/work` in jobs (`results/<run>/...`, `data/...`; models at
  `results/<run>/model`). Pull with `hf buckets cp hf://buckets/mszarski/gencad-work/<path> <local>`; unpack data
  tars under `gencad/data/rich/`. Full data/results inventory in `HANDOFF.md` section 8.
- **Before every launch** move local `data/` aside (the snapshot uploads it):
  `mv data ../_gencad_data_aside && mkdir data && touch data/.keep`, launch, then
  `rm data/.keep && rmdir data && mv ../_gencad_data_aside data`. An empty `data/` breaks `entry.sh`.
- Local `gencad/results/` is gitignored and does not survive the container: copy anything important to the bucket.
- `h200x8` can sit in SCHEDULING for hours; `rtx-pro-6000x8` starts in minutes. Prices: h200 $5/h, h200x8 $40/h,
  a100-large $2.5/h, rtx-pro-6000x8 $22/h, cpu-upgrade $0.03/h, cpu-xl $1/h.
- Background watchers die after ~2 h and on container restarts (a crashed job once went unnoticed ~35 h): re-arm
  watchers that also stop on ERROR, and check job state directly.
- `pkill -f <pattern>` can kill your own shell; kill by PID.

## Evaluation conventions

- **Own IoU for generated/real parts** (renders are framed to the part, so absolute size is unobservable; "gt" IoU is
  meaningless there). **gt IoU for DeepCAD** ("own" can collapse to ~0 for a correct solid).
- Compare models with **paired differences on the same parts** (rows files), against the previous best. On DeepCAD
  report dup vs novel splits (38 % of test programs duplicate training ones).
- Score DeepCAD with `scripts/cq_iou.py --rows`, generated/real with `scripts/rich_iou.py`. Bulk real-part scoring
  is slow locally (hours for 5k): use the `richiou` job stage (cpu-xl). If scoring reports `timeout/crash` rather than
  ~0, rerun with fewer `--workers`.
- Real-part GT programs are `cq.importers.importStep("data/rich/real*/...")` with relative paths: score from
  `gencad/` with the shards unpacked under `data/rich/`.
- Real train split excludes every test design id (`datagen/real_test_designs.txt`; id = file name up to first `_`).
- GenCAD's 5 renders per uid are mostly a *different* part (only `_0` shows the GT); use our renders or `--true_view`.

## Gotchas

- In HF Jobs `os.cpu_count()` returns host CPUs -> oversubscribed pools, CAD builds time out, parts silently lost.
  Size pools with `mm.sysutil.usable_cpus()` everywhere.
- Code that executes generated programs must use `mm.cadquery_io.run_cadquery` / `mm.cad_helpers.namespace()` (injects
  `near()`; CadQuery's `NearestToPointSelector` compares edge centres, so circular edges can't be picked).
- Global selectors like `'>Z'` misbehave on YZ/XZ workplanes: generators build in XY and rotate once at the end.
- vLLM in jobs: needs gcc (`need_vllm` installs it), `VLLM_USE_FLASHINFER_SAMPLER=0`; GRPO uses the in-process engine
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) for weight sync. `VLLM_NIGHTLY=1` installs nightly.
- Job image: old `brotli` breaks Hub downloads of new base models (`vlmcq` upgrades it); needs torch>=2.6. Large
  multi-image batches OOM: use smaller bs + grad accum and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Claude API key: env var `GENCAD_ANTHROPIC_KEY` (older notes say `GENCAD_API_KEY`; `ANTHROPIC_*` names are
  reserved). Run `ANTHROPIC_API_KEY="$GENCAD_ANTHROPIC_KEY" python scripts/claude_baseline.py ...`. **Never print,
  echo or reveal the key, its prefix or its length.**
- Batches API: custom_ids only `[a-zA-Z0-9_-]`; 256 MB per batch (~5,000 one-image requests; split larger). After any
  interruption (e.g. credit issue) check every batch's `request_counts` for errored/expired. Rich/real parts need
  `--rich` (the DeepCAD prompt allows only sketch-and-extrude).
- Licences: Text2CAD is gated (CC BY-NC-SA 4.0): use only `text2cad_v1.1/*.csv`, never commit its data. CAD-Recode is
  CC BY-NC 4.0 (matters for any product use).
- Upstream GenCAD: training code hard-codes CUDA, `configAE.py` defaults are debug-sized,
  `utils/image_processor.process_image` naming is inverted (`img_type="image"` applies Canny).

## Working conventions

- Commit messages are prefixed `gencad: ...` and usually name the result or feature.
- When an experiment finishes, record it in `gencad/CLAUDE.md` (results table + readings); keep `REPORT.md` in step
  for headline changes and `HANDOFF.md` current (state, spend vs budget, next steps).
- `.gitignore` drops `*.png` repo-wide; galleries and grids go to the bucket, not git.
