# Handoff: image/text → CAD program (the `gencad/` project)

> **Note:** this handoff was written in the old `mszarski/claude_playground` monorepo, before the project moved here.
> References to that repo, its `claude/...` branches, `trajectory_classifier`, the `bd` tracker, or "don't edit
> CLAUDE.md" are out of date. Where this file and CLAUDE.md disagree, CLAUDE.md wins.

Written 2026-11-02 at the end of a long Claude Code session, for a fresh session that has none of that
conversation. The project lives in `gencad/` of `mszarski/claude_playground` (branch `claude/gencad-multimodal`)
and is moving to its own repository; paths below are relative to `gencad/` unless they start with `/` or `hf://`.
The detailed lab notebook is `gencad/CLAUDE.md` (per-experiment sections, commands, numbers); the readable summary
is `gencad/REPORT.md`. This file is the bridge: what matters, what is in flight, and how the user likes to work.

---

## 1. Goal

**Build a model that turns pictures (and optionally text) of a mechanical part into an executable CAD program**
(CadQuery Python that builds a valid B-rep solid), that runs **locally** on one GPU (no cloud API needed at
inference; the user said explicitly: "I want to make the model good, not have a cloud connection unnecessarily").

It started as a replication of GenCAD (arXiv 2409.16294, image → latent → DeepCAD program) plus text/image
conditioning, and became a fine-tuned vision-language model (Qwen3-VL-2B) that writes CadQuery.

**Success, as it evolved in the conversation:**
- Beat GenCAD and published baselines on the DeepCAD test set (done: ~92 volume IoU vs GenCAD ~22; cadrille + RL,
  the best published number we know, reports 92.2 on the whole test set).
- Handle *real engineering parts* — fillets, chamfers, holes, revolves, shells, patterns — that DeepCAD lacks. The
  yardstick now is the **600-part real Fusion 360 Gallery test set** (own IoU, 4 views), not our generated parts.
- Be "significantly" better than before overall (the user's phrase: "keep going until you've got something
  significant"), measured with paired comparisons on the same parts.
- Eventually: a usable tool (images in → CadQuery + STEP + preview out), robust to images that are not our renders.

---

## 2. Current state (2026-11-02)

### Best model and pipeline
- **Best model: `results/vlm_grpo_real2/model`** in the HF bucket `mszarski/gencad-work` (Qwen3-VL-2B, full-res 896 px
  2×2 view grid). Lineage: `vlm_cq_v1` (SFT DeepCAD + CAD-Recode) → `vlm_rft_v1` (rejection-sampling FT) →
  `vlm_cq_fullres` (896 px) → `vlm_grpo_fullres` (online RL) → `vlm_rich_v1` (SFT + generated rich parts + revision
  data) → `vlm_grpo_rich` (RL DeepCAD + generated) → `vlm_grpo_real` (RL + real Fusion 360 parts) → `vlm_grpo_real2`
  (second real-part RL round, larger generated share).
- **Inference pipeline** ("full pipeline"): greedy answer → `scripts/orient_fix.py` (try 24 right-angle rotations,
  keep the one whose renders best match the input views) → one revision turn (`scripts/revise_vllm.py --rounds 1`:
  model sees input + render of its own program or the build error) → orient again → **guard**
  (`scripts/revise_guard.py`: keep the revision only if its render matches the input better). Everything uses only
  the input images, never ground truth.

### Latest numbers (paired, same parts; IoU in %)
Real parts = 600 Fusion 360 Gallery test designs, own IoU (shape only). DeepCAD = 1997 test parts, gt IoU.
Generated = 1989 parts from our 21 families, own IoU.

| test | `vlm_grpo_fullres` (start of the rich/real phase), one answer | `vlm_grpo_real2` one answer | `vlm_grpo_real2` full pipeline |
|---|---|---|---|
| **Real, 4 views** | 59.9 | 65.1 | **68.0** (+8.2 ± 1.0) |
| Real, 1 view | 57.0 | 59.5 | **62.6** (+5.6 ± 0.9) |
| DeepCAD, 4 views | 91.6 | 91.4 | **91.9** (+0.3 ± 0.3) |
| DeepCAD, 1 view | 86.6 | 86.5 | **87.1** (+0.5 ± 0.3) |
| DeepCAD, 1 view + expert text | 89.1 | 89.2 | **90.0** (+0.8 ± 0.3) |
| Generated, 4 views | ~55.5 (1489-part subset) | 79.8 | **83.1** |

Other reference points on the real test (4 views): **Claude Opus 5.5 zero-shot 66.5** (1 view: 50.3; only 1.2 %
broken code); **per-part pick between our one answer and Claude by render agreement: 72.8** (best real-part number
so far; `results/ens_real/`). `vlm_grpo_real` (first real-part RL round): full pipeline 67.3.

### In flight at handoff — Claude "teacher data" for real parts (task: make Claude generate good training programs)
Goal: Claude writes CadQuery for every real *training* part; programs scoring high against the real STEP become
training targets (first real-geometry programs we have ever had), and Claude's corrections of its own weak attempts
become revision-turn examples.
- **First attempts, all 14,165 real training parts, 4 views, effort high** — all succeeded, all collected, all scored:
  - batch 1 `msgbatch_01EjsTZdQiP1xz81oxRz9rFL`: 5,000 parts, $50.58, own IoU 65.5, invalid 1.4 % →
    `results/claude_teacher_real/` (41 % ≥ 0.8, 25 % ≥ 0.9, 13.6 % ≥ 0.95)
  - batch 2 `msgbatch_01MsPhEoCMeBoBkSzYc3dmt1`: 4,600 parts, $45.08, own IoU 65.4 → `results/claude_teacher_real2/`
  - batch 3 `msgbatch_011tcUwpoNBghdUSeQxaQfcw`: 4,565 parts, $44.84, own IoU 66.0 → `results/claude_teacher_real3/`
  - each dir in the bucket has `batch.json`, `preds.pkl`, `rich_rows.pkl`, `rich_iou.json`.
- **Second attempts ("retry": input views + render of its own program or the build error → corrected program)**
  for parts with own IoU < 0.8:
  - batch 1 retry: 2,948 parts, batches `msgbatch_01UCNJx8gUu8bbnCGZJr3jfE` + `msgbatch_01RAbaRmomyAstWeg7yjaHCT`,
    all succeeded, **$51.47**, collected → bucket `results/claude_retry_real/{batch.json,preds.pkl,attempts.tar}`.
    **Scoring job running at handoff:** HF job `6ae7eeb30e0338d1bb7db918` (`richiou`, cpu-xl) → writes
    `results/claude_retry_real/rich_rows.pkl` + `rich_iou.json` in the bucket.
  - batches 2 and 3 retry: were being prepared locally (rendering ~2,700 weak attempts each, then submitting) by
    `python scripts/claude_baseline.py retry --src results/claude_teacher_real{2,3} --below 0.8
    --out results/claude_retry_real{2,3}`. **They may not have been submitted** if the container was reclaimed —
    check `results/claude_retry_real{2,3}/batch.json` in the bucket; if missing, re-run that command (needs the real
    training parts unpacked locally, see Pointers). Expected cost ~$50 each (retries cost ~1.7 cents/part: two
    images per request — my earlier $90-for-all estimate was too low).
- **Not started:** building the training set from these and training on it (Next steps §7.1).

Claude API spend so far: DeepCAD baselines ~$45 (earlier), real test $11.58, teacher first attempts $140.50,
retry batch 1 $51.47 → ~$250 total, plus ~$100 if retry batches 2–3 go out.

### HF compute spend (approximate)
Tier 1/2 phase ~$176 of a $200 budget; real-part extension ~$52 of an extra $100; `vlm_grpo_real2` ~$30; Qwen3-VL-4B
pilot ~$61 (stopped); Mistral Large 4 attempts ~$4; scoring/orient CPU jobs cents to a few dollars. The user approved
$200, then +$100, then ~$170 for a 4B model (of which ~$61 used). Ask before committing new large spend.

### What works / what is half-finished
- Works: data generation (`datagen/`, 21 families, ~1.1M parts), real-part prep (`scripts/prep_real.py`), SFT
  (`train_vlm_cq.py`), online RL (`train_vlm_grpo.py`, several weighted part sources), vLLM prediction/revision,
  scoring (`scripts/cq_iou.py`, `scripts/rich_iou.py`), orientation search, guard, galleries, Claude batch baseline +
  teacher + retry (`scripts/claude_baseline.py`), the whole eval chain (`scripts/final_chain.sh`), HF Jobs stages.
- Half-finished: teacher data (above); `scripts/mistral_real.py` + `mistral` job stage (work, but vLLM does not yet
  support Mistral Large 4); `scripts/rot_iou.py` (GT-side rotation diagnostic — undercounts, don't trust it).
- Not started: robustness to other cameras/lighting; a usable local tool; a fast renderer.

---

## 3. Decisions and reasons

- **VLM writing CadQuery instead of GenCAD's latent pipeline.** The latent diffusion prior plateaued at ~11 IoU on
  novel parts whatever we changed (encoders, views, decoder, sampler); the decoder ceiling was 84, so the
  image→latent mapping was the bottleneck. Qwen3-VL-2B writing DeepCAD tokens generalised (novel 32–43); writing
  **CadQuery** with CAD-Recode's ~1M programs jumped to 89 (4 views). CadQuery = mature OpenCascade kernel + huge
  corpus.
- **Qwen3-VL-2B, not bigger.** Small enough to run locally and to iterate cheaply. A 4B pilot learned only ~3 points
  faster per step (see Dead ends). Claude Opus zero-shot was weaker on DeepCAD (83.4) — size without task training
  doesn't win; data and feedback did.
- **Four fixed cameras, 2×2 grid, full resolution (896 px grid, 448 per view).** GenCAD's own "views" mostly showed
  the wrong part (only 19.4 % true renders) — we re-rendered everything. Downscaled grids lost detail (+1.3 from
  full res).
- **Online RL (GRPO) as the main post-training.** On-policy vLLM sampling with in-process weight sync every step,
  32 prompts × 8 samples, lr 2e-6, reward = IoU (gt for DeepCAD, own for generated/real), invalid code −0.5.
  It beat small HF-sampled RL, RFT and DPO. Each SFT step later partly undid RL gains; each RL step recovered them,
  so the recipe alternates SFT (new skills/data) → RL.
- **Own IoU for generated/real parts.** Renders are framed to the part, so absolute size is unobservable; "gt" IoU is
  meaningless there (~0–40). DeepCAD keeps gt IoU (canonical frame learnable).
- **Generated "rich" parts (Tier 1):** 21 procedural CadQuery families written and audited by Claude sub-agents to
  one spec (`datagen/README.md`): literal code, build in XY then one final `.rotate`, edges picked with `near()`,
  machine-checked (one valid solid, size/centre limits, deterministic). Chosen after the Spectral Labs "Quartz"
  post: real features + an executing environment.
- **Revision turn + render guard (Tier 2).** Training a "look at what you built and fix it" turn helped DeepCAD but
  hurt generated parts unguarded; a render-agreement guard made it safe everywhere and became part of the pipeline.
- **Orientation search at inference** instead of retraining: most failures on brackets/sheet/extrusions were the right
  part turned the wrong way (generator habit: final `.rotate`). Cheap and fair (input images only).
- **Real Fusion 360 Gallery parts** for test (600, balanced revolve/chamfer/fillet/plain, ≤120 faces) and training
  (14,165, no shared design id with the test set). The generated test set overstated progress (+25 there vs +2–5 on
  real parts at first); real parts are the yardstick.
- **Mixing weights in RL:** first real-part run 45 % real / 15 % generated / 40 % DeepCAD dented generated parts;
  second run 36/24/40 fixed it.
- **Picked step 600 over step 400** for `vlm_grpo_real` (best real one-answer, DeepCAD tied).
- **Claude as teacher, not as a runtime dependency (so far).** On real parts Claude zero-shot ≈ our model from 4
  views and rarely writes broken code; the two fail on different parts (ensemble 72.8). Distilling its good programs
  keeps the product local. Whether to also ship a Claude-assisted mode is an open question for the user.
- **Rejected:** fine-tuning huge models (DeepSeek V4.1 Flash 552B MoE, Kimi K3.1 ~3T, Mistral Large 4 1T) — far beyond
  budget; switching to Mistral's Ministral family (sideways move, porting cost); SolidCraft (Rust CAD app with JSON
  command scripts — young kernel, no data; maybe a product front end later); usd-brep (NVIDIA SMLib kernel with
  OpenUSD I/O — no data, Python 3.12-only wheel; maybe an export target or healer later); a separate revision-only
  model `vlm_rev_v2` as default (+0.75 on generated parts, costs a second model).

---

## 4. Dead ends (don't repeat)

- **GenCAD-style latent prior variants** (multimodal diffusion prior `mm_v1`, multi-view prior, end-to-end image
  encoder, token decoders, AR decoder, best-of-N reranking, stochastic sampling): all within ~1 point on novel parts.
- **DPO** on rejection-sampling pairs: v1 caused likelihood displacement (broken code rose to 8 %); v2 with nll 1.0,
  lr 3e-7: no gain.
- **Small on-policy RL with HF generation** (`train_vlm_rl.py`): +1.5 for 1 view only. Use `train_vlm_grpo.py`.
- **Untrained revision turn** on the old model: changes nothing (59.5 → 59.5).
- **Revision turn trained only on DeepCAD revisions, unguarded, on generated parts:** 77.1 → 62.1 (rewrites 87 % of
  programs). Always use the guard. Rich-part revision data (`vlm_rev_v2`) made it neutral unguarded, +0.75 guarded.
- **Imitation (SFT) after RL** erodes RL gains on image modes (DeepCAD view1 −4 to −7). Plan an RL top-up after any
  SFT, or use the SFT model only for a side role (e.g. revision).
- **More rounds of the same real-part RL:** round 2 gave +0.8 ± 0.7 on real. Diminishing returns.
- **Qwen3-VL-4B with a compressed recipe** (12k-step SFT from base, then RL): at step 6000 DeepCAD val view4 76.9 vs
  the 2B's ~73.7 at the same step (2B: 72.4 @5k, 79.1 @10k, 91.1 @30k). Projected to end below our 2B; stopped at
  step ~6000 (~$58, no checkpoint kept). A fair test needs the 2B's full ~30k-step SFT + RL (~$300+).
- **Running Mistral Large 4 (1T MoE) ourselves:** no HF inference provider; the official FP8 checkpoint (1,061 GB)
  only fits 8×H200 (scheduled for hours without starting); the NVFP4 checkpoint (610 GB) fits 8×RTX PRO 6000 but
  **vLLM 0.32.0 and nightly 0.33.0rc1 cannot parse its config** (`TypeError: argument of type 'NoneType' is not
  iterable` in `vllm/transformers_utils/configs/mistral.py` — it treats it as Mistral Large 3 and expects
  `llama_4_scaling`). Only Mistral's custom image `vllm/vllm-openai:mistral-large-4` works, and HF Jobs can't run
  our script inside it. Wait for upstream vLLM support or use Mistral's API (needs a key).
- **`scripts/rot_iou.py`** (best of 24 rotations against GT) undercounted badly; trust `orient_fix.py` + `rich_iou.py`.
- **Cloud scoring with `os.cpu_count()` sized pools** silently lost parts (see Gotchas).

---

## 5. Gotchas

**Infrastructure / HF Jobs**
- `jobs/launch.sh` uploads a snapshot of `gencad/` including `data/` (except big extracted folders). **Before every
  launch:** `mv data ../_gencad_data_aside && mkdir data && touch data/.keep`, launch, then
  `rm data/.keep && rmdir data && mv ../_gencad_data_aside data`. An empty `data/` is dropped and breaks `entry.sh`.
- Inside HF Jobs `os.cpu_count()` returns the host's CPUs (64 on an 8-vCPU job) → oversubscribed pools, CAD builds
  time out, parts silently lost. Every script uses `mm/sysutil.usable_cpus()` (cgroup quota); keep it that way.
- vLLM in jobs needs gcc (triton JIT) — `need_vllm` installs it; FlashInfer sampler disabled
  (`VLLM_USE_FLASHINFER_SAMPLER=0`, its JIT needs nvcc). GRPO uses the in-process engine
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) to sync weights (`model_runner.get_model().load_weights`, then
  `reset_prefix_cache`). `VLLM_NIGHTLY=1` installs the vLLM nightly.
- The job image's old `brotli` breaks Hub downloads of new base models (`process() takes no keyword arguments` in
  httpx2) — the `vlmcq` stage now runs `pip install -U "brotli>=1.2"`.
- 4B SFT at bs 8 OOMed at step 250 on a long multi-image batch → bs 4 × accum 8 and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (now set in the `vlmcq` stage).
- `h200x8` ($40/h) can sit in SCHEDULING for hours; `rtx-pro-6000x8` (Blackwell, 768 GB, $22/h) started in minutes.
- Hardware prices: h200 $5/h, h200x8 $40/h, a100-large $2.5/h, cpu-upgrade (8 vCPU) $0.03/h, cpu-xl (16 vCPU) $1/h.
- Results are written to the bucket via the `/work` mount (`results/` is a symlink in jobs). Locally, `gencad/results/`
  is gitignored and **does not survive the container** — copy anything important to the bucket
  (`hf buckets cp <file> hf://buckets/mszarski/gencad-work/results/<run>/<file>`).
- In this cloud session: background watchers die after 2 h and on container restarts; a missed watcher once let a
  crashed 4B job go unnoticed for ~35 h. Always re-arm a watcher that also stops on ERROR, and check job state
  directly (`hf jobs inspect <id> --format json`, `hf jobs logs <id>`).
- `pkill -f <pattern>` can kill your own shell when the pattern appears in the command line; kill by PID.
- Scoring 5k real parts locally (STEP import + IoU) on 4 CPUs takes hours → use the `richiou` job stage (cpu-xl).

**Data / metrics**
- "gt" IoU is meaningless for generated/real parts (unobservable size): use own IoU. For DeepCAD prefer gt; the own
  normalisation occasionally collapses to ~0 for a correct solid (inflated bbox, 8–13 parts per model).
- GenCAD's five "views" per part are mostly a *different* part; we render our own 4 cameras
  (`scripts/render_views.py`: iso top (1,1,1), iso bottom (−1,−1,−1), front (0,−1,0), top (0,0,1)).
- CadQuery's `NearestToPointSelector` compares edge *centres* (circular edges can never be picked); generators use the
  `near((x,y,z),...)` helper (`mm/cad_helpers.py`), which is injected into every runtime via `namespace()` — any code
  that executes generated programs must use `mm.cad_helpers.namespace()` / `mm.cadquery_io.run_cadquery`.
- Global selectors like `'>Z'` misbehave on YZ/XZ workplanes → generators build in XY and rotate once at the end
  (which is also why the model learned the wrong-orientation habit that `orient_fix.py` undoes).
- Real-part "GT programs" are `r = cq.importers.importStep("data/rich/real*/<shard>/step/<name>.stp")` — relative
  paths, so scoring must run from `gencad/` with the shards unpacked under `data/rich/`.
- Test-set leakage: real training split excludes every design id in the test set (`datagen/real_test_designs.txt`;
  design id = file name up to the first `_`).
- Text2CAD is gated (CC BY-NC-SA 4.0); use only `text2cad_v1.1/*.csv`; never commit its data. CAD-Recode is
  CC BY-NC 4.0 — note for any product use.
- Upstream GenCAD gotchas (only relevant to the old latent pipeline) are listed in `gencad/CLAUDE.md` → Gotchas.

**Claude / Anthropic API**
- The key is in env var **`GENCAD_ANTHROPIC_KEY`** in this session (older notes say `GENCAD_API_KEY`; `ANTHROPIC_*`
  names are reserved). Run as `ANTHROPIC_API_KEY="$GENCAD_ANTHROPIC_KEY" python scripts/claude_baseline.py ...`.
  **Never print, echo, or reveal the key, its prefix or its length** (the auto-mode classifier blocks it, and the
  user expects it).
- Batch custom_ids allow only `[a-zA-Z0-9_-]` (modes like `view1+expert` become `view1-expert`).
- The Batches API caps a batch at 256 MB: ~5,000 single-image requests fit; split larger sets (`retry` chunks at
  2,500 two-image requests). `pip install anthropic` locally if missing.
- The DeepCAD prompt only allows sketch-and-extrude; for rich/real parts use `--rich` (full-feature `TASK_RICH`).
- The user once had to fix an API credit issue mid-run; after any such interruption check every batch with
  `client.messages.batches.retrieve(id).request_counts` (errored/expired counts).

---

## 6. The user's preferences (how to work with them)

- **Plain language, bottom line first.** They repeatedly asked "so are we better overall?", "why is this worse?",
  "status". Answer with a short honest verdict and a small table against the *previous best* (paired, same parts),
  then the caveats. Don't oversell: when the generated-part gain didn't transfer to real parts, saying so plainly was
  the right call.
- **Show, don't just tell.** They asked to see outputs "with my eyes" → galleries (`scripts/gallery.py`,
  `scripts/gallery_rich.py`), sent as images.
- **Budget is theirs to set.** They give explicit budgets ($200, then +$100, then ~$170 for the 4B). Give cost
  estimates up front, track spend, and **ask (with options + a recommendation) before exceeding a budget or starting
  a large spend.** Use AskUserQuestion for real decisions (e.g. stop/continue the 4B, which model size). Smaller,
  clearly-in-scope spends (a few dollars of scoring) are fine without asking. Be explicit when an estimate was wrong
  (e.g. Claude retries cost more than estimated).
- **They like ambition and volume:** "make ridiculous amounts of data and use yourself (Claude) to check it",
  "make Claude make more programs, like triple that", "keep going until you've got something significant". Use
  sub-agents for authoring/auditing generators; use automated checks for volume.
- **Local model is the goal**, cloud APIs are for data/teaching/reference, not a runtime dependency — confirm with
  them before proposing a Claude-in-the-loop product.
- **They bring external links** (articles, model cards, repos: Spectral Labs Quartz, SolidCraft, Mistral Large 4,
  usd-brep, DeepSeek, Kimi) and want a straight "would this make a difference to us?" — read the primary source,
  separate claims from evidence, and tie it to our bottleneck (data/feedback, not size).
- **Process rules from the session/repo:** always commit and push work (a stop hook enforces a clean tree); commit
  messages end with the attribution lines the harness provides; never put model identifiers in commits/PRs; **don't
  create PRs unless asked**; don't edit `gencad/CLAUDE.md` when told it is updated separately (this handoff);
  project instructions in the root `CLAUDE.md` (trajectory_classifier) don't apply to `gencad/`.
- **Status updates:** frequent "status" pings — reply with where each job is, the latest numbers, ETA, spend.
  When a watcher dies or a job fails, say so and what you did about it.

---

## 7. Next steps (priority order)

1. **Finish the Claude teacher data.**
   - Make sure retry batches 2 and 3 were submitted (bucket `results/claude_retry_real{2,3}/batch.json`); if not,
     re-run `scripts/claude_baseline.py retry` for them (~$50 each — tell the user the cost first if not yet spent).
   - Collect: `ANTHROPIC_API_KEY="$GENCAD_ANTHROPIC_KEY" python scripts/claude_baseline.py collect --out
     results/claude_retry_real2 --wait` (same for 3); upload `preds.pkl` to the bucket; score with
     `RUN=claude_retry_real2 REAL=1 jobs/launch.sh richiou` (same for 3); check the running job
     `6ae7eeb30e0338d1bb7db918` for retry 1.
   - Build the training set: per part keep the best of (first attempt, retry) with own IoU ≥ 0.8 (consider 0.9 for a
     cleaner set) as SFT targets (`input_stem` = the real part's render stem). Also build revision examples from
     retries: (first attempt + its render/error → corrected program) when the corrected one scores ≥ 0.8, in the
     `revision.parquet` format of `scripts/sample_rich_rev.py` (`rid, uid, views, prompt, attempt, attempt_iou,
     target, kind, error, attempt_stem, input_stem`). Expect ~6–8k first-attempt targets plus several thousand
     corrected ones. Needs a small loader in `train_vlm_cq.py` for "target programs on real parts" (like `--rft`
     but with `input_stem` paths), or write them as a rich-style shard (`parts.parquet` with `uid, family, code` and
     a `renders/` dir) and pass it via `--rich`.
2. **Train on it**: short SFT from `vlm_grpo_real2` (low lr ~5e-6–1e-5, a few thousand steps, mix teacher programs
   with the usual DeepCAD/CAD-Recode/generated data), then a GRPO top-up (real + generated + DeepCAD, weights like
   `--rich_w 0.4 0.6 --rich_p 0.6`) because SFT erodes RL gains. Evaluate with the full chain
   (`scripts/final_chain.sh <prefix> <orient prefix> <compare prefix>` after launching `revise` + `orient` jobs).
   Target: close the gap to the Claude+ours ensemble (72.8 real, 4 views). Budget estimate: ~$15–25 SFT + ~$30 RL +
   ~$5 eval — ask the user.
3. **Robustness check** (cheap, CPU): re-render the 600 real test parts with other cameras / lighting / render style /
   fewer views; measure the drop. Decides whether the model is usable outside our camera rig (and whether to train
   with randomised renders).
4. **Usable local tool**: images (+ text) → CadQuery + STEP + preview, running the full pipeline; needs a fast
   renderer for the orientation search (current pure-Python rasteriser: 96 renders/part, minutes per part).
5. **Optional/cheap references**: Mistral Large 4 via Mistral's API (needs a key; script = adapt
   `claude_baseline.py`) or self-hosted once vLLM supports it (`jobs/launch.sh mistral`, `FLAVOR=rtx-pro-6000x8`,
   `INIT=mistralai/Mistral-Large-4-1T-A52B-NVFP4 --mistral_format`).
6. **Bigger model, done properly** (only if the user wants to spend ~$300+): Qwen3-VL-4B with the full 30k-step SFT +
   RL recipe.

**Open questions for the user**
- Is a Claude-assisted mode (ensemble 72.8 vs 68.0 local) acceptable for some uses, or strictly local?
- Budget for the teacher-data SFT + RL (step 2) and for retry batches 2–3 if not yet sent.
- Quality cutoff for teacher programs (0.8 vs 0.9) — can be decided from a small ablation.
- Target input for the product: our renders, arbitrary renders, or photos (photos would need new data)?

---

## 8. Pointers

**Code (all under `gencad/`)**
- Models/IO: `mm/vlm_cad.py` (prompt/messages, `compose_views`, `revision_text`, per-model grid config
  `gencad_vlm.json`), `mm/cadquery_io.py` (`run_cadquery`, `iou`), `mm/cad_helpers.py` (`near`, `namespace`),
  `mm/sysutil.py` (`usable_cpus`).
- Training: `train_vlm_cq.py` (SFT; `--rich`, `--revision dir1 dir2`, `--rev_keep_rich`, `--rft`, `--grid_side`,
  `--model`), `train_vlm_grpo.py` (online RL; `--rich dir1 dir2 --rich_w ... --rich_p`), `train_vlm_dpo.py`,
  `train_vlm_rl.py` (superseded).
- Data: `datagen/` (spec `README.md`, `check.py`, `preview.py`, `families/*.py`), `scripts/gen_rich_data.py`,
  `scripts/prep_real.py` (Fusion 360), `scripts/sample_rich_rev.py` + `scripts/build_revision.py` (revision data),
  `scripts/sample_rft.py` + `scripts/build_rft.py`, `scripts/render_views.py`.
- Inference/eval: `scripts/predict_vllm.py`, `scripts/revise_vllm.py` (`--revise_model`), `scripts/orient_fix.py`,
  `scripts/revise_guard.py`, `scripts/cq_iou.py` (DeepCAD), `scripts/rich_iou.py` (generated/real),
  `scripts/final_chain.sh` (score → wait for orient jobs → merge → guard → paired comparisons),
  `scripts/gallery.py`, `scripts/gallery_rich.py`, `scripts/select_render.py`.
- Baselines/teachers: `scripts/claude_baseline.py` (`submit [--rich DIR --sample --n --exclude_batch]`, `collect
  [--wait]`, `retry --src --below --out`), `scripts/mistral_real.py`.
- Jobs: `jobs/launch.sh <stage> [args]` (env knobs `RUN INIT STEPS RICH RICH_TEST REAL REV REV_RICH SHARD N_SHARDS
  VLLM_NIGHTLY FLAVOR TIMEOUT`; `DRY=1` prints), `jobs/entry.sh` (stages incl. `vlmcq vlmgrpo vllmpred revise
  orient richiou realgen richgen revdata richrev rftsample mistral`).
- Docs: `gencad/CLAUDE.md` (lab notebook, per-experiment commands and numbers), `gencad/REPORT.md` (summary),
  `gencad/datagen/README.md` (generator spec).

**HF bucket `hf://buckets/mszarski/gencad-work`** (mounted at `/work` in jobs; `HF_TOKEN` in the environment)
- Models: `results/vlm_grpo_real2/model` (best), `results/vlm_grpo_real/model`, `results/vlm_grpo_rich/model`,
  `results/vlm_rev_v2/model` (revision-only), `results/vlm_rich_v1/model`, `results/vlm_grpo_fullres/model`.
  Note `results/vlm_cq_v1/model/` has six stale `.tmp*` files of 4.9 GB (interrupted saves).
- Data: `data/rich/train_{0..27}.tar` (generated, ~1.1M parts: 20 shards of the named families, 8 of random_mix),
  `data/rich/test_{0,1}.tar` (generated test 1489 + 500), `data/rich/test_real.tar` (600 real test parts →
  `data/rich/real`), `data/rich/real_train_{0..3}.tar` (14,165 real training parts → `data/rich/real_train`),
  `data/revision.tar` (DeepCAD revision data), `data/rich_revision.tar` (generated-part revision data),
  `data/renders.tar`, `data/cad_vec.tar`, `data/deepcad_cq.parquet`, `data/cad_recode.parquet`,
  `data/recode_renders_{0..3}.tar`, `data/text_index.json`.
- Results used above: `results/final2_{real,rich,dc}` (+ `_o` merged/guarded) = `vlm_grpo_real2` evals;
  `results/final_*` = `vlm_grpo_real`; `results/real_old` (old model on real); `results/vlm_grpo_fullres_test`
  (old model on DeepCAD); `results/rich_base_grpo_fullres` (old model on generated test_0); `results/claude_real`
  (Claude on real test); `results/ens_real` (Claude+ours pick); `results/claude_teacher_real{,2,3}`,
  `results/claude_retry_real` (teacher data); `results/gallery/` (galleries).
- Fetch locally: `hf buckets cp hf://buckets/mszarski/gencad-work/<path> <local>`; unpack tars under
  `gencad/data/rich/`.

**Key HF job IDs (namespace `mszarski`)**: `vlm_grpo_real2` 6ade7192046eb82b1ece808d; `vlm_grpo_real`
6add50d4046eb82b1ece1379; `vlm_grpo_rich` 6ad9a27d0a35baab53a26a05; `vlm_rich_v1` 6ad8f3775275124786e2885b; 4B pilot
6ae067ce3fb22a235dd27986 (cancelled); retry-1 scoring 6ae7eeb30e0338d1bb7db918 (running at handoff).

**External resources**
- Fusion 360 Gallery segmentation STEP release:
  `https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1_extended_step.zip`
  (42,912 STEP files + per-face labels; 0 ExtrudeSide, 1 ExtrudeEnd, 2 CutSide, 3 CutEnd, 4 Fillet, 5 Chamfer,
  6 RevolveSide, 7 RevolveEnd).
- Base models: `Qwen/Qwen3-VL-2B-Instruct` (ours), `Qwen/Qwen3-VL-4B-Instruct` (pilot).
- Datasets: DeepCAD, CAD-Recode v1.5 (CC BY-NC 4.0), Text2CAD v1.1 (gated, CC BY-NC-SA 4.0).
- Papers: GenCAD arXiv 2409.16294; cadrille arXiv 2505.22914 (RL baseline, 92.2); Ortho2CAD arXiv 2607.08891
  (Qwen3-VL-8B → CadQuery); Spectral Labs "Building CAD for machine intelligence" (motivated Tier 1/2).
- Considered: Mistral Large 4 (`mistralai/Mistral-Large-4-1T-A52B`, `-NVFP4`; vLLM image
  `vllm/vllm-openai:mistral-large-4`), DeepSeek V4.1 Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, multimodal, MIT,
  552B MoE), Kimi K3.1 (open weights promised ~Nov 4), SolidCraft (`github.com/tangency-labs/solidcraft`), usd-brep
  (`github.com/NVIDIA-Omniverse/usd-brep`, `pip install usd-brep`, Python 3.12 only).
