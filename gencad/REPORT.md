# Image/text → CAD: project report (as of 2026-10-25)

## Where we are, in one paragraph

We set out to replicate GenCAD (image → CAD program) and extend it to text and images. The original GenCAD design
(image → latent → program) turned out to be a dead end for parts it had not seen in training: about 11 % volume IoU on
novel parts, and nothing we did to it moved that much. Switching to a pretrained vision-language model (Qwen3-VL-2B)
that writes **CadQuery code**, trained on DeepCAD plus ~1M procedurally generated CAD-Recode programs, changed
everything. After post-training (rejection sampling, full-resolution inputs, online reinforcement learning on IoU),
our current best model, `vlm_grpo_fullres`, reaches **91.4 IoU from four views (88.4 on novel parts)** with a single
answer. That is 4× GenCAD, 8 points above Claude Opus 5.5 zero-shot, and close to the best published result we
know of (cadrille with RL, 92.2). It is a 2B model that runs locally; nothing at inference depends on a cloud API.

**Update (Oct 22): realistic parts.** DeepCAD has no fillets, chamfers, holes, revolves, shells or patterns, and the
best model scored only ~55 on such parts. We built 21 procedural part families (~1.1M checked parts), trained on them
plus a "look at what you built and fix it" revision turn, then ran online RL on both kinds of parts. The new model,
`vlm_grpo_rich`, scores **80–81 on realistic parts (+25)** and, with one revision turn guarded by render agreement,
**slightly beats the previous best on every DeepCAD image mode** (4 views 91.8 vs 91.6). See section 11. On *real*
designs (Fusion 360 Gallery) the gain was much smaller at first (+5 with the full pipeline); RL on 14k real training
parts then brought it to **+7.4 (67.3 vs 59.9)**, with DeepCAD slightly up as well. Best model: `vlm_grpo_real`.

## The task and how we measure it

- **Input:** renders of a part (one isometric view, or a 2×2 grid of four fixed cameras), optionally a text
  description (Text2CAD prompts at four detail levels; "expert" is the most detailed).
- **Output:** a CAD program that builds a solid.
- **Data:** DeepCAD (~150k train programs, CC-licensed) and CAD-Recode v1.5 (~1M procedural CadQuery programs,
  CC BY-NC 4.0) for training; 2000 DeepCAD test parts for evaluation (1997 whose ground truth builds).
- **Split that matters:** 38.5 % of test parts are near-copies of training parts ("dup", n = 770); the rest are
  "novel" (n = 1227). Novel parts are the honest measure; we report both.
- **Metric:** volume IoU between the built solid and the ground truth, in % (invalid code = 0), computed by
  `scripts/cq_iou.py` with OpenCascade. "gt" normalisation keeps position and size errors; "own" fits each solid to
  its own box (fair to zero-shot models). The two agree within ~0.3 points for our models.
- **Comparisons:** all deltas below are paired (same parts) with standard errors of ~0.3–0.5 points.

## What we did, in order

### 1. Replicating GenCAD (done)
Built a harness around the released GenCAD code and checkpoints. Our evaluation reproduces upstream's own inference
numbers within sampling noise. GenCAD gets whole programs right for ~10 % of test parts, almost all of them
near-copies of training parts.

### 2. Conditioning on text and images with a new diffusion prior (`mm_v1` and variants)
Replaced GenCAD's image prior with a multimodal diffusion prior over the same CAD latent, conditioned on SigLIP2
image embeddings and Qwen3 text embeddings. Results on latent-space and program metrics:
- Better than GenCAD (image retrieval R@1 0.44 vs 0.22; command accuracy 0.67 vs 0.60), but the gain came from the
  new denoiser and training, not the new encoders.
- Text helps on top of an image, and more detailed text helps more. Expert text alone beats GenCAD's image prior.

### 3. Diagnosing why it stalled
- **Skewed rectangles:** the CSR decoder picks every coordinate independently, so latents between two modes give
  trapezoids. Best-of-N reranking, stochastic sampling and an autoregressive decoder each fixed little.
- **Duplicates:** whole-program matches came almost entirely from near-copies of training parts. On duplicates,
  plain nearest-neighbour retrieval beat every generator.
- **Decoder ceiling:** decoding each novel part's *true* latent gives 84 IoU (65 % exact programs), while the best
  prior got 0.3 % exact. The bottleneck was mapping images/text to the right latent, not the decoder.
- Multi-view priors, an end-to-end tuned image encoder, and token-level text/image decoders all landed within about
  1 point of each other on novel parts.

### 4. A data bug: the training images mostly showed the wrong part
GenCAD's five "views" per part are not five camera angles: four of them show a differently proportioned part, and
our scripts (like upstream) usually picked one of those. Only 19.4 % of test parts were evaluated on their true
render. We re-rendered every program from four fixed cameras and used those from then on.

### 5. Measuring geometry, and the size of the gap
With IoU as the metric, every model so far scored 11–24 on novel parts (median Chamfer distance ~70–130: wrong
shapes, not near misses), while cadrille reported 78–82 on DeepCAD from images. That was the gap to close.

### 6. A pretrained VLM writing programs (`vlm_cad_v1`, then `vlm_cq_v1`)
- **Qwen3-VL-2B writing DeepCAD tokens** (`vlm_cad_v1`): the first model that generalised. Novel IoU 32 (1 view),
  34 (4 views), 43 (1 view + expert text).
- **Qwen3-VL-2B writing CadQuery, plus CAD-Recode data** (`vlm_cq_v1`, 30k steps on an H200): the big jump. DeepCAD
  converted to CadQuery (fidelity 0.98) plus ~1M procedural programs. **4 views 89.1 (novel 85.2)**, 1 view 82.9,
  1 view + expert 88.4, expert only 80.1. For the first time images carried more information than text.
- **Scoring fix:** the cloud scoring job had silently lost 2–6 % of parts to worker timeouts. Local re-scoring
  raised every vlm_cq number by 2–5 points. Scoring now runs locally and reports timeouts.

### 7. Claude Opus 5.5 zero-shot, as a reference
Same renders, a prompt describing the camera layout and an example of the style, no training, via the Batches API
(~$18 per 2000 parts). **4 views 83.4 (novel 80.6)**; 1 view + expert 80.5. Decent, but 5.7 points below our model.
Text barely helped it. Using Claude only as a fallback when our code fails to build gave +0.7.

### 8. Picking among candidates (render agreement)
Render each candidate program from the input cameras and keep the one whose grey-level image best matches the
input. With our model's own 8 samples plus its greedy answer, 4-view IoU rose from 89.1 to **91.7** (novel 89.3),
fully local. Adding Claude as a candidate added little for 4 views (91.9). Perfect picking would reach 94.3: the model
already produced better answers, it just did not put them first. That motivated post-training.

### 9. Post-training the model on its own outputs
| method | what it is | result on test (4 views, own IoU) |
|---|---|---|
| Small on-policy RL | 300 steps × 16 parts, HF sampling | +1.5 for 1 view only, flat elsewhere |
| Rejection-sampling fine-tuning (RFT) | 30k train parts × 9 samples via vLLM; best program per part by GT IoU kept (20.7k targets); fine-tuned on them | 89.6 (+0.5); 1 view +2.2 |
| DPO | 13k best-vs-worse pairs from the same samples | v1 made outputs worse (broken code rose to 8 %); v2 with strong anchoring: no change |
| Online RL (GRPO) | vLLM samples from the *current* policy, weights synced into vLLM every step; 1000 steps × 32 parts × 8 samples; reward = IoU | **90.5 (+1.45 vs original)**, every mode improved, fewer invalid programs |
| Full-resolution 4-view input | the 2×2 grid had been downscaled to 224 px per view; fed at 896 px (448 per view) | 90.4 (+1.3 vs original) |
| **Full-res + online RL** | online RL starting from the full-res model | **91.4 (+2.4 vs original; novel +3.2)** |

The two last gains came from different places (sharper inputs vs better answers) and added up.

### 10. Visual check
Galleries of real test parts (`scripts/gallery.py`, bucket `results/gallery/`): simple novel parts are right for
every method; the RL models mostly win by turning broken or non-building outputs into correct solids; the main
remaining failure is occasional code that does not build on simple parts (~1.5 % of parts).

### 11. Realistic parts and a revision turn (Oct 21-22)
- **Data (Tier 1).** 21 procedural CadQuery families (flanges, brackets, enclosures, gears, pulleys, shafts, knobs,
  bent sheet, rib panels, random feature mixes...) written and audited by Claude sub-agents to one spec, every part
  machine-checked (builds, one valid solid, sane size, deterministic). ~1.1M training parts on cheap CPU jobs; a
  held-out test set of 1,989 parts (`datagen/`, `scripts/gen_rich_data.py`, `scripts/rich_iou.py`).
- **Revision turn (Tier 2).** The model gets its own previous program plus either the build error or a render of what
  it built, and writes a corrected program. Training data: 46k examples from the model's own DeepCAD attempts.
- **Training.** Imitation training from the best model on a mix of DeepCAD, CAD-Recode, rich parts (40 %) and revision
  examples (15 %), 10k steps (`vlm_rich_v1`), then 600 steps of online RL with half rich, half DeepCAD prompts
  (`vlm_grpo_rich`). Imitation alone gained +24 on rich parts but cost 4-5 points on DeepCAD images (it undid the
  earlier RL); the RL step recovered DeepCAD and added +3 on rich parts.
- **Render-agreement guard.** On rich parts the revision turn rewrote most programs and *hurt* (77 → 62), because the
  revision data was DeepCAD-only. Keeping a revision only when its render matches the input views better than the
  first attempt's (no ground truth needed) makes it a gain everywhere.

Final model `vlm_grpo_rich` (test sets; DeepCAD gt IoU, paired against `vlm_grpo_fullres` on the same 1,997 parts;
rich parts own IoU, previous best measured on the 1,489 named-family parts):
| input | previous best | new, one answer | new + guarded revision | paired diff (guarded) |
|---|---|---|---|---|
| DeepCAD 1 view | 86.6 | 85.9 | **87.1** | +0.55 ± 0.29 |
| DeepCAD 4 views | 91.6 | 90.9 | **91.8** | +0.23 ± 0.29 |
| DeepCAD 1 view + expert text | 89.1 | 89.0 | **89.9** | +0.71 ± 0.28 |
| rich parts, 1 view | ~50.6 | 72.7 | **73.5** | ≈ +23 |
| rich parts, 4 views | ~55.5 | 80.2 | **81.4** | ≈ +25 |
Programs that fail to build: 0.6-1.1 % on DeepCAD with revision (was ~1.4 %); rich 2 % (was 9-11 %). Gallery:
bucket `results/gallery/gallery_rich_view4.png`. Weakest rich families: bent sheet, L-brackets, enclosures,
extrusions, the hardest random mixes.

Follow-up (Oct 23): revision examples from rich parts (25.6k, the model's own attempts with the generator program
as target) trained into a separate revision model (`vlm_rev_v2`; imitation training again eroded first-attempt
quality, so the first attempt still comes from `vlm_grpo_rich`). The unguarded revision no longer hurts rich parts
(80.2 → 80.2 instead of → 65.2), and with the guard rich parts gain +0.75 ± 0.32 (4 views, 82.1) and +1.0 ± 0.33
(1 view, 74.5) while DeepCAD is unchanged within ±0.4. Small, and it costs a second model; the revision turn cannot
fix parts the model does not know how to build (bent sheet, enclosures, L-brackets got slightly worse).

**Real-part check (Oct 24).** 600 real user designs from the Fusion 360 Gallery (150 each with revolves, chamfers,
fillets, plain), rendered with our cameras, own IoU, paired:
| | previous best | new, one answer | new + guarded revision |
|---|---|---|---|
| 4 views | 59.9 | 57.5 (−2.3 ± 1.3) | **63.6 (+3.8 ± 1.1)** |
| 1 view | 57.0 | 54.4 (−2.6 ± 1.0) | **59.1 (+2.1 ± 0.9)** |
The large gain on our generated test set **does not carry over to real designs**: there the new model's first
answer is slightly worse (filleted parts: 19 % of programs fail to build, mostly generator-style fillets that
fail on real geometry). The guarded revision repairs most of that, so the full pipeline is +2–4 points better than
before on real parts — real, but far from +25. The generated test set overstates progress; real parts are now the
yardstick.

**Orientation search (Oct 24, no training).** Many wrong answers on brackets, bent sheet and extrusions were the right
part turned the wrong way. Trying the 24 right-angle orientations of each answer and keeping the one whose renders
best match the input pictures (input images only, no ground truth) lifts the generated test set 80.2 → 83.9
(L-brackets 66 → 80, bent sheet 53 → 66, extrusions 66 → 79), leaves DeepCAD unchanged, and on real parts brings the
full pipeline (orientation search + guarded revision) to **65.0 vs 59.9 for the previous best (+5.2 ± 1.1, 4 views)**
and 60.1 vs 57.0 (+3.1 ± 0.9, 1 view). The previous best model gains nothing from it: the habit came from our
generated data, and this undoes it.

**Training on real parts (Oct 25, extra $100).** 14,165 real Fusion 360 parts (no design shared with the test set)
rendered with our cameras, then 600 steps of online RL with 45 % real-part prompts (reward: shape match against the
real STEP), 15 % generated, 40 % DeepCAD. Final model `vlm_grpo_real`, paired against the model we started the
phase with (`vlm_grpo_fullres`):
| test | before | new, one answer | new, full pipeline | difference (pipeline) |
|---|---|---|---|---|
| **real parts, 4 views** | 59.9 | 64.5 | **67.3** | **+7.4 ± 1.0** |
| real parts, 1 view | 57.0 | 59.5 | **62.4** | +5.4 ± 0.9 |
| DeepCAD, 4 views | 91.6 | 91.2 | **92.0** | +0.4 ± 0.3 |
| DeepCAD, 1 view | 86.6 | 86.5 | **87.0** | +0.4 ± 0.3 |
| DeepCAD, 1 view + expert text | 89.1 | 89.2 | **89.9** | +0.8 ± 0.3 |
| generated parts, 4 views | ~55.5 | 78.8 | **82.2** | ≈ +26 |
(Full pipeline: first answer → orientation search → one revision → orientation search → keep the revision only if
its render matches the input better.) Real filleted parts went from 50.5 to 64.2 for the first answer alone. The
model now ties the old best on DeepCAD with a single answer and beats it on real designs; with the full pipeline it
is better on every test set we have.

A second round of the same RL (`vlm_grpo_real2`, with more generated parts in the mix) adds a little: real parts
68.0 with the full pipeline (+0.8 ± 0.7 over `vlm_grpo_real`, +8.2 ± 1.0 over the starting model), generated parts
83.1 (+0.9 ± 0.3), DeepCAD unchanged. Returns from repeating this RL are now small.

**Qwen3-VL-4B pilot (stopped).** A shortened recipe on the 4B model (12k-step fine-tune, then RL) learned only ~3
points faster per step than the 2B had; at that pace it would have ended below our 2B on DeepCAD, so it was
stopped halfway (~$60). A fair test of a bigger model needs the 2B's full training length (~$300+).

**Claude Opus 5.5 zero-shot on real parts.** With a prompt allowing the full CadQuery feature set, Claude scores
66.5 on the 600 real parts from 4 views (our best: 65.1 single answer, 68.0 full pipeline, +1.6 ± 1.1) with almost no
broken code (1.2 %), but only 50.3 from one view (ours 62.6). So the earlier "Claude is weaker" (DeepCAD, 83 vs 91)
was mostly about DeepCAD's conventions: on real designs from 4 views a frontier model is already about as good as
our trained 2B — which makes it a good teacher for real-part training data.

**Engineering drawings in millimetres (Nov 24-25).** To target the CADGenBench benchmark (drawing sheet -> STEP,
scored at true size) we made ~130k synthetic dimensioned drawings with SolidCraft (CAD-Recode, our generator
families, our real-part programs), in two styles, and fine-tuned the 2B on them (`vlm_draw_pilot`, then `vlm_draw2`).
On 548 held-out drawings in the more realistic style, volume IoU at true size went from ~0 (render model) to 45 (pilot)
to 60 (`vlm_draw2`), 64 with best-of-8 and a consensus picker; median size error 2.6 %. Simple CAD-Recode-style parts
reach ~80-86; busy generated parts ~39 (sizes misread). 4 of the 6 public CADGenBench sheets we tried build (6 of 6
with best-of-8); no benchmark submission yet. Render skills kept (DeepCAD view4 91.7).

## Current standings (test set, own IoU %, all parts / novel parts)

| model | 1 view | 4 views | 1 view + expert text | expert text only |
|---|---|---|---|---|
| GenCAD (released)* | – | ~22 / 11 (1 image) | – | – |
| our pre-VLM best (token decoder + text)* | – | – | ~44 / 24 | – |
| vlm_cad_v1 (VLM, DeepCAD tokens)* | 44 / 32 | 46 / 34 | 55 / 43 | 52 / 40 |
| Claude Opus 5.5, zero-shot | – | 83.4 / 80.6 | 80.5 / 80.2 | – |
| vlm_cq_v1 (VLM, CadQuery + CAD-Recode) | 82.9 / 78.2 | 89.1 / 85.2 | 88.4 / 84.2 | 80.1 / 73.9 |
| + online RL (`vlm_grpo_v1`) | 85.3 / 80.4 | 90.5 / 87.2 | 89.3 / 85.7 | 81.3 / 75.7 |
| **+ full-res + online RL (`vlm_grpo_fullres`, best)** | **86.5 / 81.9** | **91.4 / 88.4** | 89.0 / 85.3 | **81.5 / 76.0** |
| **rich + RL + guarded revision (`vlm_grpo_rich`, gt IoU)** | **87.1** / 82.5¹ | **91.8** / 88.1¹ | **89.9** / 85.6¹ | – |
| local best-of-9 picking (on vlm_cq_v1) | – | 91.7 / 89.3 | 91.2 / 88.3 | – |
| cadrille, published (whole DeepCAD test) | 81.7 (SFT) | 86.1 (SFT), 92.2 (+RL) | – | – |

¹ Novel-part figures for the unguarded revision (the guard picks per part, novel split not tabulated).
\* Scored with the earlier geometry script (gt normalisation), so indicative; the gap is far larger than any protocol
difference. cadrille scores the whole DeepCAD test set, we score a 2000-part subset.

## Tools built along the way
- `scripts/sample_rft.py`: vLLM sampling with ground-truth scoring in parallel (30k parts × 9 programs in 1.6 h).
- `scripts/predict_vllm.py`: test predictions in ~5 min instead of ~3 h (validated to within 0.1 IoU).
- `train_vlm_grpo.py`: online RL with in-process vLLM weight sync and fast vLLM validation.
- `train_vlm_rl.py`, `train_vlm_dpo.py`, `train_vlm_cq.py --rft`, `scripts/build_rft.py`: the other post-training
  methods.
- `scripts/select_render.py`: render-agreement picking. `scripts/fallback.py`: fallback combination.
- `scripts/claude_baseline.py`: the Claude zero-shot baseline. `scripts/gallery.py`, `scripts/gallery_rich.py`:
  visual galleries.
- `datagen/` + `scripts/gen_rich_data.py`: rich-part generation and checks; `scripts/rich_iou.py`: scoring.
- `scripts/revise_vllm.py`: the revision turn at inference; `scripts/revise_guard.py`: the render-agreement guard;
  `scripts/orient_fix.py`: orientation search; `scripts/prep_real.py`: the Fusion 360 Gallery real-part test set.
- A per-model input setting (`gencad_vlm.json`), so every loader feeds a model the image layout it was trained on.

## Caveats
- The selector and every result use renders from our own renderer; real photos would need more work.
- The 256-part validation set is too small to see ~1-point changes; only the 2000-part test set settles them.
- The "own" IoU collapses to ~0 on 8–13 correct parts per model (inflated bounding box); this costs 0.2–0.4 points
  and changes no ranking. Prefer "gt" IoU when they disagree.
- Remaining failures: complex stepped parts, and occasional code that fails to build (~1.5 %).

## Cost (Hugging Face compute; Claude API separately)
Since the Claude baseline (Oct 15): roughly $300 of H200/A100 time up to the full-res RL model; the realistic-parts
phase (Oct 20-22) about $130 more (data generation ~$5, imitation training ~$62, RL ~$52, predictions ~$10) of a
$200 budget; ~$40 more for the rich revision data and `vlm_rev_v2` (~$176 in total); then ~$52 of an extra $100
for real-part training and its evaluation. Claude API about $45 for the two zero-shot baselines. Earlier phases were not costed here.

## Possible next steps
1. **Better first attempts on the weak families** (done: rich revision data only made the revision safe, +0.75):
   RL focused on bent sheet, brackets and enclosures, or more varied data for them.
2. **More families aimed at the weak spots** (sheet metal, brackets with features on several faces, shells with
   cut-outs), and harder random mixes.
3. **Best-of-N with render picking** on top of the guarded revision (no training).
4. **A larger base model (Qwen3-VL-4B) with the full recipe**, ~$200+.

Where things are: code and notes on branch `claude/gencad-multimodal` (`gencad/`); models, predictions and per-part
scores in the bucket `mszarski/gencad-work` (best model: `results/vlm_grpo_real2/model`; earlier: `results/vlm_grpo_rich/model`, `results/vlm_grpo_fullres/model`). Detailed lab notes,
commands and gotchas: `gencad/CLAUDE.md`.
