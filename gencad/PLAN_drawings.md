# Plan: drawings -> CAD with a local agent, scored like CADGenBench (Nov 24)

Goal: a local Qwen3-VL model plus our own agent loop that turns an engineering drawing (and, for edits, a STEP + a
text change) into an exact CAD part in millimetres, measured with CADGenBench's metrics, and a CADGenBench entry
that beats the frontier-model baselines (HF agent baselines: Claude Fable 5 0.45, GPT-5.6 0.41, Claude Opus 4.7
0.37; fine-tuned Qwen3-VL-2B 0.04, 8B 0.09, 235B 0.24; top commercial agents ~0.87-0.90).

## What we know (checked Nov 24)
- CADGenBench (github.com/huggingface/cadgenbench, Apache-2.0; inputs ODC-BY; ground truth private): 81 parts from
  Mecado. 49 generation tasks: a 3072x2118 A2 drawing sheet (orthographic views, detail views, dimensions, notes such
  as "all radii 2", small isometric; 3 tasks have 2 sheets). 32 editing tasks: input STEP + one-sentence change
  ("increase the block height from 30 mm to 40 mm") + iso/front/top/right renders (1024x768) + mesh. Output: STEP.
- Score: invalid = 0; generation 0.4 shape + 0.4 interface + 0.2 topology; editing 0.6 shape gain over the input +
  0.3 interface + 0.1 topology. Shape = mean(surface F1, volume IoU) after rigid ICP alignment (no scaling: size in
  mm counts). Interface = per mating feature (hole, slot, boss) the IoU of a keep-out / keep-in volume (`jig_<group>
  __<i>__KOR|KIR.step`), pose-searched +-1 deg / +-1 %, ramp 0.80 -> 0, 0.95 -> 1; a group scores its worst feature.
  Topology = Betti numbers. Their scorer runs locally on any part that has these files.
- Their baseline agent: one tool (run Python in a work dir), writes build123d (CadQuery backend exists), renders,
  iterates until [DONE].
- SolidCraft: drawings work headlessly (`drawing.create standard_views`, `drawing.view.*`, `drawing.export png/pdf`,
  hidden-line removal); `drawing.dimension.from_model` only finds sketch/feature dimensions, none on our converted
  parts, so dimensions must be added by us (`drawing.dimension.add`) from the geometry `to_solidcraft.py` traces.
  Our programs convert to SolidCraft at ~99.5 % identical geometry (real, self targets, CAD-Recode); generators need
  converter work (cones, mirror, taper, shells).

## Phase 0: metric and harness: DONE Nov 24 (results in CLAUDE.md "CADGenBench and drawings")
Our model, CADGenBench score on 600 real parts: 0.22 as written, 0.64 at the true size (best-of-9 0.68); 2B reads
~68 % of dimensions zero-shot at 91 % precision, like 4B / 8B. Step 4 (their agent harness with our model) not run.

## Phase 0 (original text)
1. Install cadgenbench; score our 600 real test parts with its shape/topology/validity (absolute mm and scale-fitted,
   to see what absolute size costs us today).
2. Auto interface regions for our parts: from quiddity features of the GT (holes, slots, pockets -> keep-out
   cylinders/prisms with a clearance; bosses -> keep-in), written as `jig_*` STEP files; run their interface scorer on
   our predictions and Claude's. Check it ranks parts sensibly (gallery of red/amber overlays).
3. Add `cad_score` (their weights) to `scripts/rich_iou.py` rows and as an RL reward option in `train_vlm_grpo.py`.
4. Run their baseline agent harness locally with our model behind an OpenAI-compatible vLLM endpoint (CadQuery
   backend), on the public inputs, to get a first honest number for "our model, their agent" (no submission yet).

## Phase 1 + phase 2 pilot: DONE Nov 24 (CLAUDE.md "Drawing -> CadQuery pilot")
28.6k drawings; pilot SFT 1k steps: abs IoU on held-out drawings 0.6 -> 54.8, size error 52 % -> 4.8 %, valid 95 %.
Cost ~$30 (data ~$4, SFT ~$22, evals ~$3).

## Phase 2A + 2B: DONE Nov 25 (CLAUDE.md "Drawings phase 2A + 2B")
101k more drawings in a more realistic style + 2000-step SFT (`vlm_draw2`): abs IoU on held-out style-v2 drawings
45.1 -> 59.9 (+14.8 paired), size error 7.9 % -> 2.6 %, 4 / 6 CADGenBench sheets build; best-of-8 consensus 63.5, every
pick builds (6 / 6 sheets). Generated parts still ~39. Cost ~$71 of the $60-80 approved.

## Style v3 + read-out round: STOPPED Nov 26 (CLAUDE.md "Drawings style v3 + read-out targets")
Found that our model line cannot read text (base 2B reads 70 % of dimensions, ours 2 %); given correct numbers it
builds at true size (51 IoU). Paused at the user's call (~$44). Next: a reader model, or redo the line from the base
with a reading task in every mix.

## Phase 1 (original text): drawing data (CPU jobs, ~$10-25)
1. Close the converter gaps (cones, mirror, taper, shells) so generator families convert.
2. Dimensioning pass (`scripts/make_drawing.py`): per part, SolidCraft views (front/top/right + iso, sometimes a
   section or detail view), then dimensions we choose from traced geometry: overall sizes, hole diameters and
   positions from datum edges, extrude depths, fillet/chamfer callouts or an "all radii r" note, bend notes for
   sheet parts. Randomise style towards CADGenBench's (A2/A3 sheet, title block, first/third angle, scale, line
   weights, a few omitted or redundant dimensions), export PNG at 3072 px.
3. Sources: CAD-Recode (1M), generator families (1.1M), our verified real-part programs (7k); start with ~200k
   drawings. Ship a stripped release build of `solidcraft-cli` to the bucket for jobs. Spot-check ~200 drawings by eye
   (and optionally a cheap Claude audit: "can the part be rebuilt from this drawing alone?").
4. Edit data: from `build_feature_edits.py` pairs (A -> B) plus parameter changes (scale one dimension), with a
   generated one-line instruction ("add 4 x d6 through holes on a 40 mm square", "increase the flange height from 30
   to 40 mm"); input STEP + renders, target program = edit applied to the imported STEP.

## Phase 2: model (H200, ~$80-120)
1. SFT from `vlm_grpo_self` on drawing -> CadQuery (absolute mm) + edit tasks + a share of the old mix. Open question:
   can a 2B read dimension text at the resolution we can afford (A2 sheet = tiny digits)? Test first with a 1k-step
   pilot (~$5): dimension-reading accuracy on held-out drawings. If poor: crops/tiles of the sheet, or 4B.
2. GRPO with reward = cad_score on our own parts (shape in mm + auto-interface + topology), keeping every checkpoint.

## Phase 3: our agent (local, part of Phase 2 budget)
Loop around the local model, no cloud at runtime: write CadQuery -> build (error text back) -> render the result as
the same drawing views -> compare per view with the input (silhouette / line agreement) and measure: overall
dimensions, hole list (quiddity) vs the dimensions the model read -> short feedback -> revise; best-of-N with the
drawing-agreement picker (our best-of-9 lesson: picking gave +5 where training on edits gave 0). Train the revise turn
with GRPO per-turn rewards (infra from `vlm_grpo_feat`).

## Phase 4: CADGenBench entry
Run on the 81 public inputs, submit once (later at most 2-3 times; no tuning on their leaderboard). Report honestly:
generation and editing separately, vs the HF baselines and the fine-tuned Qwen entries.

## Decision gates
- After Phase 0: does their interface score separate our good and bad parts? Does our current model score anything
  through their agent? (Expect ~0.05-0.15, like other small fine-tunes.)
- After the Phase 2 pilot: can the model read dimensions? If not, stop or change resolution/size before the full run.
- Budget ask before Phase 2.

## Risks
Synthetic drawings may not look like Mecado's (style gap); 49 generation parts make scores noisy (~+-0.03);
sheet-metal parts dominate their set (our bent_sheet family is the weakest); our interface regions are a proxy for
their hidden hand-made ones.
