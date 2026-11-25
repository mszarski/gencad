# Results

All numbers are volume IoU in % (invalid code scores 0). Comparisons between models are **paired** (same parts) with
standard errors of roughly 0.3-1 point; the full record, with commands, is in `gencad/CLAUDE.md`.

Two normalisations:
- **gt**: both solids in the ground truth's frame, so position and size errors count (DeepCAD).
- **own**: each solid fitted to its own bounding box, so only shape counts. Used where absolute size is not visible
  from the input (renders are framed to the part).

## 1. DeepCAD (renders → program)

2000 test parts (1997 whose ground truth builds); 38.5 % are near-copies of training parts ("dup"), the rest are
"novel". One greedy answer, own IoU, all / novel:

| model | 1 view | 4 views | 1 view + expert text |
|---|---|---|---|
| GenCAD (released) | – | ~22 / 11 | – |
| Claude Opus 5.5, zero-shot | – | 83.4 / 80.6 | 80.5 / 80.2 |
| `vlm_cq_v1` (SFT, CadQuery + CAD-Recode) | 82.9 / 78.2 | 89.1 / 85.2 | 88.4 / 84.2 |
| `vlm_grpo_fullres` (+ 896 px grid + online RL) | **86.5 / 81.9** | **91.4 / 88.4** | 89.0 / 85.3 |
| `vlm_grpo_self` (current best overall; gt IoU) | 86.5 | 91.0 | 89.3 |
| cadrille, published (whole DeepCAD test) | 81.7 (SFT) | 86.1 (SFT), 92.2 (+RL) | – |

Later models trade a few tenths on DeepCAD for large gains on realistic and real parts (below).

## 2. Procedural parts with real features

21 generator families (flanges, brackets, enclosures, shafts, sheet metal, …), 1989 test parts, own IoU, 4 views:
55.5 before training on them → 80.2 (one answer) → **83.9** with orientation search; weakest families are bent sheet
metal, L-brackets and enclosures.

## 3. Real designs (Fusion 360 Gallery)

600 test parts (revolve / chamfer / fillet / plain, 150 each), no design shared with the 14k training parts. Own IoU,
4 views:

| method | own IoU | detail score |
|---|---|---|
| `vlm_grpo_fullres` (before real-part training) | 59.9 | – |
| Claude Opus 5.5, zero-shot | 66.5 | 69.7 |
| `vlm_grpo_self`, one answer | 66.5 | 68.8 |
| `vlm_grpo_self`, full pipeline (orient + guarded revision) | 69.0 | – |
| **`vlm_grpo_self`, best-of-9 picked by render agreement** | **71.9** | **73.6** |

Best-of-9 vs Claude: +5.4 ± 1.1 paired. Robustness: rotating the cameras by 15-30° still costs 12-19 points; the model
is tied to its training cameras.

## 4. Engineering drawings → part in millimetres

Target: [CADGenBench](https://github.com/huggingface/cadgenbench) (drawing sheet → STEP, scored at true size). Our
synthetic drawings are made with SolidCraft from our programs. See [drawings.md](drawings.md).

| model | held-out set | builds | true-size IoU | shape-only IoU | reads size? |
|---|---|---|---|---|---|
| `vlm_draw2` | style v2 (parts at their dataset size) | 95 % | 59.9 | 74.8 | looked like it (2.6 % median size error) |
| `vlm_draw2` | **style v3, random sizes 10-500 mm** | 96 % | **11.1** | 75.0 | **no** (size correlation -0.01) |
| `vlm_draw3` | style v3 | training now | | | |

Reading: the earlier "true size" gains came from CAD-Recode parts all being 200 mm long, plus a dimensioning bug on
round parts; the model learned shapes but not to read overall size. Style v3 fixes the drawings and varies every
part's size, and its targets start by listing the dimensions read off the sheet.

## Things that did not help

DPO on sampled pairs, small on-policy RL with HF generation, imitation of Claude's programs (different coding style
broke the model), a 4B model on a shortened recipe, multi-turn feature-edit RL, and picking drawing answers by OCR'd
dimensions. Details and reasons in `gencad/CLAUDE.md` and `HANDOFF.md`.
