# Drawings → CAD, and CADGenBench

Goal: from an engineering drawing sheet (orthographic views, dimensions, notes), write a CadQuery program for the part
**at its real size in millimetres**, as scored by [CADGenBench](https://github.com/huggingface/cadgenbench) (49
drawing → STEP tasks; 0.4 shape + 0.4 interface + 0.2 topology, rigid alignment without rescaling). Plan:
`gencad/PLAN_drawings.md`.

## Synthetic drawings (`scripts/make_drawing.py`)

Each training program is built, exported to STEP, inserted into [SolidCraft](https://github.com/tangency-labs/solidcraft)
(headless CLI), drawn with standard views (front / top / right / iso, first or third angle) on a random A4 / A3 / A2
sheet, dimensioned from the B-rep, and exported as a ~2400 px PNG. Style v3 (current):
- overall sizes measured between the part's true extremes (corners, or for round parts an edge-on circle or the
  outlines of the end cylinders);
- levels of flat faces, every concentric diameter at a centre, hole positions from an edge (or centre-to-centre on
  round parts), "4X Ø6.0" callouts for repeated holes, a few radii or an "ALL RADII r" note;
- staggered dimension lines and labels so numbers do not collide; 0-2 decimals.

Every program is first rescaled to a random size (longest side 10-500 mm, log-uniform; `scripts/scale_program.py`
scales length literals only and checks the result is exactly k times the original), so the printed numbers are the
only way to get the size right. Training targets start with a read-out line, e.g.
`# read: size 97.90 x 47.00 x 23.30; dia 5x 4.50; levels z 3.40; from edge x 63.65, y 23.50`, then the program.

## What we learned

- A 2B model reads dimension digits zero-shot about as well as 4B / 8B (~68 % recall, ~91 % precision); associating
  numbers with features is the hard part.
- CADGenBench's own scorer on our real-part predictions (from renders, with automatic mating-feature regions): 0.64 when rescaled to the true size, 0.22
  as written. Absolute size is the gap.
- First drawing models (`vlm_draw_pilot`, `vlm_draw2`) learned shapes from drawings (shape-only IoU ~75) but not
  absolute size: CAD-Recode parts are all 200 mm long, and round parts' overall dimensions were printed wrong. On
  drawings at random sizes `vlm_draw2` scores 11 IoU at true size, with no correlation between predicted and true
  size. Style v3 and `vlm_draw3` (training) address this.
- Picking among 8 candidates by mutual agreement adds ~3-4 points and makes every answer build; picking by OCR'd
  dimensions did not help.

## Not done yet

A run on the 49 public CADGenBench tasks (no submission yet); section and detail views and callout notes in our
drawings; a picker that redraws each candidate and compares it with the input sheet.
