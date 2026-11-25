# Getting started

Run everything from the `gencad/` folder.

## Install

```bash
cd gencad
pip install torch torchvision -r jobs/requirements-job.txt
pip install cadquery pebble h5py trimesh pyarrow          # local scoring
./setup.sh                                               # clones upstream GenCAD at a pinned commit, applies patches/
python -m pytest tests/test_mm_smoke.py -q               # CPU smoke tests (~1-2 min)
```

Upstream GenCAD has no licence, so `setup.sh` puts it in `gencad/upstream/` and symlinks its folders in; none of it
is committed.

## Get a model and data

Weights and data are in a private Hugging Face bucket (`hf://buckets/<owner>/gencad-work`):

```bash
hf buckets sync hf://buckets/<owner>/gencad-work/results/vlm_grpo_self/model results/vlm_grpo_self/model
hf buckets cp hf://buckets/<owner>/gencad-work/data/rich/test_real.tar - | tar -x -C data/rich
```

## Predict and score

```bash
# renders -> programs (vLLM, GPU)
python scripts/predict_vllm.py --model results/vlm_grpo_self/model --rich data/rich/real --modes view4 \
    --out results/my_run/preds.pkl
# score: DeepCAD with gt IoU, generated / real parts with own IoU
python scripts/rich_iou.py --preds results/my_run/preds.pkl --out results/my_run/iou.json --rows results/my_run/rows.pkl

# drawings -> programs in mm, scored at true size
python scripts/eval_drawings.py --model results/vlm_draw2/model --drawings data/draw3_val --out results/my_run/draweval.json
```

Make a drawing of any CadQuery program or STEP file (needs a `solidcraft-cli` build):

```bash
python scripts/make_drawing.py --codes_parquet programs.parquet --cli /path/to/solidcraft-cli --out data/my_drawings
```

## Train and evaluate on Hugging Face Jobs

```bash
export HF_TOKEN=...                                  # Jobs + write access
RUN=my_sft jobs/launch.sh vlmcq --model results/vlm_grpo_self/model --steps 2000 ...
RUN=my_rl INIT=results/my_sft/model jobs/launch.sh vlmgrpo
DRY=1 RUN=x jobs/launch.sh <stage>                   # print the command only
hf jobs logs <job id> --follow
```

Stages and their hardware are listed in `jobs/launch.sh`; `jobs/entry.sh` runs them. Before launching, move the local
`data/` folder aside (the launcher uploads a snapshot of the folder); see the root `CLAUDE.md` for this and other
gotchas.

## Where to read more

- `gencad/CLAUDE.md`: every experiment, with the exact commands and numbers.
- `gencad/REPORT.md`: the story so far.
- `HANDOFF.md`: current state, spend, next steps.
