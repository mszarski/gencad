#!/usr/bin/env bash
# Launch one pipeline stage as a Hugging Face Job. Run from this gencad/ folder after ./setup.sh.
#
#   export HF_TOKEN=hf_...                 # needs Jobs + write access
#   jobs/launch.sh fetch                   # once: Drive + Text2CAD -> your bucket (CPU)
#   jobs/launch.sh text                    # Text2CAD index only (if fetch ran before access was granted)
#   jobs/launch.sh test                    # smoke tests on a GPU
#   jobs/launch.sh export                  # embeddings with the released GenCAD checkpoints
#   jobs/launch.sh embed
#   RUN=mm_v1 jobs/launch.sh mm --modalities image text
#   RUN=mm_v1 jobs/launch.sh eval
#   RUN=ar_v1 jobs/launch.sh ar               # autoregressive decoder; then eval/visual with --ar_decoder
#   OCC=1 RUN=visual_v1 jobs/launch.sh visual    # rendered GT-vs-generation grids (PRIOR=results/mm_v1/last.pt)
#   RUN=repl jobs/launch.sh csr            # full replication from scratch: csr -> ccip -> export -> dp -> eval
#
# Overrides: FLAVOR, TIMEOUT, BUCKET (default <you>/gencad-work), IMAGE, plus any env knob from entry.sh
# (RUN, EPOCHS, BS, LR, STEPS, CSR_CKPT, CCIP_CKPT, DP_CKPT, IMG_TYPE, OCC, HUB_REPO). DRY=1 prints the command.
set -euo pipefail
STAGE=${1:?usage: jobs/launch.sh <fetch|text|test|csr|ccip|export|embed|dp|mm|eval|ar|e2e|textcad|tokencad|vlmcad|vlmcq|vlmrl|vlmpred|vllmpred|revise|rftsample|vlmdpo|vlmgrpo|richgen|revdata|richrev|orient|realgen|mistral|richiou|selfsample|variants|upstream|baselines|geom|checkviews|render|fetchrecode|renderrecode|convertdc|cqiou|visual|shell> [args...]}; shift
: "${HF_TOKEN:?export HF_TOKEN first}"
NS=${NS:-$(hf auth whoami --format json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name") or d["user"])')}
BUCKET=${BUCKET:-$NS/gencad-work}
IMAGE=${IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}

# per-stage defaults: hardware and wall-clock limit
case "$STAGE" in
  fetch)        DF=cpu-upgrade; DT=6h ;;
  text)         DF=cpu-upgrade; DT=1h ;;
  test)         DF=t4-small;    DT=1h ;;
  export|embed) DF=a10g-large;  DT=4h ;;
  csr)          DF=a100-large;  DT=48h ;;
  ccip)         DF=a100-large;  DT=24h ;;
  dp|mm|ar|e2e|textcad|tokencad) DF=a100-large;  DT=24h ;;
  vlmcad|vlmcq|vlmrl|rftsample|vlmdpo|vlmgrpo|richrev) DF=h200;        DT=24h ;;
  vlmpred)      DF=a100-large;  DT=3h ;;
  vllmpred)     DF=a100-large;  DT=1h ;;
  selfsample)   DF=a100-large;  DT=4h ;;
  mistral)      DF=h200x8;      DT=3h ;;
  richiou)      DF=cpu-xl;      DT=4h ;;
  revise)       DF=a100-large;  DT=3h ;;
  eval)         DF=a10g-large;  DT=4h ;;
  geom)         DF=cpu-performance; DT=8h ;;
  checkviews)   DF=cpu-upgrade; DT=2h ;;
  render)       DF=cpu-performance; DT=8h ;;
  fetchrecode)  DF=cpu-upgrade; DT=3h ;;
  renderrecode) DF=cpu-performance; DT=12h ;;
  convertdc)    DF=cpu-performance; DT=4h ;;
  cqiou)        DF=cpu-xl;      DT=6h ;;
  richgen|revdata|orient|realgen|variants) DF=cpu-upgrade; DT=12h ;;
  *)            DF=a10g-large;  DT=4h ;;
esac
FLAVOR=${FLAVOR:-$DF}; TIMEOUT=${TIMEOUT:-$DT}

# The job gets a dereferenced snapshot of this folder: setup.sh symlinks upstream GenCAD in, and symlinks
# would not survive the upload. Large local outputs are left out (they live in the bucket).
[[ -d upstream ]] || { echo "run ./setup.sh first"; exit 1; }
SNAP=$(mktemp -d)/code
mkdir -p "$SNAP"
tar -chf - --exclude=./upstream --exclude=.git --exclude=__pycache__ --exclude=./results \
    --exclude=./data/emb --exclude=./data/dp --exclude=./data/cad_vec --exclude=./data/images \
    --exclude=./data/sketches . | tar -xf - -C "$SNAP"   # -h follows the setup.sh symlinks

ENVS=()
for k in INIT CHUNKS SPLIT RICH RICH_TEST REV REV_RICH RICH_SHARDS REAL SHARD N_SHARDS VLLM_NIGHTLY TEACHER RAND PRIOR INIT_PRIOR PREDS RUN EPOCHS BS LR STEPS SAVE_EVERY CSR_CKPT CCIP_CKPT DP_CKPT IMG_TYPE OCC HUB_REPO; do
  [[ -n "${!k:-}" ]] && ENVS+=(--env "$k=${!k}")
done

CMD=(hf jobs run --flavor "$FLAVOR" --timeout "$TIMEOUT" --secrets HF_TOKEN
     -v "$SNAP:/code" -v "hf://buckets/$BUCKET:/work"
     --name "gencad-$STAGE${RUN:+-$RUN}" --label project=gencad-mm --label stage="$STAGE"
     ${ENVS[@]+"${ENVS[@]}"} --detach
     "$IMAGE" bash /code/jobs/entry.sh "$STAGE" "$@")
echo "${CMD[*]}"
[[ "${DRY:-0}" == "1" ]] && exit 0
hf buckets create "$BUCKET" --private --exist-ok >/dev/null
"${CMD[@]}"
echo "follow with: hf jobs logs <job_id> --follow"
