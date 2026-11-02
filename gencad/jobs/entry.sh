#!/usr/bin/env bash
# Entry point for every HF Job. Launched by jobs/launch.sh as:  bash /code/jobs/entry.sh <stage> [extra args]
#
# Mounts (set by launch.sh):
#   /code  this repo (read-only snapshot of your local checkout)
#   /work  read-write HF bucket that persists between jobs:
#          /work/data/{cad_vec,images,sketches}.tar, /work/data/*.json   dataset
#          /work/ckpt/                                                   upstream pretrained checkpoints
#          /work/emb/                                                    cached embeddings
#          /work/results/                                                all training runs
#
# Env knobs: CSR_CKPT, CCIP_CKPT, DP_CKPT (default: upstream pretrained in /work/ckpt), RUN (run name),
#            IMG_TYPE (cad_image|cad_sketch), OCC=1 to install pythonocc-core for STEP/STL export.
set -euo pipefail
STAGE=${1:?stage}; shift || true
RUN=${RUN:-$STAGE}
echo "=== stage=$STAGE run=$RUN $(date -u) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cp -r /code /app && cd /app
pip install -q -r jobs/requirements-job.txt
if [[ "${OCC:-0}" == "1" ]]; then conda install -y -q -c conda-forge pythonocc-core=7.9.0; fi
mkdir -p /work/data /work/ckpt /work/emb /work/results
rm -rf results && ln -s /work/results results
ln -sfn /work/emb data/emb
ln -sfn /work/ckpt data/ckpt
rm -rf model/ckpt && ln -sfn /work/ckpt model/ckpt   # upstream inference_gencad.py looks here
for f in /work/data/*.json; do [[ -e "$f" ]] && cp -f "$f" data/; done

# bucket FUSE is slow for 100k+ small files: unpack the archives this stage needs onto local disk
need() { for d in "$@"; do
  if [[ -f /work/data/$d.tar ]]; then echo "[data] unpack $d"; tar -xf /work/data/$d.tar -C data; else echo "[data] missing /work/data/$d.tar"; fi
done; }

# CadQuery (OCP) needs system GL libraries the pytorch image does not ship
need_cq() {
  python -c "import cadquery" 2>/dev/null && return
  apt-get update -qq && apt-get install -y -qq libgl1 libglu1-mesa libxrender1 libxext6 >/dev/null
  pip install -q cadquery pebble
  python -c "import cadquery; print('[cadquery]', cadquery.__version__)"
}

# vLLM: triton JIT-compiles kernels at engine start, so the image needs a C compiler
need_vllm() {
  python -c "import vllm" 2>/dev/null && return
  apt-get update -qq && apt-get install -y -qq gcc g++ >/dev/null
  if [[ "${VLLM_NIGHTLY:-0}" == "1" ]]; then   # models newer than the last vLLM release
    pip install -q -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly
  else
    pip install -q vllm
  fi
  python -c "import vllm, torch, transformers; print('[vllm]', vllm.__version__, torch.__version__, transformers.__version__)"
}

first() { for p in "$@"; do [[ -e "$p" ]] && { echo "$p"; return; }; done; echo "${1}"; }
CSR_CKPT=${CSR_CKPT:-$(first data/ckpt/ae_ckpt_epoch1000.pth $(ls data/ckpt/*ae*.pth 2>/dev/null))}
CCIP_CKPT=${CCIP_CKPT:-$(first data/ckpt/ccip_sketch_ckpt_epoch300.pth $(ls data/ckpt/*ccip*.pth 2>/dev/null))}
# the released CCIP/DP checkpoints come in sketch (Canny edges of renders) and image flavours; keep them paired
if [[ "$CCIP_CKPT" == *sketch* ]]; then IMG_TYPE=${IMG_TYPE:-cad_sketch}; DP_KIND=sketch; else IMG_TYPE=${IMG_TYPE:-cad_image}; DP_KIND=image; fi
DP_CKPT=${DP_CKPT:-$(shopt -s nullglob; first data/ckpt/${DP_KIND}_cond_diffusion*.pt data/ckpt/*diffusion*.pt none)}
echo "[ckpt] CSR=$CSR_CKPT CCIP=$CCIP_CKPT DP=$DP_CKPT IMG_TYPE=$IMG_TYPE"

case "$STAGE" in
  fetch)      # Drive + Text2CAD -> bucket. CPU flavor is enough.
    python scripts/fetch_data.py --data /tmp/d
    for d in cad_vec images sketches multi_view_images; do
      [[ -d /tmp/d/$d ]] && { echo "[fetch] tar $d"; tar -cf /work/data/$d.tar -C /tmp/d $d; }
    done
    cp -f /tmp/d/*.json /work/data/ 2>/dev/null || true
    # fetch_data.py symlinks checkpoints into model/ckpt, which is /work/ckpt here: replace the links with files
    for f in /tmp/d/ckpt/*; do rm -f "/work/ckpt/$(basename "$f")"; cp -f "$f" /work/ckpt/; done
    ls -la /work/data /work/ckpt ;;

  text)       # Text2CAD annotations -> text_index.json (needs access to the gated SadilKhan/Text2CAD)
    # v1.1 only: the repo also has v1.0, whose prompts are noisier, and merging both would mix them per uid
    python scripts/build_text_index.py --hf_repo SadilKhan/Text2CAD --hf_patterns "text2cad_v1.1/*.csv" \
      --split_json data/filtered_data.json \
      --out /work/data/text_index.json "$@" ;;

  test)       # unit + smoke tests on synthetic data, real encoders included
    MM_TEST_ENCODERS=1 python -m pytest tests/test_mm_smoke.py -q -s "$@" ;;

  csr)        # replication stage 1: CSR autoencoder
    need cad_vec
    python train_gencad.py csr -name "$RUN" -gpu 0 -epochs ${EPOCHS:-1000} -bs ${BS:-512} -lr ${LR:-1e-3} "$@" ;;

  ccip)       # replication stage 2: contrastive CAD-image pretraining
    need cad_vec images sketches
    python train_gencad.py ccip -name "$RUN" -gpu 0 -cad_ckpt "$CSR_CKPT" -img_type "$IMG_TYPE" \
      -epochs ${EPOCHS:-300} -bs ${BS:-128} "$@" ;;

  export)     # uid-aligned CSR latents (+ CCIP image latents)
    need cad_vec images sketches
    python scripts/export_gencad_embeddings.py --data data --csr_ckpt "$CSR_CKPT" --ccip_ckpt "$CCIP_CKPT" \
      --img_type "$IMG_TYPE" --out data/emb "$@" ;;

  embed)      # frozen SigLIP2 image + Qwen3 text embeddings
    need images
    python scripts/embed_conditions.py image --data data --out data/emb/cond_image.h5 "$@"
    if [[ -f data/text_index.json ]]; then
      python scripts/embed_conditions.py text --text_index data/text_index.json --out data/emb/cond_text.h5
    else echo "[embed] no data/text_index.json -> skipping text"; fi ;;

  dp)         # replication stage 3: upstream diffusion prior, on correctly aligned arrays
    python scripts/make_dp_arrays.py --emb_dir data/emb --split train --out_dir data/dp
    python train_gencad.py dp -name "$RUN" -gpu 0 -cad_emb data/dp/cad_zs.h5 -img_emb data/dp/img_zs.h5 \
      -steps ${STEPS:-1000000} -save_every ${SAVE_EVERY:-50000} "$@" ;;

  ar)         # autoregressive CAD decoder on the frozen CSR latents (mm/ar_decoder.py)
    need cad_vec
    python train_ar_decoder.py --emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" --name "$RUN" \
      --steps ${STEPS:-100000} "$@" ;;

  e2e)        # fine-tune the SigLIP2 image encoder end to end with a trained prior (train_e2e.py)
    need images
    python train_e2e.py --init_prior "${INIT_PRIOR:-results/mm_v1/last.pt}" --emb_dir data/emb --data data \
      --name "$RUN" --steps ${STEPS:-20000} "$@" ;;

  textcad)    # text -> CAD program decoded from prompt tokens (train_text_cad.py; needs programs.npz in /work/emb)
    need cad_vec
    python train_text_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-40000} "$@" ;;

  tokencad)   # image (+ text) -> CAD program decoded from patch/prompt tokens (train_token_cad.py)
    need cad_vec images
    python train_token_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-40000} "$@" ;;

  vlmcad)     # Qwen3-VL fine-tuned to write CAD programs from renders / prompts (train_vlm_cad.py)
    need cad_vec
    if [[ " $* " == *" --images images "* ]]; then need images; else need renders; fi
    python train_vlm_cad.py --emb_dir data/emb --data data --name "$RUN" --steps ${STEPS:-15000} "$@" ;;

  vlmcq)      # Qwen3-VL writing CadQuery, DeepCAD (converted) + CAD-Recode (train_vlm_cq.py); --predict for test preds
    need_cq
    need cad_vec renders recode_renders_0 recode_renders_1 recode_renders_2 recode_renders_3
    cp /work/data/deepcad_cq.parquet /work/data/cad_recode.parquet data/
    if [[ "${RICH:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    if [[ "${REV:-0}" == "1" ]]; then tar -xf /work/data/revision.tar -C data; fi
    if [[ "${REV_RICH:-0}" == "1" ]]; then tar -xf /work/data/rich_revision.tar -C data; fi
    if [[ "${REAL:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    if [[ "${TEACHER:-0}" == "1" ]]; then tar -xf /work/data/claude_teacher.tar -C data; fi   # scripts/build_teacher.py
    pip install -q -U "brotli>=1.2"   # the image's brotli breaks Hub downloads of base models (httpx2 decoder kwargs)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # long multi-image batches fragment memory (4B OOM)
    if [[ "${RAND:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_rand_*.tar; do tar -xf "$t" -C data/rich; done; fi
    python train_vlm_cq.py --data data --name "$RUN" --steps ${STEPS:-30000} "$@" ;;

  vlmrl)      # RL on IoU for the CadQuery writer (train_vlm_rl.py), DeepCAD parts only; INIT = SFT model dir
    need_cq
    need cad_vec renders
    cp /work/data/deepcad_cq.parquet data/
    python train_vlm_rl.py --data data --name "$RUN" --init "${INIT:-results/vlm_cq_v1/model}" --steps ${STEPS:-300} "$@" ;;

  rftsample)  # greedy + sampled programs per DeepCAD part with vLLM, scored vs GT (scripts/sample_rft.py); INIT = model dir
    need_cq
    need cad_vec renders
    cp /work/data/deepcad_cq.parquet data/
    need_vllm
    python scripts/sample_rft.py --data data --model "${INIT:-results/vlm_cq_v1/model}" --out "results/$RUN" "$@" ;;

  vlmdpo)     # DPO on rejection-sampling pairs (train_vlm_dpo.py); INIT = policy/reference init, CHUNKS = sample_rft output
    need_cq
    need cad_vec renders
    cp /work/data/deepcad_cq.parquet data/
    python train_vlm_dpo.py --data data --name "$RUN" --init "${INIT:-results/vlm_rft_v1/model}" \
        --chunks "${CHUNKS:-results/rft_v1}" ${STEPS:+--steps $STEPS} "$@" ;;

  vlmgrpo)    # online RL: vLLM sampling + in-process weight sync (train_vlm_grpo.py); INIT = start model
    need_cq
    need cad_vec renders
    cp /work/data/deepcad_cq.parquet data/
    need_vllm
    if [[ "${RICH:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    if [[ "${REAL:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    if [[ "${RAND:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_rand_*.tar; do tar -xf "$t" -C data/rich; done; fi
    python train_vlm_grpo.py --data data --name "$RUN" --init "${INIT:-results/vlm_rft_v1/model}" ${STEPS:+--steps $STEPS} "$@" ;;

  revise)     # build-inspect-fix inference (scripts/revise_vllm.py) -> results/$RUN/revise.pkl; INIT = model dir
    need renders
    if [[ "${RICH_TEST:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/test*.tar; do tar -xf "$t" -C data/rich; done; fi
    need_cq
    need_vllm
    python scripts/revise_vllm.py --data data --model "${INIT:?set INIT=<model dir>}" \
        --uids_from results/baselines_vlm_final/preds.pkl --out "results/$RUN/revise.pkl" "$@" ;;

  vllmpred)   # greedy test predictions with vLLM (scripts/predict_vllm.py) -> results/$RUN/preds.pkl; INIT = model dir
    need renders            # RICH_TEST=1: also unpack the rich test shards (pass --rich data/rich/test --out ...)
    if [[ "${RICH_TEST:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/test*.tar; do tar -xf "$t" -C data/rich; done; fi
    need_vllm
    python scripts/predict_vllm.py --data data --model "${INIT:?set INIT=<model dir>}" \
        --uids_from results/baselines_vlm_final/preds.pkl --out "results/$RUN/preds.pkl" "$@" ;;

  revdata)    # revision-turn training data from sample_rft chunks (scripts/build_revision.py) -> /work/data/revision.tar
    need_cq
    cp /work/data/deepcad_cq.parquet data/
    python scripts/build_revision.py --chunks "${CHUNKS:-results/rft_v1}" --out data/revision "$@"
    tar -cf /work/data/revision.tar -C data revision && ls -la /work/data/revision.tar ;;

  richrev)    # revision-turn data from rich parts: model attempts vs generator programs (scripts/sample_rich_rev.py)
    need_cq     # INIT = model dir; RICH_SHARDS = train shards to sample from -> /work/data/rich_revision.tar
    need_vllm
    mkdir -p data/rich; for sh in ${RICH_SHARDS:-0 1 2 3 20 21}; do tar -xf "/work/data/rich/train_$sh.tar" -C data/rich; done
    python scripts/sample_rich_rev.py --model "${INIT:?set INIT=<model dir>}" --rich data/rich/train --out data/rich_revision "$@"
    tar -cf /work/data/rich_revision.tar -C data rich_revision && ls -la /work/data/rich_revision.tar ;;

  richgen)    # rich-feature parts (datagen/families) -> /work/data/rich/<split>_<shard>.tar (scripts/gen_rich_data.py)
    need_cq
    SPLIT=${SPLIT:-train}
    python scripts/gen_rich_data.py --split "$SPLIT" --out "data/rich/$SPLIT" "$@"
    SH=$(ls data/rich/$SPLIT)
    mkdir -p /work/data/rich && tar -cf "/work/data/rich/${SPLIT}_${SH}.tar" -C data/rich "$SPLIT/$SH" && ls -la "/work/data/rich/${SPLIT}_${SH}.tar" ;;

  vlmpred)    # test-set CadQuery predictions of a vlm_cq-style model dir (INIT) -> results/$RUN/preds.pkl
    need renders
    python train_vlm_cq.py --data data --predict "${INIT:?set INIT=<model dir>}" \
        --uids_from results/baselines_vlm_final/preds.pkl --out_preds "results/$RUN/preds.pkl" "$@" ;;

  mm)         # the new multimodal prior
    python train_mm_prior.py --emb_dir data/emb --name "$RUN" ${HUB_REPO:+--hub_repo $HUB_REPO} "$@" ;;

  eval)
    need cad_vec
    ARGS=(--emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" --out "results/$RUN/eval.json")
    [[ -f "results/$RUN/last.pt" ]] && ARGS+=(--prior "results/$RUN/last.pt")
    [[ "$DP_CKPT" != none && -f "$DP_CKPT" ]] && ARGS+=(--gencad_dp_ckpt "$DP_CKPT")
    python eval_mm.py "${ARGS[@]}" "$@" ;;

  upstream)   # released GenCAD through upstream's own inference path, scored like eval_mm (cross-check of gencad_dp)
    need cad_vec images sketches
    python scripts/eval_upstream_inference.py --data data --emb_dir data/emb --ckpt_dir data/ckpt \
      --out "results/$RUN/eval.json" "$@" ;;

  baselines)  # train/test duplication + nearest-neighbour baselines, and priors split into duplicated vs novel parts
    need cad_vec
    [[ " $* " == *" --token_cad "* ]] && need images
    [[ " $* " == *" --vlm_cad "* ]] && need renders
    python scripts/baselines.py --emb_dir data/emb --data data --csr_ckpt "$CSR_CKPT" \
      --out "results/$RUN/report.json" "$@" ;;

  geom)       # IoU / Chamfer / invalidity of saved predictions (scripts/geom_eval.py; installs OCC itself)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    need cad_vec
    python scripts/geom_eval.py --data data --preds "${PREDS:-results/$RUN/preds.pkl}" --out "results/$RUN/geom.json" "$@" ;;

  render)     # our own fixed-camera renders of every GT program -> /work/data/renders.tar (scripts/render_views.py)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    pip install -q pebble
    need cad_vec
    python scripts/render_views.py --data data --out data/renders "$@"
    if [[ " $* " == *" --limit "* ]]; then mkdir -p "results/$RUN" && cp -r data/renders "results/$RUN/"   # preview
    else tar -cf /work/data/renders.tar -C data renders && ls -la /work/data/renders.tar; fi ;;

  fetchrecode) # CAD-Recode v1.5 (~1M CadQuery programs) -> /work/data/cad_recode.parquet
    command -v git >/dev/null || { apt-get update -qq && apt-get install -y -qq git >/dev/null; }
    python scripts/fetch_cad_recode.py --out /work/data/cad_recode.parquet "$@" ;;

  renderrecode) # CAD-Recode programs -> renders with our cameras; one shard per job -> /work/data/<RUN>.tar
    need_cq
    cp /work/data/cad_recode.parquet data/
    python scripts/render_cad_recode.py --parquet data/cad_recode.parquet --out data/recode_renders "$@"
    tar -cf "/work/data/$RUN.tar" -C data recode_renders && ls -la "/work/data/$RUN.tar" ;;

  convertdc)  # DeepCAD programs -> CadQuery source + fidelity vs cadlib -> /work/data/deepcad_cq.parquet
    need_cq
    need cad_vec
    python scripts/convert_deepcad.py --data data --out /work/data/deepcad_cq.parquet "$@" ;;

  realgen)    # real-part training shard from the Fusion 360 Gallery STEP release (scripts/prep_real.py)
    need_cq     # SHARD/N_SHARDS -> /work/data/rich/real_train_<shard>.tar; test designs excluded
    python -c "import urllib.request as u; u.urlretrieve('https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/segmentation/s2.0.1/s2.0.1_extended_step.zip', '/tmp/f360.zip')"
    python scripts/prep_real.py --zip /tmp/f360.zip --out data/rich/real_train --shard "${SHARD:-0}" --n_shards "${N_SHARDS:-4}" \
        --exclude_designs datagen/real_test_designs.txt "$@"
    mkdir -p /work/data/rich && tar -cf "/work/data/rich/real_train_${SHARD:-0}.tar" -C data/rich "real_train/${SHARD:-0}" \
        && ls -la "/work/data/rich/real_train_${SHARD:-0}.tar" ;;

  mistral)    # zero-shot open-weight VLM via vLLM on all GPUs (scripts/mistral_real.py); INIT = HF model id
    need_vllm   # RICH_TEST=1 unpacks the rich/real test shards; e.g. --rich data/rich/real --modes view4
    if [[ "${RICH_TEST:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/test*.tar; do tar -xf "$t" -C data/rich; done; fi
    export HF_XET_HIGH_PERFORMANCE=1
    python scripts/mistral_real.py --model "${INIT:?set INIT=<hf model id>}" --out "results/$RUN/preds.pkl" "$@" ;;

  richiou)    # score a rich/real preds file (scripts/rich_iou.py) -> results/$RUN/rich_{iou.json,rows.pkl}
    need_cq     # REAL=1 unpacks the real training parts (GT STEP files), RICH_TEST=1 the test shards
    if [[ "${REAL:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    if [[ "${RICH_TEST:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/test*.tar; do tar -xf "$t" -C data/rich; done; fi
    python scripts/rich_iou.py --preds "results/$RUN/preds.pkl" --out "results/$RUN/rich_iou.json" --rows "results/$RUN/rich_rows.pkl" "$@" ;;

  selfsample) # our model's greedy + sampled programs on rich/real train parts (scripts/sample_real.py); INIT = model dir
    need_vllm   # REAL=1 unpacks the real training parts; writes results/$RUN_<shard>/preds.pkl (score each with richiou)
    if [[ "${REAL:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    python scripts/sample_real.py --model "${INIT:?set INIT=<model dir>}" --out "results/$RUN" "$@" ;;

  variants)   # randomised re-renders of rich/real parts (scripts/render_variants.py) -> /work/data/rich/real_rand_<shard>.tar
    need_cq     # REAL=1 unpacks the real training parts; SHARD = which shard dir to render
    if [[ "${REAL:-0}" == "1" ]]; then mkdir -p data/rich; for t in /work/data/rich/real_train_*.tar; do tar -xf "$t" -C data/rich; done; fi
    python scripts/render_variants.py --src data/rich/real_train --out data/rich/real_rand --shards "${SHARD:-0}" "$@"
    (cd data/rich && tar -cf "/work/data/rich/real_rand_${SHARD:-0}.tar" real_rand/*/"${SHARD:-0}") && ls -la "/work/data/rich/real_rand_${SHARD:-0}.tar" ;;

  orient)     # orientation search on saved predictions (scripts/orient_fix.py) + scoring -> results/$RUN/orient*.{pkl,json}
    need_cq     # RICH_TEST=1: rich/real test parts (scored by rich_iou.py); else DeepCAD test (cq_iou.py)
    if [[ "${RICH_TEST:-0}" == "1" ]]; then
      mkdir -p data/rich; for t in /work/data/rich/test*.tar; do tar -xf "$t" -C data/rich; done
      python scripts/orient_fix.py --data data --out "results/$RUN/orient.pkl" "$@"
      python scripts/rich_iou.py --preds "results/$RUN/orient.pkl" --out "results/$RUN/orient_iou.json" --rows "results/$RUN/orient_rows.pkl"
    else
      need cad_vec renders
      python scripts/orient_fix.py --data data --out "results/$RUN/orient.pkl" "$@"
      python scripts/cq_iou.py --data data --preds "results/$RUN/orient.pkl" --out "results/$RUN/orient_iou.json" --rows "results/$RUN/orient_rows.pkl"
    fi ;;

  cqiou)      # IoU of saved predictions (DeepCAD programs or CadQuery code) vs GT, in CadQuery (scripts/cq_iou.py)
    need_cq
    need cad_vec
    python scripts/cq_iou.py --data data --preds "${PREDS:-results/$RUN/preds.pkl}" --out "results/$RUN/cq_iou.json" "$@" ;;

  checkviews) # do all 5 renders of a uid show its GT program? (scripts/check_views.py; installs OCC itself)
    [[ "${OCC:-0}" == "1" ]] || conda install -y -q -c conda-forge pythonocc-core=7.9.0
    need cad_vec images
    python scripts/check_views.py --data data --out "results/$RUN/report.json" "$@" ;;

  visual)     # rendered side-by-side check (needs OCC=1): GT vs GenCAD vs the prior under each condition
    need cad_vec images
    PRIOR=${PRIOR:-results/mm_v1/last.pt}
    ARGS=(--prior "$PRIOR" --csr_ckpt "$CSR_CKPT" --emb_dir data/emb --data data --out "results/$RUN")
    [[ "$DP_CKPT" != none && -f "$DP_CKPT" ]] && ARGS+=(--gencad_dp_ckpt "$DP_CKPT")
    python scripts/visual_check.py "${ARGS[@]}" "$@" ;;

  shell)      # arbitrary command, e.g. launch.sh shell -- python infer_mm.py ...
    [[ "${1:-}" == "--" ]] && shift
    "$@" ;;

  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "=== done $STAGE $(date -u) ==="
