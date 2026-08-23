#!/usr/bin/env bash
# PixWorld stage 1 -- fine-tune Wan2.2-TI2V-5B into a pixel-space multi-view 3D generator.
#
#   Single node:   bash train/train_l2p.sh
#   Multi node:    NNODES=4 NODE_RANK=$i MASTER_ADDR=<host> bash train/train_l2p.sh
#
# Every setting below can be overridden from the environment, e.g.
#   MAX_STEPS=100000 LR=3e-5 bash train/train_l2p.sh
#
# CKPT_LPIPS, CKPT_GS_DEC and CKPT_GS_HEAD trade recompute for memory. Leave them at 1 at
# 16 views. GS_HEAD_CHUNK and LPIPS_CHUNK are further levers that are not numerically free;
# leave them at 0 unless you are out of memory.
#
# The geometry perception loss is OFF by default (GEO_LOSS=none). Turning it on adds a
# frozen ~1B-parameter multi-view backbone with gradients, and neither backbone ships with
# PixWorld:
#   GEO_LOSS=pi3  GEO_REPO=/path/to/Pi3  GEO_WEIGHTS=/path/to/Pi3/checkpoints/model.safetensors
#   GEO_LOSS=vggt GEO_REPO=/path/to/vggt GEO_WEIGHTS=facebook/VGGT-1B
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# ------------------------------------------------------------------ what to train on --
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your dataset root (see train/DATASET.md)}
WAN_PATH=${WAN_PATH:-./weights/Wan2.2-TI2V-5B-Diffusers}
OUT_DIR=${OUT_DIR:-./runs/l2p}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
mkdir -p "$OUT_DIR" "$LOG_DIR"

if [ ! -d "$WAN_PATH/transformer" ]; then
  echo "ERROR: $WAN_PATH does not look like a Wan2.2-TI2V-5B-Diffusers checkout."
  echo "  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir $WAN_PATH"
  exit 1
fi

# ------------------------------------------------------------------------ topology --
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# ------------------------------------------------------------------- resume policy --
# An explicit RESUME wins; otherwise pick up the newest checkpoint in OUT_DIR; otherwise
# cold-start from Wan2.2. Sorting is numeric, so step9000 does not beat step10000.
if [ -z "${RESUME:-}" ]; then
  RESUME=$(ls "$OUT_DIR"/model_step*.pt 2>/dev/null \
    | sed -E 's/.*model_step([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
fi
if [ -n "${RESUME:-}" ]; then
  echo "=== resuming from $RESUME ==="
  RESUME_ARG=(--resume "$RESUME")
else
  echo "=== cold start from $WAN_PATH ==="
  RESUME_ARG=()
fi

# ---------------------------------------------------------------------- environment --
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"

# gsplat JIT-compiles its CUDA extension on the first rasterization() call, not at import.
# Compile it once here, single-process, before torchrun.
rm -f "$TORCH_EXTENSIONS_DIR"/*/gsplat_cuda/lock 2>/dev/null || true
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extension warmed')"
python - <<'PY'
import torch, diffusers
print(f"[pixworld] torch {torch.__version__} | diffusers {diffusers.__version__}")
PY

# ------------------------------------------------------------------------- training --
python -m torch.distributed.run \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --log-dir "$LOG_DIR" --tee 3 \
  train/train_l2p.py \
    --data_root "$DATA_ROOT" --wan_path "$WAN_PATH" --out_dir "$OUT_DIR" \
    "${RESUME_ARG[@]}" \
    --views ${VIEWS:-16} --k_novel ${K_NOVEL:-16} --n_target ${N_TARGET:-16} \
    --both_ends_prob ${BOTH_ENDS_PROB:-1.0} \
    --height ${HEIGHT:-480} --width ${WIDTH:-832} \
    --batch_size ${BATCH_SIZE:-1} --workers ${WORKERS:-6} \
    --source_weights "${SOURCE_WEIGHTS:-}" \
    --shift ${SHIFT:-16.0} --sigma_lo_frac ${SIGMA_LO_FRAC:-0.2} \
    --gs_sigma_hi ${GS_SIGMA_HI:-0.5} \
    --pure_noise_prob ${PURE_NOISE:-0.1} --pure_clean_prob ${PURE_CLEAN:-0.05} \
    --i2mv_frac ${I2MV_FRAC:-0.5} --cond_drop_prob ${COND_DROP:-0.1} \
    --gamma_ip ${GAMMA_IP:-0.1} \
    --render_w ${RENDER_W:-1.0} --render_lpips_w ${RENDER_LPIPS_W:-0.1} \
    --depth_w ${DEPTH_W:-1.0} --depth_tv_w ${DEPTH_TV_W:-0.0} \
    --opacity_w ${OPACITY_W:-0.0} \
    --anchor_w_in_gs ${ANCHOR_W_IN_GS:-1.0} --anchor_lpips_in_gs ${ANCHOR_LPIPS_IN_GS:-0} \
    --lpips_weight ${LPIPS_W:-0.1} --lpips_gate ${LPIPS_GATE:-0.7} \
    --gs_sigma_weight ${GS_SIGMA_WEIGHT:-inv_sigma} --gs_sigma_min ${GS_SIGMA_MIN:-0.05} \
    --geo_loss ${GEO_LOSS:-none} --geo_w ${GEO_W:-0.05} --geo_gate ${GEO_GATE:-0.7} \
    --geo_weights "${GEO_WEIGHTS:-}" --geo_repo "${GEO_REPO:-}" \
    --geo_resolution ${GEO_RESOLUTION:-224} --geo_taps ${GEO_TAPS:-1} \
    --geo_dist ${GEO_DIST:-mse} \
    --geo_max_views ${GEO_MAX_VIEWS:-0} \
    --lr ${LR:-5e-5} --lr_min ${LR_MIN:-5e-6} --lr_warmup ${LR_WARMUP:-2000} \
    --lr_decay ${LR_DECAY:-cosine} --max_steps ${MAX_STEPS:-50000} \
    --weight_decay ${WD:-0.01} --gs_wd ${GS_WD:-1e-6} --grad_clip ${GRAD_CLIP:-1.0} \
    --ema_decay ${EMA_DECAY:-0.999} \
    --n_edge_blocks ${N_EDGE_BLOCKS:-5} --train_patchify ${TRAIN_PATCHIFY:-1} \
    --ckpt_blocks ${CKPT_BLOCKS:-1} --ckpt_gs_dec ${CKPT_GS_DEC:-1} \
    --ckpt_gs_head ${CKPT_GS_HEAD:-1} --ckpt_lpips ${CKPT_LPIPS:-1} \
    --gs_head_chunk ${GS_HEAD_CHUNK:-0} --lpips_chunk ${LPIPS_CHUNK:-0} \
    --rope_precast ${ROPE_PRECAST:-1} --gs_depth_max ${GS_DEPTH_MAX:-50.0} \
    --log_every ${LOG_EVERY:-20} --save_every ${SAVE_EVERY:-2000} \
    --keep_ckpts ${KEEP_CKPTS:-6} --save_optim ${SAVE_OPTIM:-1} \
    --sample_every ${SAMPLE_EVERY:-2000} --sample_steps ${SAMPLE_STEPS:-50} \
    --sample_cfg ${SAMPLE_CFG:-5.0} --sample_rescale ${SAMPLE_RESCALE:-0.7} \
    --sample_sigma_switch ${SAMPLE_SIGMA_SWITCH:-0.5} --sample_shift ${SAMPLE_SHIFT:-16.0} \
    --n_val_scenes ${N_VAL_SCENES:-2} --n_val_prompts ${N_VAL_PROMPTS:-2} \
    --seed ${SEED:-0} \
    --wandb ${WANDB:-0} --wandb_project ${WANDB_PROJECT:-pixworld} \
    --wandb_name "${WANDB_NAME:-l2p}"
