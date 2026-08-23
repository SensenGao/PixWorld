#!/usr/bin/env bash
# PixWorld stage 2 -- distil the 50-step model into a 4-step one (DMD2).
#
#   TEACHER=runs/l2p/model_step50000.pt bash train/train_dmd2.sh
#
# The teacher checkpoint seeds all three networks (student, critic, frozen teacher) and
# must contain a trained Gaussian stack.
#
# MAX_STEPS counts BOTH phases. At DIS_PER_GEN=4 the cycle is 5, so the default 10000
# steps is 2000 generator updates and 8000 critic updates.
#
# Three 5B transformers are resident at once: two sharded and trainable, the frozen teacher
# sharded and held in bf16 (SCORE_BF16=1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to your dataset root (see train/DATASET.md)}
WAN_PATH=${WAN_PATH:-./weights/Wan2.2-TI2V-5B-Diffusers}
TEACHER=${TEACHER:?set TEACHER to a stage-1 checkpoint, e.g. runs/l2p/model_step50000.pt}
OUT_DIR=${OUT_DIR:-./runs/dmd2}
LOG_DIR=${LOG_DIR:-${OUT_DIR}/logs}
mkdir -p "$OUT_DIR" "$LOG_DIR"

[ -f "$TEACHER" ] || { echo "ERROR: teacher checkpoint '$TEACHER' not found"; exit 1; }

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

# Resume the newest DMD checkpoint in OUT_DIR unless told otherwise. Note this is a
# distillation resume, NOT a change of teacher: TEACHER stays what it was.
if [ -z "${RESUME:-}" ]; then
  RESUME=$(ls "$OUT_DIR"/model_step*.pt 2>/dev/null \
    | sed -E 's/.*model_step([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2- || true)
fi
if [ -n "${RESUME:-}" ]; then
  echo "=== resuming distillation from $RESUME (teacher stays $TEACHER) ==="
  RESUME_ARG=(--resume "$RESUME")
else
  echo "=== starting distillation from $TEACHER ==="
  RESUME_ARG=()
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$HOME/.cache/torch_extensions}
mkdir -p "$TORCH_EXTENSIONS_DIR"
rm -f "$TORCH_EXTENSIONS_DIR"/*/gsplat_cuda/lock 2>/dev/null || true
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA extension warmed')"

python -m torch.distributed.run \
  --nnodes "$NNODES" --node_rank "$NODE_RANK" --nproc_per_node "$GPUS_PER_NODE" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  --log-dir "$LOG_DIR" --tee 3 \
  train/train_dmd2.py \
    --data_root "$DATA_ROOT" --wan_path "$WAN_PATH" --teacher "$TEACHER" \
    --out_dir "$OUT_DIR" "${RESUME_ARG[@]}" \
    --n_gen_steps ${N_GEN_STEPS:-4} --gen_shift ${GEN_SHIFT:-3.0} \
    --gs_step ${GS_STEP:--1} --step_last_w ${STEP_LAST_W:-0.4} \
    --dmd_sigma_hi ${DMD_SIGMA_HI:-0.98} --dmd_sigma_lo ${DMD_SIGMA_LO:-0.02} \
    --dmd_narrow_prob ${DMD_NARROW_PROB:-0.1} \
    --views ${VIEWS:-16} --k_novel ${K_NOVEL:-16} \
    --height ${HEIGHT:-480} --width ${WIDTH:-832} \
    --batch_size ${BATCH_SIZE:-1} --workers ${WORKERS:-6} \
    --source_weights "${SOURCE_WEIGHTS:-}" \
    --i2mv_frac ${I2MV_FRAC:-0.5} --cond_drop_prob ${COND_DROP:-0.1} \
    --task_w "${TASK_W:-1,3,1}" --real_cfg ${REAL_CFG:-3.0} \
    --consistency_w ${CONSISTENCY_W:-0.1} --depth_reg_w ${DEPTH_REG_W:-0.01} \
    --opacity_w ${OPACITY_W:-0.01} --sigma_min_convert ${SIGMA_MIN_CONVERT:-0.01} \
    --n_tok_embed ${N_TOK_EMBED:-32} \
    --dis_per_gen ${DIS_PER_GEN:-4} \
    --lr_gen ${LR_GEN:-1e-6} --lr_dis ${LR_DIS:-5e-7} \
    --max_steps ${MAX_STEPS:-10000} \
    --warmup_steps ${WARMUP_STEPS:--1} --decay_steps ${DECAY_STEPS:--1} \
    --weight_decay ${WD:-1e-6} --gs_wd ${GS_WD:-1e-6} --grad_clip ${GRAD_CLIP:-1.0} \
    --n_edge_blocks ${N_EDGE_BLOCKS:-5} --train_patchify ${TRAIN_PATCHIFY:-1} \
    --dis_train_all ${DIS_TRAIN_ALL:-1} --ema_decay ${EMA_DECAY:-0.0} \
    --ckpt_blocks ${CKPT_BLOCKS:-1} --ckpt_gs_dec ${CKPT_GS_DEC:-1} \
    --ckpt_gs_head ${CKPT_GS_HEAD:-1} --gs_head_chunk ${GS_HEAD_CHUNK:-0} \
    --rope_precast ${ROPE_PRECAST:-1} --gs_depth_max ${GS_DEPTH_MAX:-50.0} \
    --score_bf16 ${SCORE_BF16:-1} \
    --log_every ${LOG_EVERY:-20} --log_gen_every ${LOG_GEN_EVERY:-2} \
    --save_every ${SAVE_EVERY:-1000} --keep_ckpts ${KEEP_CKPTS:-4} \
    --save_optim ${SAVE_OPTIM:-1} \
    --sample_every ${SAMPLE_EVERY:-1000} --n_val_scenes ${N_VAL_SCENES:-2} \
    --seed ${SEED:-0} \
    --wandb ${WANDB:-0} --wandb_project ${WANDB_PROJECT:-pixworld-dmd2} \
    --wandb_name "${WANDB_NAME:-dmd2}"
