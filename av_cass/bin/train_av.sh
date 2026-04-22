#!/usr/bin/env bash
set -euo pipefail
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate is not available on PATH. Activate the training environment first." >&2
  exit 127
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"
: "${RESULTS_DIR:?Set RESULTS_DIR to the training output directory.}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the corrected AVDnR dataset root.}"
: "${INIT_CKPT:?Set INIT_CKPT to the stage-1 AO checkpoint.}"
: "${CAVP_CKPT:?Set CAVP_CKPT to the CAVP checkpoint.}"
: "${TALKNET_CKPT:?Set TALKNET_CKPT to the TalkNet checkpoint.}"
: "${VISUAL_ENCODER_TYPE:=both}"
: "${ATTENTION_HEAD_DIM:=64}"
: "${NUM_PROCESSES:=4}"
: "${MAIN_PROCESS_PORT:=29500}"
accelerate launch --multi_gpu --num_processes "$NUM_PROCESSES" --main_process_port "$MAIN_PROCESS_PORT" --mixed_precision fp16 \
  train_avdnr_fp16_spec_RFM_av.py \
  --results-dir "$RESULTS_DIR" \
  --audio_files_dir "$DATASET_ROOT" \
  --visual_encoder_type "$VISUAL_ENCODER_TYPE" \
  --attention_head_dim "$ATTENTION_HEAD_DIM" \
  --init-ckpt "$INIT_CKPT" \
  "$@"
