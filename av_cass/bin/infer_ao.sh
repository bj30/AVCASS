#!/usr/bin/env bash
set -euo pipefail
if ! command -v torchrun >/dev/null 2>&1; then
  echo "torchrun is not available on PATH. Activate the inference environment first." >&2
  exit 127
fi
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"
: "${DATASET_ROOT:?Set DATASET_ROOT to the corrected AVDnR dataset root.}"
: "${SAMPLE_DIR:?Set SAMPLE_DIR to the inference output directory.}"
: "${CKPT:?Set CKPT to the audio-only checkpoint path.}"
: "${NUM_GPUS:=1}"
: "${PER_PROC_BATCH_SIZE:=4}"
: "${MASTER_PORT:=29500}"
torchrun --nnodes=1 --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  test_ddp_avdnr_spec.py \
  --model UNet2d_S2 \
  --audio_files_dir "$DATASET_ROOT" \
  --sample-dir "$SAMPLE_DIR" \
  --load_whole 1 \
  --ckpt "$CKPT" \
  --per-proc-batch-size "$PER_PROC_BATCH_SIZE" \
  "$@"
