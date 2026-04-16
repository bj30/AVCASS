#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$SCRIPT_DIR"
: "${PRED_ROOT:?Set PRED_ROOT to the predicted output root.}"
: "${GT_ROOT:?Set GT_ROOT to the ground-truth dataset root.}"
: "${SYMLINK_ROOT:?Set SYMLINK_ROOT to a temporary symlink directory.}"
: "${RESULTS_OUT:?Set RESULTS_OUT to the evaluation output directory.}"
: "${MODEL_NAME:?Set MODEL_NAME for the summary JSON.}"

ARGS=(
  --pred-root "$PRED_ROOT"
  --gt-root "$GT_ROOT"
  --symlink-root "$SYMLINK_ROOT"
  --results-out "$RESULTS_OUT"
  --model-name "$MODEL_NAME"
)

if [[ -n "${DEVICE:-}" ]]; then
  ARGS+=(--device "$DEVICE")
fi

if [[ -n "${STEM:-}" ]]; then
  ARGS+=(--stem "$STEM")
fi

PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" python eval/eval_avdnr.py "${ARGS[@]}" "$@"
