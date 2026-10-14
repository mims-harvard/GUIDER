#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN:-python3}" "$ROOT/eval.py" \
  --model "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}" \
  ${ADAPTER_PATH:+--adapter "$ADAPTER_PATH"} \
  --data "${EVAL_FILE:?Set EVAL_FILE}" \
  --output "${OUTPUT_FILE:-outputs/jointopt/predictions.jsonl}" \
  ${MAX_NEW_TOKENS:+--max-new-tokens "$MAX_NEW_TOKENS"} \
  ${EVAL_BATCH_SIZE:+--batch-size "$EVAL_BATCH_SIZE"} "$@"
