#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON_BIN:-python3}" "$ROOT/train.py" \
  --model "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}" \
  --train-file "${TRAIN_FILE:?Set TRAIN_FILE}" \
  --output-dir "${OUTPUT_DIR:-outputs/jointopt}" \
  ${EPOCHS:+--epochs "$EPOCHS"} ${LEARNING_RATE:+--lr "$LEARNING_RATE"} \
  ${BATCH_SIZE:+--batch-size "$BATCH_SIZE"} ${GRADIENT_ACCUMULATION_STEPS:+--grad-accum "$GRADIENT_ACCUMULATION_STEPS"} \
  ${NUM_GENERATIONS:+--generations "$NUM_GENERATIONS"} ${MAX_PROMPT_LENGTH:+--max-prompt "$MAX_PROMPT_LENGTH"} \
  ${MAX_COMPLETION_LENGTH:+--max-completion "$MAX_COMPLETION_LENGTH"} \
  ${CONTROLLER_SAMPLES:+--controller-samples "$CONTROLLER_SAMPLES"} \
  ${CONTROLLER_TEMPERATURE:+--controller-temperature "$CONTROLLER_TEMPERATURE"} \
  ${MC_TEMPERATURE:+--mc-temperature "$MC_TEMPERATURE"} \
  ${CONTROLLER_LOSS_WEIGHT:+--controller-weight "$CONTROLLER_LOSS_WEIGHT"} \
  ${FLOW_REWARD_WEIGHT:+--flow-weight "$FLOW_REWARD_WEIGHT"} "$@"
