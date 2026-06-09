#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/coda_llava_7b.yaml}"

CODA_DATA_ROOT="${CODA_DATA_ROOT:-/path/to/CODA-data}"
CODA_LM_ROOT="${CODA_LM_ROOT:-${CODA_ROOT:-${CODA_DATA_ROOT}/CODA-LM}}"
TRAIN_ANN="${TRAIN_ANN:-${CODA_LM_ROOT}/Train/vqa_anno/region_perception.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${CODA_DATA_ROOT}}"
CLASS_EMBEDDINGS="${CLASS_EMBEDDINGS:-outputs/class_embeddings/class_embeddings_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/adapter_llava15}"

echo "Config: ${CONFIG}"
env PYTHONPATH="src:${PYTHONPATH:-}" python -m seeing.cli.train_adapter_llava15 \
  --train-annotation "${TRAIN_ANN}" \
  --image-root "${IMAGE_ROOT}" \
  --class-embeddings "${CLASS_EMBEDDINGS}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-3}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --lr "${LR:-0.0001}" \
  --llava-model-path "${LLAVA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}" \
  --llava-model-name "${LLAVA_MODEL_NAME:-llava-v1.5-7b}" \
  --conv-mode "${CONV_MODE:-llava_v1}"
