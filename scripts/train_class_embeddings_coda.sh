#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/coda_llava_7b.yaml}"

CODA_DATA_ROOT="${CODA_DATA_ROOT:-/path/to/CODA-data}"
CODA_LM_ROOT="${CODA_LM_ROOT:-${CODA_ROOT:-${CODA_DATA_ROOT}/CODA-LM}}"
TRAIN_ANN="${TRAIN_ANN:-${CODA_LM_ROOT}/Train/vqa_anno/region_perception.jsonl}"
FEATURE_DIR="${FEATURE_DIR:-outputs/roi_features}"
CLASS2IDX="${CLASS2IDX:-configs/coda_class2idx.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/class_embeddings}"

echo "Config: ${CONFIG}"
env PYTHONPATH="src:${PYTHONPATH:-}" python -m seeing.cli.train_class_embeddings \
  --train-annotation "${TRAIN_ANN}" \
  --feature-dir "${FEATURE_DIR}" \
  --class2idx "${CLASS2IDX}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-64}"
