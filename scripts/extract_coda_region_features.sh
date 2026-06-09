#!/usr/bin/env bash
set -euo pipefail

CODA_DATA_ROOT="${CODA_DATA_ROOT:-/path/to/CODA-data}"
CODA_LM_ROOT="${CODA_LM_ROOT:-${CODA_ROOT:-${CODA_DATA_ROOT}/CODA-LM}}"
TRAIN_ANN="${TRAIN_ANN:-${CODA_LM_ROOT}/Train/vqa_anno/region_perception.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${CODA_DATA_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/roi_features}"

env PYTHONPATH="src:${PYTHONPATH:-}" python -m seeing.cli.extract_region_features \
  --annotation "${TRAIN_ANN}" \
  --image-root "${IMAGE_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX:-train}" \
  --bbox-format "${BBOX_FORMAT:-xyxy}"
