#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/coda_llava_7b.yaml}"

CODA_DATA_ROOT="${CODA_DATA_ROOT:-/path/to/CODA-data}"
CODA_LM_ROOT="${CODA_LM_ROOT:-${CODA_ROOT:-${CODA_DATA_ROOT}/CODA-LM}}"
TEST_ANN="${TEST_ANN:-${CODA_LM_ROOT}/Test/vqa_anno/region_perception.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${CODA_DATA_ROOT}}"
MODE="${MODE:-full}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval_llava15}"

CMD=(
  env PYTHONPATH="src:${PYTHONPATH:-}" python -m seeing.cli.eval_llava15_coda
  --annotation "${TEST_ANN}"
  --image-root "${IMAGE_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --mode "${MODE}"
  --llava-model-path "${LLAVA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
  --llava-model-name "${LLAVA_MODEL_NAME:-llava-v1.5-7b}"
  --conv-mode "${CONV_MODE:-llava_v1}"
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
)

if [[ "${MODE}" == "refine" || "${MODE}" == "full" ]]; then
  CMD+=(--class-embeddings "${CLASS_EMBEDDINGS:-outputs/class_embeddings/class_embeddings_best.pth}")
  CMD+=(--adapter "${ADAPTER:-outputs/adapter_llava15/adapter_last.pth}")
fi

if [[ "${MODE}" == "hints" || "${MODE}" == "full" ]]; then
  CMD+=(--detection-dir "${DETECTION_DIR:-saved_sample_objects}")
  CMD+=(--top-k "${TOP_K:-3}")
fi

if [[ "${NUM_SAMPLES:-0}" != "0" ]]; then
  CMD+=(--num-samples "${NUM_SAMPLES}")
fi

echo "Config: ${CONFIG}"
"${CMD[@]}"
