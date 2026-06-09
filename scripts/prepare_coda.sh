#!/usr/bin/env bash
set -euo pipefail

CODA_DATA_ROOT="${1:-${CODA_DATA_ROOT:-}}"

if [ -z "${CODA_DATA_ROOT}" ]; then
  echo "Usage: bash scripts/prepare_coda.sh /path/to/CODA-data"
  echo
  echo "Expected layout:"
  echo "  /path/to/CODA-data/val/images"
  echo "  /path/to/CODA-data/test/images"
  echo "  /path/to/CODA-data/CODA-LM/Train"
  exit 1
fi

env PYTHONPATH="src:${PYTHONPATH:-}" python tools/convert_coda_region_to_vqa.py \
  --coda-root "${CODA_DATA_ROOT}" \
  --splits Train Val Test Mini
