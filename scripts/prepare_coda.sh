#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/prepare_coda.sh /path/to/CODA-root"
  exit 1
fi

CODA_ROOT="$1"

python tools/convert_coda_region_to_vqa.py \
  --coda-root "$CODA_ROOT" \
  --splits Train Val Test Mini

