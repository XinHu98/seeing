# Scripts

This directory contains the CODA-LM + LLaVA-1.5-7B reproduction commands.

Entry points:

- `prepare_coda.sh`: convert CODA-LM region perception annotations to VQA JSONL.
- `extract_coda_region_features.sh`: extract DINO ROI features from CODA-LM boxes.
- `train_class_embeddings_coda.sh`: learn multi-modal class embeddings.
- `train_adapter_coda.sh`: train the visual token refinement adapter while keeping the VLM frozen.
- `eval_coda.sh`: evaluate CODA-LM with `baseline`, `hints`, `refine`, or `full` mode.

All scripts use environment variables for paths. The default convention is:

```bash
export CODA_DATA_ROOT=/path/to/CODA-data
export CODA_LM_ROOT=${CODA_DATA_ROOT}/CODA-LM
export IMAGE_ROOT=${CODA_DATA_ROOT}

MODE=baseline NUM_SAMPLES=20 bash scripts/eval_coda.sh
```

`CODA_ROOT=/path/to/CODA-data/CODA-LM` is still accepted for backward compatibility, but `CODA_DATA_ROOT` plus `CODA_LM_ROOT` is clearer for new runs.
