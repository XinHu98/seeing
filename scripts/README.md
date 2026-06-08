# Scripts

This directory contains the CODA-LM + LLaVA-1.5-7B reproduction commands.

Entry points:

- `prepare_coda.sh`: convert CODA-LM region perception annotations to VQA JSONL.
- `extract_coda_region_features.sh`: extract DINO ROI features from CODA-LM boxes.
- `train_class_embeddings_coda.sh`: learn multi-modal class embeddings.
- `train_adapter_coda.sh`: train the visual token refinement adapter while keeping the VLM frozen.
- `eval_coda.sh`: evaluate CODA-LM with `baseline`, `hints`, `refine`, or `full` mode.

All scripts use environment variables for paths, for example:

```bash
CODA_ROOT=/path/to/CODA-data/CODA-LM MODE=baseline NUM_SAMPLES=20 bash scripts/eval_coda.sh
```
