# Scripts

This directory will contain the CODA-LM reproduction commands.

Planned entry points:

- `prepare_coda.sh`: convert CODA-LM region perception annotations to VQA JSONL.
- `train_class_embeddings_coda.sh`: learn multi-modal class embeddings.
- `train_adapter_coda.sh`: train the visual token refinement adapter while keeping the VLM frozen.
- `eval_coda.sh`: evaluate CODA-LM with visual refinement and object hints.

The first public code release will fill these scripts with executable commands.

