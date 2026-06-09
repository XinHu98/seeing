# Run CODA-LM With LLaVA-1.5

This guide is the shortest complete path for running the public CODA-LM + LLaVA-1.5-7B code.

## 0. Expected Data Layout

Arrange CODA images and CODA-LM annotations like this:

```text
/path/to/CODA-data/
├── val/
│   └── images/
├── test/
│   └── images/
└── CODA-LM/
    ├── Train/
    ├── Val/
    ├── Test/
    └── Mini/
```

Set paths once:

```bash
export CODA_DATA_ROOT=/path/to/CODA-data
export CODA_LM_ROOT=${CODA_DATA_ROOT}/CODA-LM
export IMAGE_ROOT=${CODA_DATA_ROOT}
```

`CODA_DATA_ROOT` contains `val/`, `test/`, and `CODA-LM/`.
`CODA_LM_ROOT` contains `Train/`, `Val/`, `Test/`, and `Mini/`.

## 1. Convert CODA-LM Region Perception Files

```bash
bash scripts/prepare_coda.sh "${CODA_DATA_ROOT}"
```

This creates VQA-style files such as:

```text
${CODA_LM_ROOT}/Train/vqa_anno/region_perception.jsonl
${CODA_LM_ROOT}/Test/vqa_anno/region_perception.jsonl
```

It also creates red-box images under:

```text
${CODA_DATA_ROOT}/val/images_w_boxes/
${CODA_DATA_ROOT}/test/images_w_boxes/
```

## 2. Run A Small Baseline Test

Run this first. It does not need class embeddings, adapter weights, or detection priors.

```bash
MODE=baseline NUM_SAMPLES=20 bash scripts/eval_coda.sh configs/coda_llava_7b.yaml
```

Outputs:

```text
outputs/eval_llava15/llava15_coda_baseline.jsonl
outputs/eval_llava15/llava15_coda_baseline_summary.json
```

## 3. Train Class Embeddings

Extract DINO ROI features from CODA-LM training regions:

```bash
bash scripts/extract_coda_region_features.sh
```

Train the class prototype bank:

```bash
bash scripts/train_class_embeddings_coda.sh configs/coda_llava_7b.yaml
```

Output:

```text
outputs/class_embeddings/class_embeddings_best.pth
```

## 4. Train The Visual Token Adapter

```bash
bash scripts/train_adapter_coda.sh configs/coda_llava_7b.yaml
```

Output:

```text
outputs/adapter_llava15/adapter_last.pth
```

## 5. Evaluate Refinement

Use the trained class embeddings and adapter:

```bash
MODE=refine bash scripts/eval_coda.sh configs/coda_llava_7b.yaml
```

Output:

```text
outputs/eval_llava15/llava15_coda_refine.jsonl
outputs/eval_llava15/llava15_coda_refine_summary.json
```

## 6. Evaluate Object Hints Or Full Mode

`hints` and `full` modes use external object-prior text files. Put them in a directory such as:

```text
saved_sample_objects/
├── 0001.txt
├── 0002.txt
└── ...
```

Each file should be tab separated:

```text
object	count	confidence
traffic_cone	1	0.93
bollard	2	0.81
```

Run hints only:

```bash
DETECTION_DIR=saved_sample_objects MODE=hints bash scripts/eval_coda.sh configs/coda_llava_7b.yaml
```

Run visual refinement plus object hints:

```bash
DETECTION_DIR=saved_sample_objects MODE=full bash scripts/eval_coda.sh configs/coda_llava_7b.yaml
```

## 7. Output Metrics

The public summary currently reports a lightweight label-mention proxy for smoke tests:

```text
outputs/eval_llava15/llava15_coda_<mode>_summary.json
```

The paper GPT-score evaluation protocol will be documented with the checkpoint release.

