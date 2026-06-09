# Data

This repository focuses on the CODA-LM region perception task.

## CODA-LM Layout

After downloading CODA-LM and CODA images, arrange files as:

```text
CODA-root/
├── val/
│   └── images/
├── test/
│   └── images/
└── CODA-LM/
    ├── Train/
    │   └── val_*.json
    ├── Val/
    │   └── test_*.json
    ├── Test/
    │   └── test_*.json
    └── Mini/
        └── test_*.json
```

For the main experiments, we convert `region_perception` annotations into VQA-style JSONL files:

```text
CODA-root/
└── CODA-LM/
    ├── Train/vqa_anno/region_perception.jsonl
    └── Test/vqa_anno/region_perception.jsonl
```

The scripts use the following path convention:

```bash
export CODA_DATA_ROOT=/path/to/CODA-data
export CODA_LM_ROOT=${CODA_DATA_ROOT}/CODA-LM
export IMAGE_ROOT=${CODA_DATA_ROOT}
```

`CODA_DATA_ROOT` contains the image folders (`val/`, `test/`) and the `CODA-LM/` annotation folder. `CODA_LM_ROOT` contains the CODA-LM splits (`Train/`, `Val/`, `Test/`, `Mini/`).

Each converted sample has:

```json
{
  "question_id": 0,
  "image": "val/images_w_boxes/0001_object_1.jpg",
  "question": "Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
  "answer": "...",
  "bbox": [x1, y1, x2, y2],
  "label": "bollard"
}
```

## Main Counts

| Split | Images | Region QA pairs |
| --- | ---: | ---: |
| Train | 4,884 | 10,727 |
| Test | 500 | 1,123 |

## What Not To Commit

Do not commit raw datasets, generated images, DINO features, checkpoints, prediction files, or GPT evaluation outputs. Use external storage such as Hugging Face, Google Drive, or GitHub Releases for large artifacts.

## Optional GeoBench

GeoBench is a secondary cross-domain experiment. It is useful for showing transfer beyond ground-level driving scenes, but CODA-LM is the primary benchmark and reproduction path.
