#!/usr/bin/env python3
"""Convert CODA-LM region perception annotations to VQA JSONL.

This script reads CODA-LM original annotations and creates red-rectangle
images plus a compact JSONL file for the region perception task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm


QUESTION = (
    "Please describe the object inside the red rectangle in the image and "
    "explain why it affect ego car driving."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coda-root",
        type=Path,
        required=True,
        help="Root directory containing val/, test/, and CODA-LM/.",
    )
    parser.add_argument(
        "--annotation-dir",
        default="CODA-LM",
        help="CODA-LM annotation directory name under --coda-root.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Val", "Test", "Mini"],
        help="Splits to convert.",
    )
    parser.add_argument(
        "--box-width",
        type=int,
        default=2,
        help="Width of the red rectangle drawn on images.",
    )
    return parser.parse_args()


def image_source_from_annotation(json_path: Path) -> tuple[str, str]:
    """Return source image directory and image filename from an annotation name."""
    prefix, image_id = json_path.stem.split("_", maxsplit=1)
    return prefix, f"{image_id}.jpg"


def convert_split(coda_root: Path, annotation_dir: str, split: str, box_width: int) -> int:
    split_dir = coda_root / annotation_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    json_files = sorted(split_dir.glob("*.json"))
    output_dir = split_dir / "vqa_anno"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "region_perception.jsonl"

    samples = []
    question_id = 0

    for json_path in tqdm(json_files, desc=f"Converting {split}"):
        with json_path.open("r", encoding="utf-8") as f:
            annotation = json.load(f)

        image_dir, image_name = image_source_from_annotation(json_path)
        image_path = coda_root / image_dir / "images" / image_name
        boxed_dir = coda_root / image_dir / "images_w_boxes"
        boxed_dir.mkdir(parents=True, exist_ok=True)

        for region_id, region in annotation.get("region_perception", {}).items():
            x, y, w, h = region["box"]
            bbox_xyxy = [x, y, x + w, y + h]

            boxed_name = f"{Path(image_name).stem}_object_{region_id}.jpg"
            boxed_path = boxed_dir / boxed_name

            if not boxed_path.exists():
                image = Image.open(image_path).convert("RGB")
                draw = ImageDraw.Draw(image)
                draw.rectangle(bbox_xyxy, outline="red", width=box_width)
                image.save(boxed_path)

            samples.append(
                {
                    "question_id": question_id,
                    "image": str(Path(image_dir) / "images_w_boxes" / boxed_name),
                    "question": QUESTION,
                    "answer": region["description and explanation"],
                    "bbox": bbox_xyxy,
                    "label": region["category_name"],
                }
            )
            question_id += 1

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {output_path}")
    return len(samples)


def main() -> None:
    args = parse_args()
    total = 0
    for split in args.splits:
        total += convert_split(args.coda_root, args.annotation_dir, split, args.box_width)
    print(f"Done. Converted {total} region perception samples.")


if __name__ == "__main__":
    main()

