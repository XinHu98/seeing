"""Extract DINOv3 ROI features for CODA-LM class embedding training."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel
import torchvision.transforms.functional as TF

from seeing.data.coda import load_coda_region_jsonl, resolve_image_path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize_transform(image: Image.Image, patch_size: int):
    width, height = image.size
    out_h = int(math.ceil(height / patch_size)) * patch_size
    out_w = int(math.ceil(width / patch_size)) * patch_size
    return TF.to_tensor(TF.resize(image, (out_h, out_w)))


@torch.no_grad()
def extract_feature(image: Image.Image, model, device: torch.device, patch_size: int) -> np.ndarray:
    tensor = resize_transform(image, patch_size)
    tensor = TF.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD).unsqueeze(0).to(device)
    output = model(tensor)
    cls = output.last_hidden_state[:, 0, :]
    return F.normalize(cls, dim=-1).squeeze(0).detach().cpu().numpy()


def crop_region(image: Image.Image, bbox, bbox_format: str) -> Image.Image:
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    if bbox_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    width, height = image.size
    x1 = max(0, min(width, int(round(x1))))
    x2 = max(0, min(width, int(round(x2))))
    y1 = max(0, min(height, int(round(y1))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return image
    return image.crop((x1, y1, x2, y2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CODA-LM DINO ROI features")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="outputs/roi_features")
    parser.add_argument("--output-prefix", default="train")
    parser.add_argument("--dino-model", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.dino_model).to(device).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_coda_region_jsonl(args.annotation)
    failed = 0
    for sample in tqdm(samples, desc="extracting"):
        try:
            image_path = resolve_image_path(args.image_root, sample.image)
            image = Image.open(image_path).convert("RGB")
            roi = crop_region(image, sample.bbox, args.bbox_format)
            feature = extract_feature(roi, model, device, args.patch_size)
            np.save(output_dir / f"{args.output_prefix}_{sample.question_id}.npy", feature)
        except Exception as exc:
            failed += 1
            print(f"failed question_id={sample.question_id} image={sample.image}: {exc}")

    print(f"done. samples={len(samples)} failed={failed} output_dir={output_dir}")


if __name__ == "__main__":
    main()
