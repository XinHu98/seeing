"""Run CODA-LM inference with LLaVA-1.5 baseline/refinement/object-hint modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from seeing.data.coda import load_coda_region_jsonl, resolve_image_path
from seeing.evaluation.metrics import label_mention_accuracy
from seeing.inference.priors import format_object_hints, load_detection_priors, top_objects_for_image
from seeing.models.adapter import ClassGuidedVisualAdapter, load_class_embeddings
from seeing.models.llava15 import generate_baseline, generate_with_refined_tokens, load_llava15


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CODA-LM with LLaVA-1.5")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="outputs/eval_llava15")
    parser.add_argument("--mode", choices=["baseline", "hints", "refine", "full"], default="full")
    parser.add_argument("--llava-model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--llava-model-name", default="llava-v1.5-7b")
    parser.add_argument("--class-embeddings")
    parser.add_argument("--adapter")
    parser.add_argument("--detection-dir")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_refinement = args.mode in {"refine", "full"}
    use_hints = args.mode in {"hints", "full"}

    bundle = load_llava15(args.llava_model_path, args.llava_model_name, device=device)
    adapter = None
    class_embeddings = None
    if use_refinement:
        if not args.class_embeddings or not args.adapter:
            raise ValueError("--class-embeddings and --adapter are required for refine/full modes.")
        class_embeddings = load_class_embeddings(args.class_embeddings, device=device).float()
        adapter = ClassGuidedVisualAdapter(num_classes=class_embeddings.shape[0], dropout=0.0).to(device)
        checkpoint = torch.load(args.adapter, map_location=device)
        adapter.load_state_dict(checkpoint["model_state_dict"])
        adapter.eval()

    priors = load_detection_priors(args.detection_dir) if use_hints and args.detection_dir else {}
    samples = load_coda_region_jsonl(args.annotation)
    if args.num_samples > 0:
        samples = samples[: args.num_samples]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    for sample in tqdm(samples, desc=args.mode):
        question = sample.question
        hints = []
        if use_hints:
            hints = top_objects_for_image(sample.image, priors, args.top_k)
            hint_text = format_object_hints(hints)
            if hint_text:
                question = question + " " + hint_text
        image_path = resolve_image_path(args.image_root, sample.image)
        try:
            if use_refinement:
                prediction = generate_with_refined_tokens(
                    bundle,
                    question,
                    image_path,
                    adapter,
                    class_embeddings,
                    device,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            else:
                prediction = generate_baseline(
                    bundle,
                    question,
                    image_path,
                    device,
                    conv_mode=args.conv_mode,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
        except Exception as exc:
            prediction = f"ERROR: {exc}"

        predictions.append(
            {
                "question_id": sample.question_id,
                "image": sample.image,
                "question": sample.question,
                "question_with_hints": question,
                "object_hints": hints,
                "ground_truth": sample.answer,
                "prediction": prediction,
                "label": sample.label,
                "mode": args.mode,
            }
        )

    pred_path = output_dir / f"llava15_coda_{args.mode}.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {"mode": args.mode, "num_samples": len(predictions), "label_mention": label_mention_accuracy(predictions)}
    with (output_dir / f"llava15_coda_{args.mode}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
