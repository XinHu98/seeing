"""Train the LLaVA-1.5 visual token adapter on CODA-LM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from seeing.data.coda import load_coda_region_jsonl, resolve_image_path
from seeing.models.adapter import AdapterLoss, ClassGuidedVisualAdapter, load_class_embeddings
from seeing.models.llava15 import (
    build_inputs_embeds_with_visual_tokens,
    extract_visual_tokens,
    image_token_positions,
    load_llava15,
    make_prompt,
    make_supervised_prompt,
    process_image,
    tokenize_prompt,
)


class SupervisedCodaDataset(Dataset):
    def __init__(self, annotation: str, image_root: str, bundle, conv_mode: str, device: torch.device):
        self.samples = load_coda_region_jsonl(annotation)
        self.image_root = image_root
        self.bundle = bundle
        self.conv_mode = conv_mode
        self.device = device

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        full_prompt = make_supervised_prompt(self.bundle, sample.question, sample.answer, self.conv_mode)
        prompt_only = make_prompt(self.bundle, sample.question, self.conv_mode)
        input_ids = tokenize_prompt(self.bundle, full_prompt)
        labels = input_ids.clone()
        prefix_len = tokenize_prompt(self.bundle, prompt_only).shape[0]
        labels[: min(prefix_len, labels.shape[0])] = -100
        image_tensor = process_image(self.bundle, resolve_image_path(self.image_root, sample.image), self.device)[0]
        return {"input_ids": input_ids, "labels": labels, "image_tensor": image_tensor, "qid": sample.question_id}


def collate(batch, pad_token_id: int):
    max_len = max(row["input_ids"].shape[0] for row in batch)
    input_ids = []
    labels = []
    for row in batch:
        pad = max_len - row["input_ids"].shape[0]
        input_ids.append(torch.cat([row["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([row["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "image_tensors": torch.stack([row["image_tensor"] for row in batch]),
        "qids": [row["qid"] for row in batch],
    }


def adjusted_labels(labels: torch.Tensor, positions: list[int | None], num_visual_tokens: int, max_len: int) -> torch.Tensor:
    rows = []
    for row, pos in zip(labels, positions):
        if pos is None:
            adjusted = row
        else:
            adjusted = torch.cat(
                [
                    row[:pos],
                    torch.full((num_visual_tokens,), -100, dtype=torch.long, device=row.device),
                    row[pos + 1 :],
                ]
            )
        if adjusted.shape[0] < max_len:
            adjusted = torch.cat([adjusted, torch.full((max_len - adjusted.shape[0],), -100, dtype=torch.long, device=row.device)])
        rows.append(adjusted[:max_len])
    return torch.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CODA-LM LLaVA-1.5 visual adapter")
    parser.add_argument("--train-annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--class-embeddings", required=True)
    parser.add_argument("--output-dir", default="outputs/adapter_llava15")
    parser.add_argument("--llava-model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--llava-model-name", default="llava-v1.5-7b")
    parser.add_argument("--conv-mode", default="llava_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--autoregressive-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bundle = load_llava15(args.llava_model_path, args.llava_model_name, device=device)
    for param in bundle.model.parameters():
        param.requires_grad_(False)

    class_embeddings = load_class_embeddings(args.class_embeddings, device=device).float()
    adapter = ClassGuidedVisualAdapter(num_classes=class_embeddings.shape[0], dropout=0.1).to(device)
    loss_fn = AdapterLoss(args.reconstruction_weight, args.autoregressive_weight)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-2)

    dataset = SupervisedCodaDataset(args.train_annotation, args.image_root, bundle, args.conv_mode, device)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate(rows, bundle.tokenizer.pad_token_id),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        totals = {"total": 0.0, "reconstruction": 0.0, "autoregressive": 0.0}
        batches = 0
        for batch in tqdm(dataloader, desc=f"epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            image_tensors = batch["image_tensors"].to(device=device, dtype=torch.float16)

            with torch.no_grad():
                original_tokens = extract_visual_tokens(bundle.model, image_tensors)
            refined_tokens, _, _ = adapter(original_tokens.float(), class_embeddings)
            positions = image_token_positions(input_ids, bundle.constants.IMAGE_TOKEN_INDEX)
            inputs_embeds, attention_mask = build_inputs_embeds_with_visual_tokens(
                bundle.model, input_ids, refined_tokens.to(dtype=original_tokens.dtype), positions
            )
            labels_final = adjusted_labels(labels, positions, original_tokens.shape[1], inputs_embeds.shape[1])

            outputs = bundle.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_final,
                return_dict=True,
            )
            loss, logs = loss_fn(refined_tokens, original_tokens, outputs.loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 0.5)
            optimizer.step()

            for key in totals:
                totals[key] += logs[key]
            batches += 1

        logs = {key: value / max(batches, 1) for key, value in totals.items()}
        print(f"epoch={epoch} total={logs['total']:.4f} recon={logs['reconstruction']:.4f} lm={logs['autoregressive']:.4f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": adapter.state_dict(),
                "class_embedding_shape": tuple(class_embeddings.shape),
                "config": vars(args),
            },
            output_dir / f"adapter_epoch_{epoch}.pth",
        )
    torch.save({"model_state_dict": adapter.state_dict(), "config": vars(args)}, output_dir / "adapter_last.pth")


if __name__ == "__main__":
    main()
