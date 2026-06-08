"""Train CODA-LM class embeddings from precomputed ROI features."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seeing.class_embeddings.model import (
    ClassEmbeddingModel,
    FeatureDataset,
    initial_prototypes,
    load_class2idx,
    train_one_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CODA-LM class embeddings")
    parser.add_argument("--train-annotation", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--class2idx", required=True)
    parser.add_argument("--output-dir", default="outputs/class_embeddings")
    parser.add_argument("--feature-prefix", default="train")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--ema-momentum", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class2idx = load_class2idx(args.class2idx)
    dataset = FeatureDataset(args.train_annotation, args.feature_dir, class2idx, prefix=args.feature_prefix)
    if len(dataset) == 0:
        raise RuntimeError("No feature files matched the annotation. Check --feature-dir and --feature-prefix.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = ClassEmbeddingModel(
        num_classes=len(class2idx),
        feature_dim=args.feature_dim,
        embedding_dim=args.embedding_dim,
        ema_momentum=args.ema_momentum,
    ).to(device)
    model.bank.init_from_features(initial_prototypes(dataset, len(class2idx), args.feature_dim, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    for epoch in tqdm(range(1, args.epochs + 1), desc="epochs"):
        logs = train_one_epoch(model, dataloader, optimizer, device)
        print(f"epoch={epoch} loss={logs['loss']:.4f} acc={logs['acc']:.4f}")
        if logs["acc"] > best_acc:
            best_acc = logs["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "bank.w": model.bank.w.detach().cpu(),
                    "projector_state_dict": model.projector.state_dict(),
                    "class2idx": class2idx,
                    "feature_dim": args.feature_dim,
                    "embedding_dim": args.embedding_dim,
                    "num_classes": len(class2idx),
                    "accuracy": best_acc,
                    "source": "coda_lm_learned_class_embeddings",
                },
                output_dir / "class_embeddings_best.pth",
            )

    print(f"saved best checkpoint to {output_dir / 'class_embeddings_best.pth'}")


if __name__ == "__main__":
    main()
