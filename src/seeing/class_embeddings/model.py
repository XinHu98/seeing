"""Class embedding learning from precomputed region visual features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from seeing.data.coda import load_coda_region_jsonl, normalize_label


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.net(x))


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, dim: int, ema_momentum: float = 0.95) -> None:
        super().__init__()
        self.ema_momentum = ema_momentum
        self.w = nn.Parameter(l2_normalize(torch.randn(num_classes, dim)))

    @torch.no_grad()
    def init_from_features(self, class_to_features: dict[int, torch.Tensor]) -> None:
        for class_id, feats in class_to_features.items():
            if feats.numel() > 0:
                self.w.data[class_id] = l2_normalize(feats.mean(0, keepdim=True)).squeeze(0)

    @torch.no_grad()
    def ema_update(self, labels: torch.Tensor, features: torch.Tensor) -> None:
        for class_id in labels.unique():
            mask = labels == class_id
            if not mask.any():
                continue
            mu = l2_normalize(features[mask].mean(0, keepdim=True)).squeeze(0)
            self.w.data[class_id] = l2_normalize(
                self.ema_momentum * self.w.data[class_id] + (1 - self.ema_momentum) * mu
            )


class ArcFaceLoss(nn.Module):
    def __init__(self, scale: float = 30.0, margin: float = 0.25) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = l2_normalize(features)
        prototypes = l2_normalize(prototypes)
        logits = features @ prototypes.t()
        row = torch.arange(features.shape[0], device=features.device)
        target = logits[row, labels]
        target_m = torch.cos(torch.acos(target.clamp(-1 + 1e-7, 1 - 1e-7)) + self.margin)
        logits = logits.clone()
        logits[row, labels] = target_m
        return F.cross_entropy(self.scale * logits, labels)


class ClassEmbeddingModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_dim: int = 1024,
        embedding_dim: int = 1024,
        ema_momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.projector = ProjectionHead(feature_dim, embedding_dim)
        self.bank = PrototypeBank(num_classes, embedding_dim, ema_momentum)
        self.loss_fn = ArcFaceLoss()

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        projected = self.projector(features)
        loss = self.loss_fn(projected, self.bank.w, labels)
        with torch.no_grad():
            self.bank.ema_update(labels, projected)
            logits = l2_normalize(projected) @ l2_normalize(self.bank.w).t()
            acc = (logits.argmax(1) == labels).float().mean()
        return loss, {"loss": float(loss.detach().cpu()), "acc": float(acc.detach().cpu())}


class FeatureDataset(Dataset):
    """Precomputed ROI feature dataset keyed by CODA-LM question_id."""

    def __init__(self, annotation: str | Path, feature_dir: str | Path, class2idx: dict[str, int], prefix: str = "train"):
        self.feature_dir = Path(feature_dir)
        self.class2idx = {normalize_label(k): int(v) for k, v in class2idx.items()}
        self.samples = []
        for sample in load_coda_region_jsonl(annotation):
            if sample.label not in self.class2idx:
                continue
            feature_path = self.feature_dir / f"{prefix}_{sample.question_id}.npy"
            if feature_path.exists():
                self.samples.append((feature_path, self.class2idx[sample.label], sample.question_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        feature_path, label, qid = self.samples[idx]
        feature = np.load(feature_path).squeeze()
        return torch.from_numpy(feature).float(), torch.tensor(label, dtype=torch.long), qid


def load_class2idx(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {normalize_label(k): int(v) for k, v in mapping.items()}


def initial_prototypes(dataset: FeatureDataset, num_classes: int, feature_dim: int, device: torch.device):
    buckets: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_classes)}
    for feature_path, label, _ in dataset.samples:
        buckets[label].append(torch.from_numpy(np.load(feature_path).squeeze()).float())
    protos = {}
    for label, features in buckets.items():
        if features:
            protos[label] = torch.stack(features).mean(0, keepdim=True).to(device)
        else:
            protos[label] = torch.randn(1, feature_dim, device=device)
    return protos


def train_one_epoch(model: ClassEmbeddingModel, dataloader, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    for features, labels, _ in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, logs = model(features, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += logs["loss"]
        total_acc += logs["acc"]
        batches += 1
    return {"loss": total_loss / max(batches, 1), "acc": total_acc / max(batches, 1)}
