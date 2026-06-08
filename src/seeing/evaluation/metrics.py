"""Lightweight public metrics for saved CODA-LM predictions."""

from __future__ import annotations


def label_mention_accuracy(predictions: list[dict]) -> dict:
    """Compute a simple label-mention proxy accuracy.

    The paper reports GPT-score evaluation. This proxy is included for quick
    smoke tests without requiring a private API key.
    """
    total = 0
    correct = 0
    for row in predictions:
        label = str(row.get("label", "")).lower()
        pred = str(row.get("prediction", "")).lower()
        if not label:
            continue
        total += 1
        correct += int(label in pred)
    return {"total": total, "correct": correct, "accuracy": correct / total if total else 0.0}
