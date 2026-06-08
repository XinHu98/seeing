"""Object-hint utilities for CODA-LM inference."""

from __future__ import annotations

from pathlib import Path


def parse_detection_file(path: str | Path) -> list[dict]:
    """Parse a simple detection-prior text file.

    Expected non-header rows are tab-separated as ``object, count, confidence``.
    """
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            rows.append({"object": parts[0].replace("_", " "), "count": int(parts[1]), "confidence": float(parts[2])})
        except ValueError:
            continue
    return rows


def load_detection_priors(detection_dir: str | Path) -> dict[str, list[dict]]:
    detection_dir = Path(detection_dir)
    if not detection_dir.exists():
        return {}
    priors = {}
    for path in detection_dir.glob("*.txt"):
        priors[path.stem] = parse_detection_file(path)
    return priors


def _image_keys(image_path: str) -> list[str]:
    stem = Path(image_path).stem
    keys = [stem]
    if "_object_" in stem:
        keys.append(stem.split("_object_")[0])
    if "_" in stem:
        keys.append(stem.split("_")[0])
    keys.extend(k.zfill(4) for k in list(keys) if k.isdigit())
    return list(dict.fromkeys(keys))


def top_objects_for_image(image_path: str, priors: dict[str, list[dict]], top_k: int = 3) -> list[str]:
    detections: list[dict] | None = None
    for key in _image_keys(image_path):
        if key in priors:
            detections = priors[key]
            break
    if not detections:
        return []
    detections = sorted(detections, key=lambda row: row["confidence"], reverse=True)
    return [row["object"] for row in detections[:top_k]]


def format_object_hints(objects: list[str]) -> str:
    if not objects:
        return ""
    if len(objects) == 1:
        return f"It might be a {objects[0]}."
    return "It might be one in '" + ", ".join(objects) + "'."
