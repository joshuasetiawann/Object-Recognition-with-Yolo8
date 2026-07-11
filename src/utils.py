"""Shared helper utilities for the YOLOv8 object recognition project."""

from __future__ import annotations

from pathlib import Path

# File extensions treated as images / videos when resolving a source.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def resolve_source_type(source: str) -> str:
    """Classify a detection source.

    Returns one of ``"webcam"``, ``"image"``, ``"video"`` or ``"directory"``.

    Args:
        source: A webcam index (e.g. ``"0"``), or a path to an image,
            a video, or a directory of files.
    """
    # A plain integer is interpreted as a webcam/device index.
    if source.isdigit():
        return "webcam"

    path = Path(source)
    if path.is_dir():
        return "directory"

    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"

    # Fall back to letting Ultralytics handle anything we don't recognise
    # (e.g. URLs or streams).
    return "image"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
