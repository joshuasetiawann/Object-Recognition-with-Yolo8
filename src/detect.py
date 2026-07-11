"""Run YOLOv8 object detection on an image, video, directory, or webcam.

Examples
--------
Image:
    python src/detect.py --source data/sample/image.jpg

Video:
    python src/detect.py --source data/sample/video.mp4 --show

Webcam:
    python src/detect.py --source 0 --show
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

from utils import ensure_dir, resolve_source_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Object recognition with YOLOv8 (image / video / webcam).",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image/video path, directory, or webcam index (e.g. 0).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path or name of the YOLOv8 weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold between 0 and 1 (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cpu', '0', '0,1', ... (default: auto).",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Directory to save annotated results (default: outputs).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window while processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output)

    source_type = resolve_source_type(args.source)
    # Ultralytics expects an int for webcam streams.
    source = int(args.source) if source_type == "webcam" else args.source

    print(f"Loading model '{args.model}' ...")
    model = YOLO(args.model)

    print(f"Running detection on {source_type}: {args.source}")
    model.predict(
        source=source,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.output,
        name="detect",
        exist_ok=True,
        show=args.show,
        stream=False,
    )

    print(f"Done. Results saved to '{args.output}/detect'.")


if __name__ == "__main__":
    main()
