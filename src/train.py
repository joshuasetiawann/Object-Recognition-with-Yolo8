"""Train a YOLOv8 model on a custom dataset.

Example
-------
    python src/train.py --data configs/data.yaml --model yolov8n.pt \
        --epochs 100 --imgsz 640
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a custom dataset.")
    parser.add_argument(
        "--data",
        default="configs/data.yaml",
        help="Path to the dataset YAML file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device: 'cpu', '0', '0,1', ... (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading base model '{args.model}' ...")
    model = YOLO(args.model)

    print(f"Starting training for {args.epochs} epochs on '{args.data}' ...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print("Training complete. Best weights saved under 'runs/detect/train/weights/'.")


if __name__ == "__main__":
    main()
