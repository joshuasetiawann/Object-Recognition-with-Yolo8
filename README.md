<div align="center">

# 🎯 Object Recognition with YOLOv8

**Real-time object detection and recognition powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📖 Overview

This project provides a clean, ready-to-use pipeline for **object recognition** using the
**YOLOv8** model family. It supports detection on **images**, **videos**, and a **live webcam**
stream, as well as **training** your own custom model on a labeled dataset.

The codebase is organized so you can go from zero to running inference in a couple of commands,
while keeping training, configuration, and outputs neatly separated.

## ✨ Features

- 🖼️ **Image detection** — run detection on a single image or a folder of images.
- 🎬 **Video detection** — process a video file frame by frame and save the annotated result.
- 📹 **Webcam detection** — real-time object recognition from your camera.
- 🧠 **Custom training** — train YOLOv8 on your own dataset via a simple config.
- ⚙️ **Configurable** — model size, confidence threshold, and device are all CLI options.
- 📦 **Lightweight** — minimal dependencies, built directly on the official Ultralytics package.

## 📁 Project Structure

```text
Object-Recognition-with-Yolo8/
├── src/                    # Source code
│   ├── detect.py           # Unified CLI entry point (image / video / webcam)
│   ├── train.py            # Train YOLOv8 on a custom dataset
│   └── utils.py            # Shared helper functions
├── configs/
│   └── data.yaml           # Dataset configuration for training
├── data/                   # Input data (images, videos) — not tracked by git
│   └── sample/
├── models/                 # Trained / downloaded model weights (.pt)
├── outputs/                # Detection results are written here
├── notebooks/              # Optional Jupyter notebooks for experiments
├── requirements.txt        # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### 1. Prerequisites

- Python **3.8+**
- (Optional) An NVIDIA GPU with CUDA for faster inference and training

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/joshuasetiawann/Object-Recognition-with-Yolo8.git
cd Object-Recognition-with-Yolo8

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> The first run will automatically download the pretrained YOLOv8 weights
> (e.g. `yolov8n.pt`) if they are not already present.

## 🕹️ Usage

All inference is handled by `src/detect.py`. Choose a `--source` and the script
figures out whether it is an image, a video, or a webcam stream.

### Detect on an image

```bash
python src/detect.py --source data/sample/image.jpg --model yolov8n.pt
```

### Detect on a video

```bash
python src/detect.py --source data/sample/video.mp4 --model yolov8n.pt
```

### Detect from your webcam

```bash
python src/detect.py --source 0 --model yolov8n.pt
```

### Useful options

| Argument        | Description                                          | Default      |
| --------------- | ---------------------------------------------------- | ------------ |
| `--source`      | Image/video path, folder, or webcam index (`0`)      | *(required)* |
| `--model`       | Path or name of the YOLOv8 weights                   | `yolov8n.pt` |
| `--conf`        | Confidence threshold (0–1)                           | `0.25`       |
| `--device`      | `cpu`, `0`, `0,1`, ... for GPU selection             | auto         |
| `--output`      | Directory to save annotated results                  | `outputs`    |
| `--show`        | Display results in a window while processing         | `False`      |

## 🧠 Training a Custom Model

1. Organize your dataset in the YOLO format and update `configs/data.yaml`:

   ```yaml
   path: ../data           # dataset root
   train: images/train     # train images (relative to path)
   val: images/val         # val images (relative to path)

   names:
     0: person
     1: car
     2: dog
   ```

2. Start training:

   ```bash
   python src/train.py --data configs/data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
   ```

3. The best weights are saved under `runs/detect/train/weights/best.pt`.
   Use them for inference:

   ```bash
   python src/detect.py --source data/sample/image.jpg --model runs/detect/train/weights/best.pt
   ```

## 🧩 Model Variants

YOLOv8 ships in several sizes — pick one based on your speed/accuracy trade-off:

| Model        | Size   | Speed       | Accuracy |
| ------------ | ------ | ----------- | -------- |
| `yolov8n.pt` | Nano   | ⚡ Fastest  | Lowest   |
| `yolov8s.pt` | Small  | Fast        | Low      |
| `yolov8m.pt` | Medium | Balanced    | Medium   |
| `yolov8l.pt` | Large  | Slow        | High     |
| `yolov8x.pt` | XLarge | 🐢 Slowest  | Highest  |

## 🛠️ Built With

- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — detection framework
- [PyTorch](https://pytorch.org/) — deep learning backend
- [OpenCV](https://opencv.org/) — image and video I/O

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request
for improvements, bug fixes, or new features.

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 model and tooling.

---

<div align="center">
Made with ❤️ for computer vision.
</div>
