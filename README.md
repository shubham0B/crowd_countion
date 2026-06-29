<div align="center">

# Vision AI — Crowd Detection

### Real-time people detection & counting powered by YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

**Vision AI — Crowd Detection** is an end-to-end computer vision system that automatically detects and counts people in images, videos, and live webcam feeds. Built on **YOLOv8**, augmented with **SAHI** tiled inference for dense scenes, **DeepSORT** for tracking persistent identities across video frames, and a **ResNet-SSD face recovery** stage to recapture occluded or distant people that the body detector misses.

The whole system ships as a clean **Streamlit web app** — no command line gymnastics, just upload media or point a camera and read the count off the screen.

> Built for crowd analytics, event monitoring, CCTV review, transit hubs, classrooms, and any scene where *“how many people are in this frame?”* is the question that matters.

---

## Key Features

- **Multi-scale person detection** — YOLOv8 (nano → x-large) tuned with confidence + IoU thresholds
- **Dense-crowd resilience** — SAHI (Slicing Aided Hyper Inference) tiles large images so far-away people aren’t lost to YOLO’s 640×640 down-scale
- **Face recovery pass** — a ResNet-SSD/YOLOv8-face fallback catches heads when bodies are occluded
- **Persistent IDs in video** — DeepSORT (Kalman + appearance embeddings) prevents double-counting across frames
- **Geometric false-positive filter** — `validate_person()` rejects boxes with bad aspect ratios or impossible sizes
- **Five model variants** at runtime — pick speed-vs-accuracy from the dropdown
- **GPU + CPU support** — automatic CUDA detection, graceful CPU fallback
- **Heatmap visualisation** of crowd density
- **Image · Video · Webcam** — three input modes from the same UI

---

## Tech Stack

| Layer | Technology |
|------|-----|
| Detection model | **YOLOv8** (Ultralytics, pretrained on COCO) |
| Tiling for dense scenes | **SAHI** 0.11 |
| Face recovery | **ResNet-10 SSD** (OpenCV DNN) + **YOLOv8-face** |
| Tracking | **DeepSORT-realtime** (Kalman filter) |
| Deep-learning framework | **PyTorch 2.2** (CUDA 11.8 / 12.1) |
| Computer vision | **OpenCV** 4.9 (headless) |
| Web UI | **Streamlit** 1.32 |
| Numerics | **NumPy**, **Pillow** |
| Language | **Python 3.10+** |

---

## Architecture

```
┌─────────────────────┐
│  Image / Video /    │
│   Webcam Input      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐    ┌──────────────────────┐
│ OpenCV frame loader │───►│ CLAHE pre-processing │
└──────────┬──────────┘    └──────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   YOLOv8 person detector (class 0)           │
│   + SAHI tiled inference for dense scenes    │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   Face-recovery pass (ResNet-SSD / YOLOv8-f) │
│   reclaims occluded / distant people         │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   validate_person() — geometric FP filter    │
│   Non-Max Suppression (IoU = 0.45)           │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   DeepSORT — persistent ID assignment        │
│   (video only)                               │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│   Streamlit UI · bounding boxes · count ·    │
│   density heatmap · output video             │
└──────────────────────────────────────────────┘
```

---

## Repository Structure

```
Vision-AI-Crowd-Detection/
├── crowd_counting/
│   ├── app.py                  # Streamlit web UI — main entry point
│   ├── config.py               # Thresholds, model paths, colors
│   ├── finetune.py             # Optional fine-tuning script
│   ├── accuracy_guide.py       # Accuracy tuning helpers
│   ├── verify_setup.py         # Quick environment check
│   ├── download_sample.py      # Pulls a sample crowd image
│   ├── requirements.txt        # Python dependencies
│   ├── crowd_data.yaml         # Dataset spec for fine-tuning
│   ├── deploy.prototxt         # ResNet-SSD face model graph
│   │
│   ├── src/                    # Core AI engine
│   │   ├── detector.py         # YOLO + SAHI + merging pipeline
│   │   ├── face_recovery.py    # Face fallback detector
│   │   ├── csrnet.py           # CSRNet density estimator (optional)
│   │   ├── heatmap.py          # Crowd density heatmap renderer
│   │   ├── live_processor.py   # Webcam / live-stream loop
│   │   ├── video_processor.py  # Video file processor with tracking
│   │   ├── validator.py        # Geometric false-positive filter
│   │   └── utils.py            # Drawing, FPS, logging helpers
│   │
│   ├── models/                 # YOLO weights (auto-downloaded)
│   ├── outputs/                # Generated annotated media
│   ├── images/                 # Drop test images here
│   ├── assets/                 # Icons / sample media
│   └── tests/
│       └── test_detector.py
│
├── LICENSE                     # MIT
├── README.md                   # You are here
└── .gitignore
```

---

## Installation

### Prerequisites
- Python **3.10+**
- (Optional) NVIDIA GPU with CUDA **11.8** or **12.1** for real-time speed
- ~2 GB free disk space (model weights auto-download on first run)

### 1. Clone

```bash
git clone https://github.com/Hohin28/Vision-AI-Crowd-Detection.git
cd Vision-AI-Crowd-Detection/crowd_counting
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install PyTorch **first** (GPU path)

```bash
# CUDA 11.8
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

CPU-only? Just skip this step — `pip install torch` from the next step grabs the CPU build.

### 4. Install everything else

```bash
pip install -r requirements.txt
```

### 5. Verify GPU detection (optional)

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 6. Launch

```bash
streamlit run app.py
```

The browser opens at **http://localhost:8501**. Larger YOLOv8 weights (`m`, `l`, `x`, `l-face`) download automatically from Ultralytics on first use — internet required only the first time.

---

## Usage

| Mode | What it does |
|------|--------------|
| **Image** | Upload `.jpg` / `.png` → bounding boxes + count overlay |
| **Video** | Upload `.mp4` / `.avi` → tracked, deduplicated, annotated video saved to `outputs/` |
| **Webcam** | Live feed with rolling person count |

Choose the YOLOv8 variant from the sidebar based on the scene:

| Variant | Best for | Speed |
|--------|----------|-------|
| **Nano** | 0–30 people, low-power devices | Fastest |
| **Small** | 0–60 people, balanced edge devices | Fast |
| **Medium** | 30–100 people | Balanced |
| **Large** *(default)* | 100+ people | Accurate |
| **X-Large** | Extreme crowds | Most accurate |

Tune the **confidence threshold** (`0.35` default) and **inference resolution** (640 → 1920) from the sidebar; higher resolution catches smaller people at the cost of speed.

---

## Model Choice — Why YOLOv8

| Model | Speed | Accuracy | Crowd handling | Verdict |
|-------|-------|----------|----------------|---------|
| **YOLOv8** | Real-time | High | Strong | **Selected** |
| Faster R-CNN | Slow | Very high | Good | Too slow for video |
| SSD | Medium | Medium | Weak in dense crowds | Loses small/far people |
| MobileNet-SSD | Very fast | Low | Weak | Accuracy too low |
| DETR | Slow | High | Good | Complex, heavy to deploy |

YOLOv8 wins on the real-time + accuracy + crowd-density triple constraint, and the Ultralytics API makes it easy to swap variants at runtime.

---

## Testing

Functional, edge-case, and accuracy testing live in `crowd_counting/tests/`.

```bash
cd crowd_counting
python -m pytest tests/
```

Coverage includes single/multi/dense scenes, low-light frames, occlusion, distant people, mixed environments, false-positive checks, and duplicate-detection checks via NMS.

---

## Roadmap

- [ ] Crowd density estimation via **CSRNet** (scaffolded in `src/csrnet.py`)
- [ ] Entry/exit counting with line-crossing logic
- [ ] Overcrowding alerts (threshold + email/webhook)
- [ ] CCTV dashboard with multi-camera support
- [ ] Dockerised deployment + REST API
- [ ] ONNX / TensorRT export for edge devices

---

## License

Released under the **[MIT License](LICENSE)** — free to use, modify, and distribute. Attribution appreciated.

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — detection backbone
- [SAHI](https://github.com/obss/sahi) — sliced inference
- [DeepSORT-realtime](https://github.com/levan92/deep_sort_realtime) — tracking
- [akanametov/yolov8-face](https://github.com/akanametov/yolov8-face) — face recovery weights
- OpenCV ResNet-10 SSD — face fallback

---

<div align="center">

**Made with Python, PyTorch, and a stubborn belief that counting people shouldn’t require a PhD.**

</div>
