"""
============================================================
verify_setup.py — Run this BEFORE starting app.py
============================================================
This script checks:
  1. Python version
  2. CUDA / GPU availability
  3. Every required library is installed
  4. YOLO can load and run a dummy inference
  5. Face model downloads correctly
  6. Output folder structure exists

HOW TO RUN:
    python verify_setup.py

Expected final output:
    ✅ ALL CHECKS PASSED — Run: streamlit run app.py
============================================================
"""

import sys
import os

PASSED = []
FAILED = []

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅  {label}" + (f"  →  {result}" if result else ""))
        PASSED.append(label)
    except Exception as e:
        print(f"  ❌  {label}  →  {e}")
        FAILED.append(label)

print("\n" + "="*60)
print("  CROWD COUNTER — SETUP VERIFICATION")
print("="*60)

# ── 1. Python version ────────────────────────────────────
print("\n[1] Python Environment")
check("Python >= 3.9",
      lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
      if sys.version_info >= (3, 9)
      else (_ for _ in ()).throw(RuntimeError(f"Need 3.9+, got {sys.version}")))

# ── 2. GPU / CUDA ────────────────────────────────────────
print("\n[2] GPU / CUDA")
def check_cuda():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        return f"GPU: {name}  ({mem} GB VRAM)"
    else:
        return "No GPU detected — will use CPU (slower)"

check("PyTorch installed", lambda: __import__('torch').__version__)
check("CUDA status", check_cuda)

# ── 3. Core Libraries ────────────────────────────────────
print("\n[3] Core Libraries")
REQUIRED = [
    ("streamlit",             "streamlit"),
    ("opencv-python",         "cv2"),
    ("numpy",                 "numpy"),
    ("Pillow",                "PIL"),
    ("ultralytics (YOLOv8)",  "ultralytics"),
    ("sahi",                  "sahi"),
    ("deep-sort-realtime",    "deep_sort_realtime"),
    ("requests",              "requests"),
]

for display_name, module_name in REQUIRED:
    def _check(m=module_name):
        mod = __import__(m)
        return getattr(mod, '__version__', 'installed')
    check(display_name, _check)

# ── 4. Folder Structure ──────────────────────────────────
print("\n[4] Project Folder Structure")
BASE = os.path.dirname(os.path.abspath(__file__))
REQUIRED_DIRS  = ["models", "outputs", "images", "src", "tests"]
REQUIRED_FILES = ["app.py", "config.py", "requirements.txt",
                  "src/__init__.py", "src/detector.py",
                  "src/face_recovery.py", "src/utils.py",
                  "src/validator.py", "src/heatmap.py",
                  "src/video_processor.py"]

for d in REQUIRED_DIRS:
    path = os.path.join(BASE, d)
    check(f"Directory: {d}/",
          lambda p=path: os.makedirs(p, exist_ok=True) or "exists")

for f in REQUIRED_FILES:
    path = os.path.join(BASE, f)
    check(f"File: {f}",
          lambda p=path: "found" if os.path.exists(p)
          else (_ for _ in ()).throw(FileNotFoundError(f"Missing: {p}")))

# ── 5. YOLO Dummy Inference ──────────────────────────────
print("\n[5] YOLOv8 Inference Test (Nano model)")
def test_yolo():
    import numpy as np
    from ultralytics import YOLO
    import torch

    # Patch for PyTorch 2.6
    try:
        import ultralytics.nn.tasks
        torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    except Exception:
        pass

    model = YOLO("yolov8n.pt")   # Downloads ~6MB if not present
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    results = model(dummy, verbose=False)
    return f"Inference OK — model device: {next(model.model.parameters()).device}"

check("YOLOv8 loads and runs inference", test_yolo)

# ── 6. Face Model Download ───────────────────────────────
print("\n[6] Face Recovery Model")
def test_face_model():
    sys.path.insert(0, BASE)
    from src.face_recovery import download_face_model, load_face_net
    download_face_model()
    net = load_face_net()
    return "ResNet face model downloaded and loaded"

check("ResNet SSD face detector", test_face_model)

# ── 7. Config Import ─────────────────────────────────────
print("\n[7] Configuration")
def test_config():
    sys.path.insert(0, BASE)
    from config import DEVICE, YOLO_MODELS, OUTPUTS_DIR
    return f"Device={DEVICE}, Models={len(YOLO_MODELS)} options"

check("config.py imports without error", test_config)

# ── FINAL REPORT ─────────────────────────────────────────
print("\n" + "="*60)
print(f"  PASSED: {len(PASSED)}   FAILED: {len(FAILED)}")
print("="*60)

if FAILED:
    print("\n❌ FIX THESE BEFORE RUNNING THE APP:\n")
    for f in FAILED:
        print(f"  → {f}")
    print("\nMost common fixes:")
    print("  pip install -r requirements.txt")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)
else:
    print("\n✅  ALL CHECKS PASSED")
    print("\n  Run the app with:")
    print("  streamlit run app.py\n")
