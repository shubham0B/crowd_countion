# CROWD COUNTING SYSTEM — INSTALLATION GUIDE
# GPU Build (NVIDIA CUDA 11.8 / 12.1)

## STEP 1 — Create a virtual environment
```
python -m venv venv
```

## STEP 2 — Activate the environment
```
# Windows:
venv\Scripts\activate

# Linux / Mac:
source venv/bin/activate
```

## STEP 3 — Install PyTorch with CUDA FIRST (do this before anything else)
```
# For CUDA 11.8:
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

## STEP 4 — Verify GPU is detected
```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected output: True  +  your GPU name (e.g. NVIDIA GeForce RTX 3060)

## STEP 5 — Install all other dependencies
```
pip install -r requirements.txt
```

## STEP 6 — Run the app
```
streamlit run app.py
```
The browser opens at http://localhost:8501

## FOLDER STRUCTURE EXPLANATION
```
crowd_counting/
├── app.py                  # Streamlit web UI — the main entry point
├── config.py               # All settings (thresholds, colors, paths)
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/                    # Core AI engine (all backend logic)
│   ├── __init__.py         # Makes src/ a Python package
│   ├── detector.py         # Main detection pipeline (YOLO + SAHI + merging)
│   ├── face_recovery.py    # ResNet SSD face detection (OpenCV DNN)
│   ├── validator.py        # validate_person() — geometric false positive filter
│   └── utils.py            # Drawing boxes, FPS, logging helpers
│
├── models/                 # YOLO .pt weight files live here after first download
│   └── (auto-downloaded)
│
├── outputs/                # Processed images/videos saved here
│
├── images/                 # Put test images here
│
├── assets/                 # Static files (icons, sample images)
│
└── tests/                  # Unit tests for each module
    └── test_detector.py
```
