# ============================================================
# src/csrnet.py — CSRNet Density Estimation for Large Crowds
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import sys
import subprocess
from typing import Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_logger

logger = get_logger(__name__)

CSRNET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "csrnet_partA.pth"
)


# ─────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, dilation=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, dilation=2, padding=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)


# ─────────────────────────────────────────
# DOWNLOAD — uses gdown (most reliable for GDrive)
# ─────────────────────────────────────────

def _install_gdown():
    """Install gdown if not present."""
    try:
        import gdown
        return True
    except ImportError:
        logger.info("Installing gdown...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown", "-q"])
        return True


def download_csrnet_weights() -> None:
    os.makedirs(os.path.dirname(CSRNET_PATH), exist_ok=True)

    if os.path.exists(CSRNET_PATH):
        size_mb = os.path.getsize(CSRNET_PATH) / (1024 * 1024)
        if size_mb > 30:
            logger.info(f"CSRNet weights found ({size_mb:.0f} MB)")
            return
        os.remove(CSRNET_PATH)

    logger.info("Downloading CSRNet weights via gdown...")
    _install_gdown()
    import gdown

    # Multiple known working Google Drive file IDs for CSRNet Part-A weights
    file_ids = [
        "1nnIHPaV9RFqKovb2G7sTX5j2O4R5R5rM",   # CSRNet ShanghaiTech Part-A
        "190DVyFm0gHqFvFB8ZzGNxiRE3Y3VWvEK",   # backup ID
    ]

    for fid in file_ids:
        try:
            url = f"https://drive.google.com/uc?id={fid}"
            logger.info(f"Trying file ID: {fid}")
            gdown.download(url, CSRNET_PATH, quiet=False)
            size_mb = os.path.getsize(CSRNET_PATH) / (1024 * 1024)
            if size_mb > 30:
                logger.info(f"Downloaded ({size_mb:.0f} MB)")
                return
            os.remove(CSRNET_PATH)
        except Exception as e:
            logger.warning(f"Failed: {e}")

    # Final fallback — HuggingFace
    try:
        import requests
        hf_url = "https://huggingface.co/ldrama/csrnet-shanghaitech/resolve/main/csrnet_partA.pth"
        logger.info("Trying HuggingFace fallback...")
        resp = requests.get(hf_url, stream=True, timeout=180)
        resp.raise_for_status()
        with open(CSRNET_PATH, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        size_mb = os.path.getsize(CSRNET_PATH) / (1024 * 1024)
        if size_mb > 30:
            logger.info(f"HuggingFace download OK ({size_mb:.0f} MB)")
            return
    except Exception as e:
        logger.warning(f"HuggingFace failed: {e}")

    raise RuntimeError(
        "All automatic downloads failed.\n\n"
        "MANUAL DOWNLOAD STEPS:\n"
        "1. Go to: https://www.kaggle.com/datasets/liuliuliu/csrnet-shanghaitech-part-a\n"
        "2. Download 'partA_best.pth'\n"
        "3. Rename it to: csrnet_partA.pth\n"
        f"4. Put it in: {os.path.dirname(CSRNET_PATH)}\\"
    )


# ─────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────

def load_csrnet() -> CSRNet:
    download_csrnet_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CSRNet()
    ckpt   = torch.load(CSRNET_PATH, map_location=device, weights_only=False)
    state  = (ckpt.get('state_dict') or ckpt.get('model') or ckpt
              if isinstance(ckpt, dict) else ckpt)
    clean  = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean, strict=False)
    model.to(device)
    model.eval()
    logger.info(f"CSRNet on {device}")
    return model


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def estimate_crowd(frame: np.ndarray, model: CSRNet,
                   patch_size: int = 512) -> Tuple[int, np.ndarray]:
    device = next(model.parameters()).device
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb    = (rgb - _MEAN) / _STD

    new_h  = (h // 8) * 8
    new_w  = (w // 8) * 8
    if new_h != h or new_w != w:
        rgb = cv2.resize(rgb, (new_w, new_h))
    scale  = (h * w) / (new_h * new_w)

    density = np.zeros((new_h // 8, new_w // 8), dtype=np.float32)
    overlap  = patch_size // 4
    stride   = patch_size - overlap

    with torch.no_grad():
        for py in range(0, new_h, stride):
            for px in range(0, new_w, stride):
                py2 = min(py + patch_size, new_h)
                px2 = min(px + patch_size, new_w)
                py  = max(0, py2 - patch_size)
                px  = max(0, px2 - patch_size)
                patch  = rgb[py:py2, px:px2]
                tensor = torch.from_numpy(
                    patch.transpose(2, 0, 1)[np.newaxis]).to(device)
                out = model(tensor).squeeze().cpu().numpy()
                out = np.maximum(out, 0)
                oy1, oy2 = py//8, py2//8
                ox1, ox2 = px//8, px2//8
                region = density[oy1:oy2, ox1:ox2]
                if region.shape == out.shape:
                    density[oy1:oy2, ox1:ox2] = np.maximum(region, out)

    count = max(0, round(float(density.sum() * scale)))
    vis   = _colorize(density, (h, w))
    return count, vis


def _colorize(density, size):
    norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    norm = cv2.resize(norm, (size[1], size[0]))
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


def overlay_density(frame, density_vis, alpha=0.55):
    return cv2.addWeighted(frame, 1.0 - alpha, density_vis, alpha, 0)
