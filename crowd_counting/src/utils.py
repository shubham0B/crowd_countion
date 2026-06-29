import cv2
import numpy as np
import logging
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOG_FORMAT, LOG_LEVEL, COLOR_BODY, COLOR_FACE, FONT, FONT_SCALE, FONT_THICKNESS, BOX_THICKNESS

def get_logger(name):
    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT,
                        handlers=[logging.StreamHandler()])
    return logging.getLogger(name)

logger = get_logger(__name__)

def draw_body_box(frame, x1, y1, x2, y2, conf):
    h, w = frame.shape[:2]
    scale      = min(w, h) / 800.0
    thickness  = max(1, int(BOX_THICKNESS * scale))
    font_scale = max(0.3, FONT_SCALE * scale)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BODY, thickness)
    label = f"{conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, FONT_THICKNESS)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 2), (x1 + tw + 4, y1), COLOR_BODY, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - baseline - 1),
                FONT, font_scale, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)

def draw_face_box(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    scale = min(w, h) / 800.0
    thickness = max(1, int(1.5 * scale))
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_FACE, thickness)

def pil_to_bgr(pil_image):
    rgb = np.array(pil_image)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
