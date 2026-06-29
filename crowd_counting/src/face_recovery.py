import cv2
import numpy as np
import os
import sys
import requests
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_MODEL_URL, FACE_MODEL_PATH, FACE_TILE_SIZE, FACE_MIN_SIZE_PX, FACE_NMS_IOU
from src.utils import get_logger

logger = get_logger(__name__)
FaceBox = Tuple[int, int, int, int]


def download_face_model():
    os.makedirs(os.path.dirname(FACE_MODEL_PATH), exist_ok=True)
    if os.path.exists(FACE_MODEL_PATH):
        size_mb = os.path.getsize(FACE_MODEL_PATH) / (1024 * 1024)
        if size_mb > 3:
            logger.info("YOLOv8-Face model already downloaded")
            return
    logger.info("Downloading YOLOv8-Face model...")
    resp = requests.get(FACE_MODEL_URL, stream=True, timeout=180)
    resp.raise_for_status()
    with open(FACE_MODEL_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    logger.info(f"YOLOv8-Face downloaded: {FACE_MODEL_PATH}")


def load_face_net():
    download_face_model()
    from ultralytics import YOLO
    import torch
    model = YOLO(FACE_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"YOLOv8-Face loaded on {device}")
    return model


def detect_faces(frame, face_model, sensitivity=0.4,
                 use_tiling=True, estimated_count=0):
    h, w = frame.shape[:2]

    # Lower confidence floors = catch more front-facing faces
    if estimated_count < 25:
        conf = max(0.35, sensitivity)
        do_tiles = False
    elif estimated_count < 80:
        conf = max(0.30, sensitivity)
        do_tiles = use_tiling
    else:
        conf = max(0.22, sensitivity)
        do_tiles = use_tiling

    raw_faces: List[FaceBox] = []

    # Pass 1: full image
    results = face_model(frame, conf=conf, iou=0.45, imgsz=1280, verbose=False)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1
            if fw >= FACE_MIN_SIZE_PX and fh >= FACE_MIN_SIZE_PX:
                raw_faces.append((x1, y1, fw, fh))

    # Pass 2: tiled for small/distant faces in dense crowds
    if do_tiles:
        tile_conf = max(0.20, conf * 0.8)
        step = int(FACE_TILE_SIZE * 0.75)
        for ty in range(0, h, step):
            for tx in range(0, w, step):
                tile = frame[ty:ty + FACE_TILE_SIZE, tx:tx + FACE_TILE_SIZE]
                th, tw = tile.shape[:2]
                if th < 32 or tw < 32:
                    continue
                t_res = face_model(tile, conf=tile_conf, iou=0.45, imgsz=640, verbose=False)
                for r in t_res:
                    for b in r.boxes:
                        fx1, fy1, fx2, fy2 = map(int, b.xyxy[0])
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(tw, fx2), min(th, fy2)
                        fw2, fh2 = fx2 - fx1, fy2 - fy1
                        if fw2 >= FACE_MIN_SIZE_PX and fh2 >= FACE_MIN_SIZE_PX and fw2 <= 150 and fh2 <= 150:
                            raw_faces.append((tx + fx1, ty + fy1, fw2, fh2))

    return nms_faces(raw_faces, iou_thresh=FACE_NMS_IOU)


def nms_faces(faces: List[FaceBox], iou_thresh=0.35) -> List[FaceBox]:
    if not faces:
        return []
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in faces], dtype=float)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = areas.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(faces[i])
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep
