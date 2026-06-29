import cv2
import numpy as np
import time
import os
import sys
import threading
import queue

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_MODELS, DEVICE, PERSON_CLASS_ID, OUTPUTS_DIR
from src.utils import get_logger, apply_clahe, draw_body_box
from src.validator import validate_person, RejectionStats
from src.detector import load_yolo

logger = get_logger(__name__)


def detect_frame(frame, model, conf=0.25, iou=0.45, imgsz=640):
    fh, fw = frame.shape[:2]
    frame  = apply_clahe(frame)
    results = model(frame, conf=conf, iou=iou,
                    classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
    count = 0
    for r in results:
        for b in r.boxes:
            if int(b.cls[0]) != PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            c = float(b.conf[0])
            ok, _ = validate_person(x1, y1, x2, y2, c, fw, fh, [])
            if ok:
                draw_body_box(frame, x1, y1, x2, y2, c)
                count += 1
    cv2.rectangle(frame, (0, 0), (fw, 50), (20, 20, 20), -1)
    cv2.putText(frame, f"People: {count}", (12, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (100, 255, 100), 2, cv2.LINE_AA)
    return frame, count


def process_video(video_path, model_label, conf=0.25, iou=0.45,
                  frame_skip=2, progress_cb=None, stop_event=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    base     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(OUTPUTS_DIR, f"{base}_counted.mp4")
    writer   = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'),
        fps_in / frame_skip, (w, h)
    )

    model = load_yolo(YOLO_MODELS[model_label])
    idx   = 0
    start = time.time()

    while True:
        if stop_event and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % frame_skip != 0:
            continue
        out_frame, count = detect_frame(frame, model, conf=conf, imgsz=640)
        writer.write(out_frame)
        elapsed = time.time() - start
        fps_out = (idx // frame_skip) / max(elapsed, 0.001)
        if progress_cb:
            progress_cb(idx, total, count, fps_out)

    cap.release()
    writer.release()
    return out_path
