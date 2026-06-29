import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import YOLO_MODELS, DEVICE, PERSON_CLASS_ID
from src.utils import get_logger, apply_clahe, draw_body_box, draw_face_box
from src.face_recovery import load_face_net, detect_faces, nms_faces, FaceBox
from src.validator import validate_person, RejectionStats

logger = get_logger(__name__)


@dataclass
class DetectionResult:
    annotated_frame  : np.ndarray
    total_count      : int   = 0
    yolo_count       : int   = 0
    face_count       : int   = 0
    raw_yolo_count   : int   = 0
    rejected_count   : int   = 0
    proc_time_sec    : float = 0.0
    avg_confidence   : float = 0.0
    rejection_stats  : RejectionStats = field(default_factory=RejectionStats)
    crowd_density    : str   = "Low"


def _apply_torch_patch():
    try:
        import torch, ultralytics
        try:
            torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
        except Exception:
            pass
        _orig = torch.load
        def _patched(*a, **kw):
            kw['weights_only'] = False
            return _orig(*a, **kw)
        torch.load = _patched
    except Exception:
        pass

_apply_torch_patch()

_model_cache = {}

def load_yolo(model_name):
    if model_name not in _model_cache:
        from ultralytics import YOLO
        logger.info(f"Loading {model_name} on {DEVICE}")
        m = YOLO(model_name)
        m.to(DEVICE)
        _model_cache[model_name] = m
    return _model_cache[model_name]

_sahi_available = False
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _sahi_available = True
except ImportError:
    pass


def _nms_bodies(boxes, iou_thresh=0.45):
    """
    Aggressive dedup for SAHI tile duplicates.
    Removes a box if ANY is true vs a higher-confidence box:
      - IoU > threshold (standard overlap)
      - smaller box >65% contained in bigger box
      - centers very close AND boxes overlap
    Kills the stacked duplicate boxes SAHI makes per person,
    while keeping genuinely separate people side by side.
    """
    if not boxes:
        return []
    arr = np.array([[b[0][0], b[0][1], b[0][2], b[0][3], b[1]] for b in boxes], dtype=float)
    x1, y1, x2, y2, sc = arr[:,0], arr[:,1], arr[:,2], arr[:,3], arr[:,4]
    areas = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(boxes[i])
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = inter / np.maximum(union, 1)
        smaller = np.minimum(areas[i], areas[rest])
        contain = inter / np.maximum(smaller, 1)
        diag = np.sqrt((x2[i]-x1[i])**2 + (y2[i]-y1[i])**2)
        cdist = np.sqrt((cx[i]-cx[rest])**2 + (cy[i]-cy[rest])**2)
        close_center = (cdist < diag * 0.30) & (iou > 0.15)
        remove = (iou > iou_thresh) | (contain > 0.65) | close_center
        order = rest[np.where(~remove)[0]]
    return keep


def process_image(frame, model_label, conf_threshold, iou_threshold,
                  imgsz, use_sahi, use_face_recovery, face_sensitivity,
                  face_net, vis_bodies_only=False) -> DetectionResult:

    start = time.time()
    model_name = YOLO_MODELS[model_label]
    fh, fw = frame.shape[:2]
    stats = RejectionStats()

    model = load_yolo(model_name)
    frame = apply_clahe(frame)

    body_conf = conf_threshold

    quick = model(frame, conf=0.30, iou=0.45,
                  classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)
    est_count = max(sum(len(r.boxes) for r in quick), 1)

    # Face detection — this is what catches people YOLO body misses
    raw_faces: List[FaceBox] = []
    if use_face_recovery and face_net is not None:
        face_conf = 0.52 - (face_sensitivity * 0.032)
        face_conf = max(0.20, min(0.55, face_conf))
        raw_faces = detect_faces(
            frame, face_net, sensitivity=face_conf,
            use_tiling=use_sahi, estimated_count=est_count)
        logger.info(f"Faces: {len(raw_faces)}  crowd_est={est_count}")

    # YOLO body detection
    if use_sahi and _sahi_available:
        yolo_boxes, confidence_sum, raw_count = _run_sahi(
            frame, model_name, body_conf, fw, fh, raw_faces, stats)
    else:
        yolo_boxes, confidence_sum, raw_count = _run_standard_yolo(
            frame, model, body_conf, iou_threshold,
            imgsz, fw, fh, raw_faces, stats)

    # Remove duplicate bodies
    valid_bodies = _nms_bodies(yolo_boxes, iou_thresh=0.45)
    confidence_sum = sum(c for _, c in valid_bodies)

    # ── COUNTING STRATEGY ──────────────────────────────────
    # Every face = one person. Every body = one person.
    # A body and a face are the SAME person ONLY if the face
    # sits in the upper portion of that body box.
    # This way: front-facing people with a visible body+face = 1
    #           back-facing people with only a body = 1
    #           people in the back with only a face = 1
    matched_faces = set()
    for bi, ((bx1, by1, bx2, by2), _) in enumerate(valid_bodies):
        bh = by2 - by1
        head_zone_top = by1 - bh * 0.10
        head_zone_bot = by1 + bh * 0.50
        for fi, (fx1, fy1, fw2, fh2) in enumerate(raw_faces):
            if fi in matched_faces:
                continue
            fcx, fcy = fx1 + fw2 // 2, fy1 + fh2 // 2
            if bx1 < fcx < bx2 and head_zone_top < fcy < head_zone_bot:
                matched_faces.add(fi)
                break  # one face per body

    # Orphan faces = faces NOT matched to any body = people YOLO missed
    orphan_faces = [f for i, f in enumerate(raw_faces) if i not in matched_faces]

    yolo_count  = len(valid_bodies)
    face_count  = len(orphan_faces)
    total_count = yolo_count + face_count
    confidence_sum += face_count * 0.80
    avg_conf = (confidence_sum / total_count * 100) if total_count > 0 else 0.0
    density  = ("Low" if total_count < 30 else
                "Medium" if total_count < 100 else "High")

    for (x1, y1, x2, y2), conf in valid_bodies:
        draw_body_box(frame, x1, y1, x2, y2, conf)
    if not vis_bodies_only:
        for (fx1, fy1, fw2, fh2) in orphan_faces:
            draw_face_box(frame, fx1, fy1, fx1 + fw2, fy1 + fh2)

    return DetectionResult(
        annotated_frame=frame, total_count=total_count,
        yolo_count=yolo_count, face_count=face_count,
        raw_yolo_count=raw_count, rejected_count=stats.total(),
        proc_time_sec=time.time() - start, avg_confidence=avg_conf,
        rejection_stats=stats, crowd_density=density)


def _run_standard_yolo(frame, model, conf, iou, imgsz, fw, fh, raw_faces, stats):
    results = model(frame, conf=conf, iou=iou,
                    classes=[PERSON_CLASS_ID], imgsz=imgsz, verbose=False)
    valid, conf_sum, raw = [], 0.0, 0
    for r in results:
        for b in r.boxes:
            if int(b.cls[0]) != PERSON_CLASS_ID:
                continue
            raw += 1
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            c = float(b.conf[0])
            ok, reason = validate_person(x1, y1, x2, y2, c, fw, fh, raw_faces)
            if ok:
                valid.append(((x1, y1, x2, y2), c))
                conf_sum += c
            else:
                _inc(stats, reason)
    return valid, conf_sum, raw


def _run_sahi(frame, model_name, conf, fw, fh, raw_faces, stats):
    sm = AutoDetectionModel.from_pretrained(
        model_type='yolov8', model_path=model_name,
        confidence_threshold=conf, device=DEVICE)
    result = get_sliced_prediction(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), sm,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.20, overlap_width_ratio=0.20)
    valid, conf_sum, raw = [], 0.0, 0
    for obj in result.object_prediction_list:
        if obj.category.id != PERSON_CLASS_ID:
            continue
        raw += 1
        x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
        c = float(obj.score.value)
        ok, reason = validate_person(x1, y1, x2, y2, c, fw, fh, raw_faces)
        if ok:
            valid.append(((x1, y1, x2, y2), c))
            conf_sum += c
        else:
            _inc(stats, reason)
    return valid, conf_sum, raw


def _inc(stats, reason):
    if reason == "too_small":   stats.too_small   += 1
    elif reason == "too_large": stats.too_large   += 1
    elif reason == "too_wide":  stats.too_wide    += 1
    elif reason == "too_thin":  stats.too_thin    += 1
    else:                       stats.low_quality += 1
