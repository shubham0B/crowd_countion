# ============================================================
# src/video_processor.py — Video & Webcam Detection Pipeline
# ============================================================

import cv2
import numpy as np
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    YOLO_MODELS, DEVICE, PERSON_CLASS_ID,
    OUTPUTS_DIR, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD
)
from src.detector    import process_image, load_yolo, DetectionResult
from src.face_recovery import load_face_net, detect_faces
from src.heatmap     import HeatmapGenerator
from src.utils       import get_logger, apply_clahe, draw_body_box, draw_face_box, bgr_to_rgb
from src.validator   import validate_person, RejectionStats

logger = get_logger(__name__)


def process_video_file(
    video_path     : str,
    model_label    : str,
    conf_threshold : float = DEFAULT_CONF_THRESHOLD,
    iou_threshold  : float = DEFAULT_IOU_THRESHOLD,
    imgsz          : int   = 640,
    use_face_recovery: bool = True,
    show_heatmap   : bool  = True,
    save_output    : bool  = True,
    frame_skip     : int   = 1
):
    """
    Processes a video file frame by frame and saves the annotated output.

    PARAMETERS:
        video_path      — Path to the input video (.mp4, .avi, .mov)
        model_label     — YOLOv8 model to use (from YOLO_MODELS dict)
        conf_threshold  — Detection confidence cutoff
        iou_threshold   — NMS overlap threshold
        imgsz           — YOLO inference resolution
        use_face_recovery — Enable ResNet face detection
        show_heatmap    — Overlay density heatmap on the video
        save_output     — Save annotated video to outputs/
        frame_skip      — Process every Nth frame (1=all, 2=every other, etc.)
                          Use frame_skip=2 on CPU to double processing speed.

    EXECUTION FLOW PER FRAME:
        1. cap.read() — Reads one frame from the video
        2. apply_clahe() — Contrast enhancement
        3. face detector (if enabled) — Gets raw_faces
        4. YOLO inference — Gets raw body boxes
        5. validate_person() — Filters each body box
        6. Spatial merge — Removes face/body duplicates
        7. heatmap.update() — Accumulates heat at detection centers
        8. Draw boxes + heatmap overlay
        9. Write frame to output video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {width}x{height} @ {fps_in:.1f}fps, {total_frames} frames")

    # Output video writer
    out_writer = None
    if save_output:
        base_name  = os.path.splitext(os.path.basename(video_path))[0]
        out_path   = os.path.join(OUTPUTS_DIR, f"{base_name}_counted.mp4")
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))
        logger.info(f"Saving output to: {out_path}")

    # Load models
    model    = load_yolo(YOLO_MODELS[model_label])
    face_net = load_face_net() if use_face_recovery else None
    heatmap  = HeatmapGenerator(width, height) if show_heatmap else None

    frame_idx    = 0
    start_time   = time.time()
    last_count   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Skip frames for speed (re-use last count on skipped frames)
        if frame_idx % frame_skip != 0:
            if out_writer:
                out_writer.write(frame)
            continue

        # ── Preprocessing ────────────────────────────────
        frame = apply_clahe(frame)
        frame_h, frame_w = frame.shape[:2]
        stats = RejectionStats()

        # ── Face Detection ────────────────────────────────
        raw_faces = []
        if face_net is not None:
            raw_faces = detect_faces(frame, face_net,
                                     sensitivity=0.45,
                                     use_tiling=False)

        # ── YOLO Inference ────────────────────────────────
        results = model(
            frame,
            conf=conf_threshold,
            iou=iou_threshold,
            classes=[PERSON_CLASS_ID],
            imgsz=imgsz,
            verbose=False
        )

        # ── Validate and Collect ──────────────────────────
        valid_bodies   = []
        detection_centers = []

        for r in results:
            for b in r.boxes:
                if int(b.cls[0]) != PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf_score = float(b.conf[0])

                is_valid, reason = validate_person(
                    x1, y1, x2, y2, conf_score,
                    frame_w, frame_h, raw_faces
                )
                if is_valid:
                    valid_bodies.append(((x1, y1, x2, y2), conf_score))
                    detection_centers.append((x1, y1, x2, y2))

        # ── Find orphan faces (YOLO missed) ───────────────
        from src.detector import _find_orphan_faces
        orphan_faces = _find_orphan_faces(raw_faces, valid_bodies)

        last_count = len(valid_bodies) + len(orphan_faces)

        # ── Heatmap Update ────────────────────────────────
        if heatmap is not None:
            heatmap.update(detection_centers)
            frame = heatmap.get_overlay(frame, alpha=0.45)

        # ── Draw Boxes ────────────────────────────────────
        for (x1, y1, x2, y2), conf_score in valid_bodies:
            draw_body_box(frame, x1, y1, x2, y2, conf_score)
        for (fx1, fy1, fw, fh) in orphan_faces:
            draw_face_box(frame, fx1, fy1, fx1 + fw, fy1 + fh)

        # ── Count Banner ──────────────────────────────────
        elapsed = time.time() - start_time
        fps_out = frame_idx / elapsed if elapsed > 0 else 0
        banner  = f"Count: {last_count}  |  Frame: {frame_idx}/{total_frames}  |  FPS: {fps_out:.1f}"
        cv2.putText(frame, banner, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2, cv2.LINE_AA)

        if out_writer:
            out_writer.write(frame)

        if frame_idx % 30 == 0:
            logger.info(f"Frame {frame_idx}/{total_frames} — Count: {last_count} — FPS: {fps_out:.1f}")

    cap.release()
    if out_writer:
        out_writer.release()
        logger.info("Video processing complete.")


def process_webcam(
    camera_index   : int  = 0,
    model_label    : str  = "Nano  — Fastest  (0–30 people)",
    conf_threshold : float = 0.40,
    iou_threshold  : float = 0.45,
    show_heatmap   : bool  = True,
    window_name    : str   = "Crowd Counter — Press Q to quit"
):
    """
    Opens the webcam and runs real-time people counting.
    Shows a live OpenCV window with bounding boxes and optional heatmap.

    PRESS Q to quit.

    HOW REAL-TIME INFERENCE WORKS:
        cap = cv2.VideoCapture(0) — Opens camera index 0 (default webcam)
        cap.read() — Grabs one frame from the camera buffer
        YOLO runs on each frame
        cv2.imshow() displays the result
        cv2.waitKey(1) processes keyboard input (1ms wait)
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera index {camera_index}")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Webcam opened: {width}x{height}")

    model   = load_yolo(YOLO_MODELS[model_label])
    heatmap = HeatmapGenerator(width, height) if show_heatmap else None

    fps_counter    = 0
    fps_start      = time.time()
    display_fps    = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_clahe(frame)
        frame_h, frame_w = frame.shape[:2]

        results = model(frame, conf=conf_threshold, iou=iou_threshold,
                        classes=[PERSON_CLASS_ID], imgsz=640, verbose=False)

        valid_bodies      = []
        detection_centers = []
        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf_score = float(b.conf[0])
                is_valid, _ = validate_person(x1, y1, x2, y2, conf_score,
                                              frame_w, frame_h, [])
                if is_valid:
                    valid_bodies.append(((x1, y1, x2, y2), conf_score))
                    detection_centers.append((x1, y1, x2, y2))

        count = len(valid_bodies)

        if heatmap:
            heatmap.update(detection_centers)
            frame = heatmap.get_overlay(frame, alpha=0.40)

        for (x1, y1, x2, y2), cs in valid_bodies:
            draw_body_box(frame, x1, y1, x2, y2, cs)

        # FPS calculation
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_start   = time.time()

        cv2.putText(frame,
                    f"People: {count}  FPS: {display_fps:.1f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (100, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam session ended.")
