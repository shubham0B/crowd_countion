#!/usr/bin/env python3
"""
auto_label_college_data.py
=============================================================================
AI-Assisted Auto-Labeling Pipeline for College Crowd Data
=============================================================================
Features:
  1. Accepts images OR videos (automatically extracts keyframes from CCTV/phone clips).
  2. Uses YOLOv8 + SAHI (sliced inference) to detect distant, small students.
  3. Uses Face-Recovery fallback to catch seated/occluded students.
  4. Generates standard YOLO annotations (.txt) with normalized coordinates.
  5. Automatically splits data into train/val sets (80/20).
  6. Creates visual preview images in 'previews/' so you can review AI labels.
  7. Generates 'college_data.yaml' ready for instant training.

Usage:
  python auto_label_college_data.py --input ./college_raw_data --output ./college_dataset
=============================================================================
"""

import os
import sys
import glob
import cv2
import numpy as np
import argparse
import shutil
import random
from pathlib import Path

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from config import YOLO_MODELS, DEVICE, FACE_MODEL_PATH
from src.face_recovery import load_face_net, detect_faces, nms_faces
from src.validator import validate_person, RejectionStats
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False


def extract_frames_from_video(video_path: str, output_dir: str, frame_interval_sec: float = 1.0) -> list:
    """Extracts frames from a video at a specified interval."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(fps * frame_interval_sec))
    video_stem = Path(video_path).stem
    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            out_name = f"{video_stem}_f{frame_idx:06d}.jpg"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, frame)
            extracted.append(out_path)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"  📹 Extracted {saved_count} frames from {Path(video_path).name}")
    return extracted


def auto_detect_boxes(frame, yolo_model, face_model, conf_thresh=0.25, use_sahi=True):
    """
    Runs multi-stage detection (YOLO + SAHI + Face recovery)
    Returns: List of [x1, y1, x2, y2] in pixel coordinates
    """
    h, w = frame.shape[:2]
    all_boxes = []

    # 1. Base YOLO detection
    results = yolo_model(frame, conf=conf_thresh, iou=0.45, classes=[0], imgsz=1280, verbose=False)
    for r in results:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            all_boxes.append((x1, y1, x2, y2))

    # 2. SAHI sliced detection for dense/far-away students
    if use_sahi and SAHI_AVAILABLE:
        try:
            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model=yolo_model,
                confidence_threshold=conf_thresh,
                category_mapping={"0": "person"},
                device=DEVICE
            )
            sahi_res = get_sliced_prediction(
                frame,
                sahi_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                verbose=0
            )
            for obj in sahi_res.object_prediction_list:
                if obj.category.id == 0:
                    x1, y1, x2, y2 = map(int, obj.bbox.to_xyxy())
                    all_boxes.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        except Exception:
            pass

    # 3. Face Recovery (for seated/occluded students)
    if face_model is not None:
        try:
            faces = detect_faces(frame, face_model, sensitivity=0.35, use_tiling=True)
            for fx, fy, fw, fh in faces:
                # Estimate a body/upper-body box from face position
                est_w = int(fw * 2.2)
                est_h = int(fh * 3.5)
                cx = fx + fw // 2
                x1 = max(0, cx - est_w // 2)
                y1 = max(0, fy - int(fh * 0.2))
                x2 = min(w, x1 + est_w)
                y2 = min(h, y1 + est_h)
                all_boxes.append((x1, y1, x2, y2))
        except Exception:
            pass

    # 4. Non-Maximum Suppression (NMS) to merge overlapping boxes
    if not all_boxes:
        return []

    boxes_arr = np.array(all_boxes, dtype=float)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(all_boxes[i])
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
        min_areas = np.minimum(areas[i], areas[rest])
        ios = inter / np.maximum(min_areas, 1)
        remove = (iou > 0.40) | (ios > 0.55)
        order = rest[np.where(~remove)[0]]

    return keep


def main():
    parser = argparse.ArgumentParser(description="AI Auto-Labeling Pipeline for College Crowd Data")
    parser.add_argument("--input", type=str, default="college_raw_data",
                        help="Folder containing raw college images or videos")
    parser.add_argument("--output", type=str, default="college_dataset",
                        help="Destination directory for YOLO dataset")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for auto-detection")
    parser.add_argument("--val-split", type=float, default=0.20,
                        help="Fraction of data for validation (default 0.20 = 20 percent)")
    parser.add_argument("--frame-interval", type=float, default=1.5,
                        help="Interval in seconds for extracting video frames")
    parser.add_argument("--no-sahi", action="store_true",
                        help="Disable SAHI sliced inference")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        print(f"⚠️ Input directory '{args.input}' was created.")
        print(f"👉 Please place your college images (.jpg, .png) or videos (.mp4) inside '{input_dir}' and run again.")
        return

    # Dataset structure
    images_train = os.path.join(output_dir, "images", "train")
    images_val   = os.path.join(output_dir, "images", "val")
    labels_train = os.path.join(output_dir, "labels", "train")
    labels_val   = os.path.join(output_dir, "labels", "val")
    previews_dir = os.path.join(output_dir, "previews")

    for d in [images_train, images_val, labels_train, labels_val, previews_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("🎓 AI-POWERED AUTO-LABELING PIPELINE FOR COLLEGE DATA")
    print("=" * 60)
    print(f"📂 Input directory:   {input_dir}")
    print(f"💾 Output dataset:   {output_dir}")
    print(f"⚡ Device:           {DEVICE.upper()}")
    print("-" * 60)

    # 1. Collect all images and extract videos
    all_image_paths = []
    temp_frames_dir = os.path.join(output_dir, "_extracted_frames")

    # Find video files
    video_exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv")
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(input_dir, "**", ext), recursive=True))

    if video_files:
        print(f"\n🎥 Found {len(video_files)} video file(s). Extracting frames...")
        for vf in video_files:
            frames = extract_frames_from_video(vf, temp_frames_dir, frame_interval_sec=args.frame_interval)
            all_image_paths.extend(frames)

    # Find existing image files
    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    for ext in img_exts:
        for p in glob.glob(os.path.join(input_dir, "**", ext), recursive=True):
            if temp_frames_dir not in p:
                all_image_paths.append(p)

    if not all_image_paths:
        print(f"\n❌ No images or videos found in '{input_dir}'!")
        print("👉 Add some images (.jpg, .png) or video clips (.mp4) and run this script.")
        return

    # Deduplicate paths
    all_image_paths = list(set(all_image_paths))
    random.seed(42)
    random.shuffle(all_image_paths)

    total_images = len(all_image_paths)
    val_count = max(1, int(total_images * args.val_split))
    val_paths = set(all_image_paths[:val_count])

    print(f"\n🖼️  Total images to label: {total_images} (Train: {total_images - val_count}, Val: {val_count})")

    # Load models
    print(f"\n🧠 Loading AI models on {DEVICE.upper()} (YOLOv8 Large + YOLOv8 Face)...")
    yolo_model = YOLO("yolov8l.pt")
    yolo_model.to(DEVICE)
    face_model = None
    try:
        if os.path.exists(FACE_MODEL_PATH):
            face_model = load_face_net()
    except Exception as e:
        print(f"  Note: Face model skipped ({e})")

    print("\n🚀 Starting auto-labeling...")
    total_persons_labeled = 0

    for idx, img_path in enumerate(all_image_paths, 1):
        img_name = Path(img_path).name
        stem = Path(img_path).stem
        is_val = img_path in val_paths

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Run AI detection
        boxes = auto_detect_boxes(
            frame, yolo_model, face_model,
            conf_thresh=args.conf,
            use_sahi=not args.no_sahi
        )
        total_persons_labeled += len(boxes)

        # Target directories
        target_img_dir = images_val if is_val else images_train
        target_lbl_dir = labels_val if is_val else labels_train

        # Copy image
        target_img_path = os.path.join(target_img_dir, img_name)
        shutil.copy2(img_path, target_img_path)

        # Write YOLO label file (.txt)
        # Format: 0 x_center y_center width height (normalized 0.0 - 1.0)
        target_lbl_path = os.path.join(target_lbl_dir, f"{stem}.txt")
        with open(target_lbl_path, "w") as f:
            for x1, y1, x2, y2 in boxes:
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + bw / 2.0
                cy = y1 + bh / 2.0
                f.write(f"0 {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")

        # Create a preview image with bounding boxes drawn
        preview_frame = frame.copy()
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(preview_frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
            cv2.putText(preview_frame, "person", (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)

        cv2.putText(preview_frame, f"AI Count: {len(boxes)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(previews_dir, f"preview_{stem}.jpg"), preview_frame)

        if idx % 10 == 0 or idx == total_images:
            print(f"  [{idx}/{total_images}] Labeled '{img_name}' — Found {len(boxes)} people")

    # Generate dataset.yaml
    yaml_content = f"""# College Crowd Dataset (AI Auto-Labeled)
path: {output_dir}
train: images/train
val: images/val

names:
  0: person
"""
    yaml_path = os.path.join(output_dir, "college_data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("\n" + "=" * 60)
    print("🎉 AUTO-LABELING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Total Images:          {total_images}")
    print(f"👥 Total People Labeled:  {total_persons_labeled}")
    print(f"📁 Dataset Directory:     {output_dir}")
    print(f"📄 Dataset YAML:          {yaml_path}")
    print(f"🖼️  Preview Images:        {previews_dir}")
    print("-" * 60)
    print("👉 Check the 'previews/' folder to visually review the AI annotations!")
    print("👉 To train the model on this data, run:")
    print(f"   python train_college_model.py --data \"{yaml_path}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()
