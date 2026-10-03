import os
import sys
import io
import time
import shutil
import zipfile
import tempfile
import inspect
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from config import BASE_DIR, MODELS_DIR, YOLO_MODELS, DEVICE
from src.utils import pil_to_bgr, bgr_to_rgb

# ─────────────────────────────────────────
# VERSION-SAFE IMAGE DISPLAY
# ─────────────────────────────────────────
_IMG_PARAMS = set(inspect.signature(st.image).parameters.keys())

def show_image(container, img, **kwargs):
    caption = kwargs.get("caption", None)
    if "use_container_width" in _IMG_PARAMS:
        container.image(img, use_container_width=True, caption=caption)
    elif "use_column_width" in _IMG_PARAMS:
        container.image(img, use_column_width=True, caption=caption)
    else:
        container.image(img, caption=caption)


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def render_boxes_on_bgr(frame_bgr, boxes, title_count=None):
    """Draws clear bounding boxes with person tags and header badge."""
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]
    scale = max(0.5, min(w, h) / 900.0)
    thickness = max(2, int(2 * scale))
    font_scale = max(0.38, 0.45 * scale)
    
    for i, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 100), thickness)
        tag = f"#{i} person"
        (tw, th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 6, max(th + 4, y1)), (0, 255, 100), -1)
        cv2.putText(vis, tag, (x1 + 3, max(th + 1, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        
    count_val = len(boxes) if title_count is None else title_count
    badge_h = max(38, int(46 * scale))
    badge_w = max(240, int(280 * scale))
    cv2.rectangle(vis, (15, 15), (15 + badge_w, 15 + badge_h), (10, 25, 15), -1)
    cv2.rectangle(vis, (15, 15), (15 + badge_w, 15 + badge_h), (0, 255, 100), 2)
    cv2.putText(vis, f"STUDENTS DETECTED: {count_val}", (25, 15 + int(badge_h * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.15, (0, 255, 150), max(1, int(1.5 * scale)), cv2.LINE_AA)
    return vis


def save_image_and_labels_to_disk(img_path, boxes, dataset_dir, is_val=False):
    """Saves image and YOLO normalized .txt labels into the training dataset folder."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(img_path)[1]
    if not ext:
        ext = ".jpg"
        
    img_folder = os.path.join(dataset_dir, "images", "val" if is_val else "train")
    lbl_folder = os.path.join(dataset_dir, "labels", "val" if is_val else "train")
    prev_folder = os.path.join(dataset_dir, "previews")
    for d in [img_folder, lbl_folder, prev_folder]:
        os.makedirs(d, exist_ok=True)
        
    # 1. Copy or write image
    dest_img_path = os.path.join(img_folder, f"{stem}{ext}")
    shutil.copy2(img_path, dest_img_path)
    
    # 2. Write YOLO format .txt file (class_id cx cy w h normalized)
    frame = cv2.imread(img_path)
    if frame is not None:
        fh, fw = frame.shape[:2]
        txt_path = os.path.join(lbl_folder, f"{stem}.txt")
        with open(txt_path, "w", encoding="utf-8") as tf:
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                bw = max(1.0, float(x2 - x1))
                bh = max(1.0, float(y2 - y1))
                cx = float(x1) + bw / 2.0
                cy = float(y1) + bh / 2.0
                norm_cx = max(0.0, min(1.0, cx / fw))
                norm_cy = max(0.0, min(1.0, cy / fh))
                norm_w = max(0.0, min(1.0, bw / fw))
                norm_h = max(0.0, min(1.0, bh / fh))
                tf.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        # 3. Save visual preview
        vis = render_boxes_on_bgr(frame, boxes)
        cv2.imwrite(os.path.join(prev_folder, f"prev_{stem}.jpg"), vis)
        
    # 4. Generate / Update college_data.yaml
    yaml_path = os.path.join(dataset_dir, "college_data.yaml")
    clean_ds_path = os.path.abspath(dataset_dir).replace('\\', '/')
    yaml_content = f"""# VisionAI Custom Labeled Dataset for Crowd Counting & Training
path: {clean_ds_path}
train: images/train
val: images/val
names:
  0: person
"""
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)


def load_existing_boxes_from_dataset(img_path, dataset_dir):
    """Loads existing YOLO bounding boxes if image was already saved previously."""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    frame = cv2.imread(img_path)
    if frame is None:
        return []
    fh, fw = frame.shape[:2]
    
    for sub in ["train", "val"]:
        lbl_file = os.path.join(dataset_dir, "labels", sub, f"{stem}.txt")
        if os.path.exists(lbl_file):
            boxes = []
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, cx, cy, bw, bh = map(float, parts[:5])
                        w_px = bw * fw
                        h_px = bh * fh
                        x1 = (cx * fw) - (w_px / 2.0)
                        y1 = (cy * fh) - (h_px / 2.0)
                        x2 = x1 + w_px
                        y2 = y1 + h_px
                        boxes.append((int(max(0, x1)), int(max(0, y1)), int(min(fw, x2)), int(min(fh, y2))))
            return boxes
    return []


# ─────────────────────────────────────────
# MAIN STUDIO TAB RENDERER
# ─────────────────────────────────────────

def render_studio_tab(tab_container, model_label, face_net):
    with tab_container:
        st.markdown("## 🏷️ Live AI Auto-Labeler & Dataset Studio")
        st.markdown(
            "Upload your **unlabeled dataset**, inspect unlabeled and AI-labeled images **side-by-side**, "
            "control every picture with **Next / Prev**, click **🤖 Add Label** to auto-detect students, and click **💾 Save** to store them in your training dataset."
        )

        # ── SESSION STATE INITIALIZATION ──
        if 'studio_idx' not in st.session_state:
            st.session_state['studio_idx'] = 0
        if 'studio_boxes' not in st.session_state:
            st.session_state['studio_boxes'] = {}
        if 'studio_img_files' not in st.session_state:
            st.session_state['studio_img_files'] = []

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        temp_upload_dir = os.path.join(BASE_DIR, "unlabeled_uploads")
        os.makedirs(temp_upload_dir, exist_ok=True)

        # ─────────────────────────────────────────
        # STEP 1: UPLOAD & IMPORT UNLABELED DATASET
        # ─────────────────────────────────────────
        st.markdown("---")
        with st.container():
            st.markdown("### 📥 1. Upload or Select Unlabeled Dataset")
            
            src_col1, src_col2 = st.columns([1.2, 1.8])
            with src_col1:
                input_mode = st.radio(
                    "Choose Dataset Import Method:",
                    [
                        "📤 Upload Images / ZIP File",
                        "📁 System Folder of Images",
                        "🎥 Extract Frames from Video"
                    ],
                    index=0,
                    help="Easily upload photos or a zip folder containing your unlabelled dataset."
                )

            with src_col2:
                # OPTION 1: Upload Images / ZIP Archive
                if input_mode == "📤 Upload Images / ZIP File":
                    uploaded_files = st.file_uploader(
                        "Drag & Drop Unlabeled Images or a .ZIP Dataset Archive",
                        type=["jpg", "jpeg", "png", "bmp", "webp", "zip"],
                        accept_multiple_files=True,
                        help="Upload multiple pictures or a .zip file with unlabelled crowd images."
                    )
                    
                    if uploaded_files:
                        new_files_added = 0
                        for uf in uploaded_files:
                            if uf.name.lower().endswith(".zip"):
                                with zipfile.ZipFile(io.BytesIO(uf.read()), 'r') as z:
                                    for zinfo in z.infolist():
                                        if not zinfo.is_dir() and zinfo.filename.lower().endswith(valid_exts):
                                            # Avoid folder prefix path issues
                                            fname = os.path.basename(zinfo.filename)
                                            if fname:
                                                out_p = os.path.join(temp_upload_dir, fname)
                                                with open(out_p, "wb") as f_out:
                                                    f_out.write(z.read(zinfo))
                                                new_files_added += 1
                            else:
                                out_p = os.path.join(temp_upload_dir, uf.name)
                                with open(out_p, "wb") as f_out:
                                    f_out.write(uf.getbuffer())
                                new_files_added += 1
                                
                        # Re-scan temp_upload_dir
                        found_files = []
                        for root, _, files in os.walk(temp_upload_dir):
                            for f in sorted(files):
                                if f.lower().endswith(valid_exts):
                                    found_files.append(os.path.join(root, f))
                        st.session_state['studio_img_files'] = found_files
                        st.success(f"✅ Loaded **{len(found_files)} unlabelled images** from upload!")

                # OPTION 2: System Folder of Images
                elif input_mode == "📁 System Folder of Images":
                    default_folder = r"C:\Users\sharm\Downloads\cctv\jecrc_1000_augmented_dataset" if os.path.exists(r"C:\Users\sharm\Downloads\cctv\jecrc_1000_augmented_dataset") else os.path.join(BASE_DIR, "college_raw_data")
                    folder_path_input = st.text_input(
                        "📂 Path to Unlabelled Images Folder on System",
                        value=default_folder,
                        help="Path containing unlabelled photos"
                    )
                    if st.button("🔄 Scan & Load Images from Folder", use_container_width=True):
                        found_files = []
                        if os.path.exists(folder_path_input):
                            for root, _, files in os.walk(folder_path_input):
                                for f in sorted(files):
                                    if f.lower().endswith(valid_exts):
                                        found_files.append(os.path.join(root, f))
                            st.session_state['studio_img_files'] = found_files
                            st.session_state['studio_idx'] = 0
                            st.success(f"✅ Loaded **{len(found_files)} images** from `{folder_path_input}`")
                        else:
                            st.warning(f"Directory not found: {folder_path_input}")

                # OPTION 3: Extract from Video
                elif input_mode == "🎥 Extract Frames from Video":
                    uploaded_vid = st.file_uploader(
                        "Upload CCTV / Phone Video (.mp4, .avi, .mov, .mkv)",
                        type=["mp4", "avi", "mov", "mkv"]
                    )
                    v_rate = st.slider("Keyframe Extraction Rate (1 frame every N seconds)", 0.2, 5.0, 1.0, 0.2)
                    if uploaded_vid and st.button("🎬 Extract Frames from Video", type="primary"):
                        from auto_label_college_data import extract_frames_from_video
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
                            tmp_v.write(uploaded_vid.read())
                            tmp_v_path = tmp_v.name
                        vid_frames_dir = os.path.join(temp_upload_dir, "video_frames")
                        with st.spinner("Extracting keyframes from video..."):
                            extracted_imgs = extract_frames_from_video(tmp_v_path, vid_frames_dir, frame_interval_sec=v_rate)
                        try: os.unlink(tmp_v_path)
                        except Exception: pass
                        st.session_state['studio_img_files'] = extracted_imgs
                        st.session_state['studio_idx'] = 0
                        st.success(f"✅ Extracted **{len(extracted_imgs)} frames** from video!")

        # ── CHECK LOADED UNLABELED IMAGES ──
        # Fallback if no images currently in state: scan college_raw_data, JECRC dataset, or temp_upload_dir
        if not st.session_state['studio_img_files']:
            fallback_files = []
            for candidate_dir in [
                os.path.join(BASE_DIR, "college_raw_data"),
                r"C:\Users\sharm\Downloads\cctv\jecrc_1000_augmented_dataset",
                temp_upload_dir
            ]:
                if os.path.exists(candidate_dir):
                    for root, _, files in os.walk(candidate_dir):
                        for f in sorted(files):
                            if f.lower().endswith(valid_exts):
                                fp = os.path.join(root, f)
                                if fp not in fallback_files:
                                    fallback_files.append(fp)
            st.session_state['studio_img_files'] = fallback_files

        img_files = st.session_state['studio_img_files']
        total_images = len(img_files)

        # ─────────────────────────────────────────
        # TARGET TRAINING DATASET SETTINGS
        # ─────────────────────────────────────────
        with st.expander("⚙️ Target Training Dataset & AI Settings", expanded=False):
            sett_c1, sett_c2, sett_c3 = st.columns([2, 1, 1])
            with sett_c1:
                target_dataset_dir = st.text_input(
                    "💾 Target Training Dataset Directory (Where Labeled Images & YOLO .txt Are Stored)",
                    value=os.path.join(BASE_DIR, "college_dataset"),
                    help="Directory where validated training images, YOLO labels, and YAML configuration will be saved."
                )
            with sett_c2:
                auto_conf = st.slider("AI Auto-Detection Confidence", 0.10, 0.80, 0.25, 0.05)
            with sett_c3:
                use_face_lbl = st.checkbox("Face Recovery Fallback", value=True)
                use_sahi_lbl = st.checkbox("SAHI Tiling (Distant Crowd)", value=False)
                
            if st.button("🧹 Clear All Uploaded Images Cache", help="Deletes temporary uploaded images"):
                for root, _, files in os.walk(temp_upload_dir):
                    for f in files:
                        try: os.unlink(os.path.join(root, f))
                        except Exception: pass
                st.session_state['studio_img_files'] = []
                st.session_state['studio_boxes'] = {}
                st.session_state['studio_idx'] = 0
                st.rerun()

        if 'target_dataset_dir' not in dir():
            target_dataset_dir = os.path.join(BASE_DIR, "college_dataset")
        if 'auto_conf' not in dir():
            auto_conf = 0.25
        if 'use_face_lbl' not in dir():
            use_face_lbl = True
        if 'use_sahi_lbl' not in dir():
            use_sahi_lbl = False

        if total_images == 0:
            st.info("💡 **No unlabelled images found yet.** Please upload images, drag-and-drop a ZIP archive, or specify a folder above to start!")
            return

        # Clamp index
        if st.session_state['studio_idx'] >= total_images:
            st.session_state['studio_idx'] = total_images - 1
        if st.session_state['studio_idx'] < 0:
            st.session_state['studio_idx'] = 0

        current_idx = st.session_state['studio_idx']
        current_img_path = img_files[current_idx]
        current_filename = os.path.basename(current_img_path)
        current_stem = os.path.splitext(current_filename)[0]

        # Check if already saved in training dataset
        train_lbl = os.path.join(target_dataset_dir, "labels", "train", f"{current_stem}.txt")
        val_lbl = os.path.join(target_dataset_dir, "labels", "val", f"{current_stem}.txt")
        is_already_saved = os.path.exists(train_lbl) or os.path.exists(val_lbl)
        saved_split = "Validation" if os.path.exists(val_lbl) else ("Train" if os.path.exists(train_lbl) else None)

        # Auto-load existing saved boxes into session cache if present
        if current_img_path not in st.session_state['studio_boxes']:
            if is_already_saved:
                loaded_b = load_existing_boxes_from_dataset(current_img_path, target_dataset_dir)
                st.session_state['studio_boxes'][current_img_path] = loaded_b
            else:
                st.session_state['studio_boxes'][current_img_path] = []

        current_boxes = st.session_state['studio_boxes'].get(current_img_path, [])

        # ─────────────────────────────────────────
        # STEP 2: PICTURE CONTROLLER & ACTION TOOLBAR (SIDE-BY-SIDE BUTTONS)
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎮 2. Picture Navigator & Labeling Controls")

        # Current Picture Header & Status Banner
        if is_already_saved:
            status_html = f"<span style='color: #00ff66; font-weight: bold;'>🟢 Saved in Training Dataset ({saved_split} Split)</span>"
        elif current_boxes:
            status_html = f"<span style='color: #ffd700; font-weight: bold;'>🟡 AI Labeled ({len(current_boxes)} Students Marked — Not Saved Yet)</span>"
        else:
            status_html = "<span style='color: #a0aec0; font-weight: bold;'>⚪ Unlabeled (Waiting for Labels)</span>"

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 12px 18px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div style="font-size: 1.15rem; font-weight: 600;">
                    🖼️ Picture <b>{current_idx + 1}</b> of <b>{total_images}</b> — <code style="color: #00e5ff; font-size: 1.05rem;">{current_filename}</code>
                </div>
                <div style="font-size: 1.0rem;">
                    Status: {status_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Button Controls: Prev, Next, Add Label, Save, Save & Next, Clear
        ctl_col1, ctl_col2, ctl_col3, ctl_col4, ctl_col5, ctl_col6 = st.columns([1.4, 1.4, 2.0, 2.0, 2.2, 1.2])

        # Button 1: Previous Picture
        with ctl_col1:
            if st.button("◀ Prev Image", use_container_width=True, help="Go to previous unlabelled image"):
                if st.session_state['studio_idx'] > 0:
                    st.session_state['studio_idx'] -= 1
                    st.rerun()

        # Button 2: Next Picture
        with ctl_col2:
            if st.button("Next ▶ Image", use_container_width=True, help="Go to next unlabelled image"):
                if st.session_state['studio_idx'] < total_images - 1:
                    st.session_state['studio_idx'] += 1
                    st.rerun()

        # Button 3: 🤖 Add Label
        with ctl_col3:
            if st.button("🤖 Add Label", type="secondary", use_container_width=True, help="Run AI detector to automatically find and mark all students in this picture"):
                with st.spinner("🤖 Running AI detection (YOLO + Face Recovery)..."):
                    from auto_label_college_data import auto_detect_boxes
                    from src.detector import load_yolo
                    raw_bgr = cv2.imread(current_img_path)
                    if raw_bgr is not None:
                        model_name = YOLO_MODELS[model_label]
                        y_model = load_yolo(model_name)
                        f_model = face_net if use_face_lbl else None
                        detected = auto_detect_boxes(raw_bgr, y_model, f_model, conf_thresh=auto_conf, use_sahi=use_sahi_lbl)
                        st.session_state['studio_boxes'][current_img_path] = detected
                        st.toast(f"🎯 Detected & Added labels for {len(detected)} students!")
                        st.rerun()

        # Button 4: 💾 Save
        with ctl_col4:
            if st.button("💾 Save to Dataset", type="primary", use_container_width=True, help="Save this labeled picture & YOLO annotation file to the target training dataset"):
                boxes_to_save = current_boxes
                if not boxes_to_save:
                    # Auto run detection if user saves directly
                    from auto_label_college_data import auto_detect_boxes
                    from src.detector import load_yolo
                    raw_bgr = cv2.imread(current_img_path)
                    if raw_bgr is not None:
                        model_name = YOLO_MODELS[model_label]
                        y_model = load_yolo(model_name)
                        f_model = face_net if use_face_lbl else None
                        boxes_to_save = auto_detect_boxes(raw_bgr, y_model, f_model, conf_thresh=auto_conf, use_sahi=use_sahi_lbl)
                        st.session_state['studio_boxes'][current_img_path] = boxes_to_save

                # 80/20 train/val split rule
                is_val_split = (current_idx % 5 == 0)
                save_image_and_labels_to_disk(current_img_path, boxes_to_save, target_dataset_dir, is_val=is_val_split)
                st.toast(f"✅ Saved '{current_filename}' ({len(boxes_to_save)} boxes) to {'val' if is_val_split else 'train'} dataset!")
                st.rerun()

        # Button 5: 💾 Save & Next (1-Click Workflow Fast Track)
        with ctl_col5:
            if st.button("💾 Save & Next ⏭", use_container_width=True, help="Save labeled picture to dataset AND immediately advance to next picture"):
                boxes_to_save = current_boxes
                if not boxes_to_save:
                    from auto_label_college_data import auto_detect_boxes
                    from src.detector import load_yolo
                    raw_bgr = cv2.imread(current_img_path)
                    if raw_bgr is not None:
                        model_name = YOLO_MODELS[model_label]
                        y_model = load_yolo(model_name)
                        f_model = face_net if use_face_lbl else None
                        boxes_to_save = auto_detect_boxes(raw_bgr, y_model, f_model, conf_thresh=auto_conf, use_sahi=use_sahi_lbl)
                        st.session_state['studio_boxes'][current_img_path] = boxes_to_save

                is_val_split = (current_idx % 5 == 0)
                save_image_and_labels_to_disk(current_img_path, boxes_to_save, target_dataset_dir, is_val=is_val_split)
                st.toast(f"✅ Saved & moving to next picture!")
                if st.session_state['studio_idx'] < total_images - 1:
                    st.session_state['studio_idx'] += 1
                st.rerun()

        # Button 6: 🗑️ Clear
        with ctl_col6:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear labels on current image"):
                st.session_state['studio_boxes'][current_img_path] = []
                st.rerun()

        # Quick Jump Slider & Fast Bounds
        jump_col1, jump_col2, jump_col3 = st.columns([1, 6, 1])
        with jump_col1:
            if st.button("⏮ First", use_container_width=True):
                st.session_state['studio_idx'] = 0
                st.rerun()
        with jump_col2:
            jump_idx = st.slider(
                "Quick Jump Slider",
                min_value=1,
                max_value=total_images,
                value=current_idx + 1,
                key=f"jump_slider_{current_idx}",
                label_visibility="collapsed"
            )
            if jump_idx - 1 != current_idx:
                st.session_state['studio_idx'] = jump_idx - 1
                st.rerun()
        with jump_col3:
            if st.button("Last ⏭", use_container_width=True):
                st.session_state['studio_idx'] = total_images - 1
                st.rerun()

        # ─────────────────────────────────────────
        # STEP 3: SIDE-BY-SIDE VISUAL COMPARISON (UNLABELED VS LABELED)
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🖼️ 3. Side-by-Side Visual Comparison")

        raw_bgr = cv2.imread(current_img_path)
        if raw_bgr is not None:
            rh, rw = raw_bgr.shape[:2]
            col_unlabeled, col_labeled = st.columns(2)

            # LEFT COLUMN: Original Unlabeled Image
            with col_unlabeled:
                st.markdown(f"#### 📷 Original Unlabeled Image (`{rw} × {rh} px`)")
                show_image(st, bgr_to_rgb(raw_bgr), caption=f"Raw Unlabeled: {current_filename}")

            # RIGHT COLUMN: Labeled Image (AI / Custom Annotated)
            with col_labeled:
                st.markdown(f"#### 🏷️ AI Labeled Image (`{len(current_boxes)} Students Marked`)")
                if current_boxes:
                    annotated_bgr = render_boxes_on_bgr(raw_bgr, current_boxes)
                    show_image(st, bgr_to_rgb(annotated_bgr), caption=f"AI Annotated: {len(current_boxes)} student bounding boxes")
                else:
                    st.info("💡 **No labels generated yet for this image.**\n\nClick **`🤖 Add Label`** or **`💾 Save & Next`** above to auto-detect students!")
                    show_image(st, bgr_to_rgb(raw_bgr), caption="Waiting for labels...")

            # ─────────────────────────────────────────
            # STEP 3.1: MANAGE, ADD & REMOVE LABELS STUDIO
            # ─────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"### 🏷️ 4. Manage, Add & Remove Labels (Currently **{len(current_boxes)} People** Marked)")
            st.markdown("Inspect all marked people below. Click **🗑️ Remove** on any false positive box or duplicate to delete it instantly, or use **➕ Add Label** to add missing students.")

            # Top Utilities Row
            util_col1, util_col2, util_col3 = st.columns([2.5, 2, 2])
            with util_col1:
                if st.button("🧹 Auto-Clean Duplicate Overlaps (Smart NMS)", use_container_width=True, help="Automatically merges overlapping boxes on the same person"):
                    if current_boxes:
                        b_arr = np.array(current_boxes, dtype=float)
                        x1 = b_arr[:, 0]; y1 = b_arr[:, 1]; x2 = b_arr[:, 2]; y2 = b_arr[:, 3]
                        areas = (x2 - x1) * (y2 - y1)
                        order = areas.argsort()[::-1]
                        cleaned = []
                        while order.size > 0:
                            i = order[0]
                            cleaned.append(current_boxes[i])
                            if order.size == 1: break
                            rest = order[1:]
                            xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
                            xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
                            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
                            union = areas[i] + areas[rest] - inter
                            iou = inter / np.maximum(union, 1)
                            ios = inter / np.maximum(np.minimum(areas[i], areas[rest]), 1)
                            rem = (iou > 0.35) | (ios > 0.50)
                            order = rest[np.where(~rem)[0]]
                        st.session_state['studio_boxes'][current_img_path] = cleaned
                        st.toast(f"🧹 Cleaned duplicates! Now {len(cleaned)} people marked.")
                        st.rerun()

            with util_col2:
                if st.button("🗑️ Clear All Labels for This Picture", use_container_width=True):
                    st.session_state['studio_boxes'][current_img_path] = []
                    st.toast("🗑️ Cleared all labels on this image.")
                    st.rerun()

            with util_col3:
                st.info(f"📊 **{len(current_boxes)} person boxes** on this picture")

            # 4.1 VISUAL CROP GALLERY WITH 1-CLICK REMOVE
            st.markdown("#### 👥 Labeled People in This Picture (Click 🗑️ to Delete False Positives):")
            if current_boxes:
                cols_per_row = 4
                for row_start in range(0, len(current_boxes), cols_per_row):
                    row_boxes = current_boxes[row_start:row_start + cols_per_row]
                    row_cols = st.columns(cols_per_row)
                    for col_i, (b_idx, box) in enumerate(zip(range(row_start, row_start + len(row_boxes)), row_boxes)):
                        bx1, by1, bx2, by2 = map(int, box[:4])
                        bx1, by1 = max(0, bx1), max(0, by1)
                        bx2, by2 = min(rw, bx2), min(rh, by2)
                        
                        crop = raw_bgr[by1:by2, bx1:bx2]
                        with row_cols[col_i]:
                            st.markdown(f"**Person #{b_idx + 1}** `[{bx2 - bx1}×{by2 - by1}px]`")
                            if crop.size > 0 and crop.shape[0] > 2 and crop.shape[1] > 2:
                                show_image(st, bgr_to_rgb(crop))
                            else:
                                st.caption("No thumbnail")
                            if st.button(f"🗑️ Remove #{b_idx + 1}", key=f"del_crop_{current_stem}_{b_idx}", type="secondary", use_container_width=True):
                                new_b_list = list(current_boxes)
                                new_b_list.pop(b_idx)
                                st.session_state['studio_boxes'][current_img_path] = new_b_list
                                st.toast(f"🗑️ Removed Person #{b_idx + 1}! Total now: {len(new_b_list)}")
                                st.rerun()
            else:
                st.info("ℹ️ No person labels currently on this picture. Use **🤖 Add Label** above or **➕ Add New Person Box** below.")

            # 4.2 ADD NEW LABEL TOOLS
            st.markdown("---")
            st.markdown("#### ➕ Add New Labels to Picture:")
            add_tab1, add_tab2 = st.tabs(["📐 Add Custom Person Box", "🤖 Re-Detect with Custom AI Confidence"])

            with add_tab1:
                st.markdown("Manually add a box for any missed person:")
                preset = st.selectbox(
                    "💡 Quick Location Presets:",
                    ["Custom Coordinates", "📍 Speaker / Presenter (Front Area)", "📍 Left Bench Students Row 1", "📍 Left Bench Students Row 2", "📍 Right Bench Students"],
                    key=f"preset_{current_stem}"
                )

                def_x1, def_y1, def_x2, def_y2 = 0, 0, min(rw, 150), min(rh, 250)
                if preset == "📍 Speaker / Presenter (Front Area)":
                    def_x1, def_y1, def_x2, def_y2 = int(rw * 0.30), int(rh * 0.30), int(rw * 0.45), int(rh * 0.65)
                elif preset == "📍 Left Bench Students Row 1":
                    def_x1, def_y1, def_x2, def_y2 = int(rw * 0.40), int(rh * 0.22), int(rw * 0.55), int(rh * 0.40)
                elif preset == "📍 Left Bench Students Row 2":
                    def_x1, def_y1, def_x2, def_y2 = int(rw * 0.45), int(rh * 0.20), int(rw * 0.60), int(rh * 0.38)
                elif preset == "📍 Right Bench Students":
                    def_x1, def_y1, def_x2, def_y2 = int(rw * 0.60), int(rh * 0.20), int(rw * 0.80), int(rh * 0.45)

                ac1, ac2, ac3, ac4 = st.columns(4)
                new_x1 = ac1.number_input("Left X1 (px)", min_value=0, max_value=rw, value=def_x1, key=f"nx1_{current_stem}")
                new_y1 = ac2.number_input("Top Y1 (px)", min_value=0, max_value=rh, value=def_y1, key=f"ny1_{current_stem}")
                new_x2 = ac3.number_input("Right X2 (px)", min_value=0, max_value=rw, value=def_x2, key=f"nx2_{current_stem}")
                new_y2 = ac4.number_input("Bottom Y2 (px)", min_value=0, max_value=rh, value=def_y2, key=f"ny2_{current_stem}")

                if st.button("➕ Add Person Box to Labels", type="primary", use_container_width=True):
                    if new_x2 > new_x1 and new_y2 > new_y1:
                        new_b_list = list(current_boxes)
                        new_b_list.append((int(new_x1), int(new_y1), int(new_x2), int(new_y2)))
                        st.session_state['studio_boxes'][current_img_path] = new_b_list
                        st.success(f"✅ Added person box! Total people now: {len(new_b_list)}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid box: Right X2 must be greater than Left X1, and Bottom Y2 must be greater than Top Y1.")

            with add_tab2:
                st.markdown("Tune AI detector settings specifically for this picture to filter out false positives:")
                rc1, rc2, rc3 = st.columns(3)
                tune_conf = rc1.slider("AI Confidence Threshold", 0.10, 0.90, 0.35, 0.05, key=f"tune_conf_{current_stem}")
                tune_face = rc2.checkbox("Enable Face Recovery", value=False, key=f"tune_face_{current_stem}", help="Uncheck if creating duplicates on seated students")
                tune_sahi = rc3.checkbox("Enable SAHI Tiling", value=False, key=f"tune_sahi_{current_stem}")

                if st.button("⚡ Re-Detect & Replace Labels for This Picture", type="primary", use_container_width=True):
                    with st.spinner("Re-running AI detector on current picture..."):
                        from auto_label_college_data import auto_detect_boxes
                        from src.detector import load_yolo
                        model_name = YOLO_MODELS[model_label]
                        y_model = load_yolo(model_name)
                        f_model = face_net if tune_face else None
                        new_det = auto_detect_boxes(raw_bgr, y_model, f_model, conf_thresh=tune_conf, use_sahi=tune_sahi)
                        st.session_state['studio_boxes'][current_img_path] = new_det
                        st.toast(f"🎯 Detected {len(new_det)} people with confidence {tune_conf:.2f}!")
                        st.rerun()

        # ─────────────────────────────────────────
        # STEP 4: BATCH AUTO-LABELER FOR REMAINING IMAGES
        # ─────────────────────────────────────────
        st.markdown("---")
        with st.expander(f"⚡ Batch Auto-Label & Save All Images in One Click ({total_images} Total)", expanded=False):
            st.markdown("Automatically run AI auto-labeling and save annotations across multiple images in your dataset:")
            b_col1, b_col2 = st.columns([2, 1])
            with b_col1:
                batch_count = st.number_input(
                    "Number of images to batch process:",
                    min_value=1,
                    max_value=max(1, total_images),
                    value=min(total_images, 50)
                )
            with b_col2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                run_batch = st.button("🚀 Start Batch Auto-Labeling", type="primary", use_container_width=True)

            if run_batch and img_files:
                from auto_label_college_data import auto_detect_boxes
                from src.detector import load_yolo
                model_name = YOLO_MODELS[model_label]
                y_model = load_yolo(model_name)
                f_model = face_net if use_face_lbl else None

                b_prog = st.progress(0)
                b_status = st.empty()
                b_preview = st.empty()

                items_to_run = img_files[:int(batch_count)]
                total_batch_people = 0

                for i, p in enumerate(items_to_run, 1):
                    fr = cv2.imread(p)
                    if fr is None:
                        continue
                    bx = auto_detect_boxes(fr, y_model, f_model, conf_thresh=auto_conf, use_sahi=use_sahi_lbl)
                    total_batch_people += len(bx)
                    st.session_state['studio_boxes'][p] = bx
                    is_val_sp = (i % 5 == 0)
                    save_image_and_labels_to_disk(p, bx, target_dataset_dir, is_val=is_val_sp)
                    
                    b_prog.progress(i / len(items_to_run))
                    b_status.markdown(f"**Processing [{i}/{len(items_to_run)}]:** `{os.path.basename(p)}` — Found **{len(bx)} students**")
                    if i % 5 == 0 or i == len(items_to_run):
                        vis_b = render_boxes_on_bgr(fr, bx)
                        show_image(b_preview, bgr_to_rgb(vis_b), caption=f"Last processed: {os.path.basename(p)}")

                st.success(f"🎉 Successfully auto-labeled and saved **{len(items_to_run)} images** ({total_batch_people:,} student bounding boxes) into `{target_dataset_dir}`!")

        # ─────────────────────────────────────────
        # STEP 5: TARGET TRAINING DATASET HEALTH & SUMMARY
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 4. Target Training Dataset Summary & Health")
        
        train_lbl_dir = os.path.join(target_dataset_dir, "labels", "train")
        val_lbl_dir = os.path.join(target_dataset_dir, "labels", "val")
        saved_train = len([f for f in os.listdir(train_lbl_dir) if f.endswith(".txt")]) if os.path.exists(train_lbl_dir) else 0
        saved_val = len([f for f in os.listdir(val_lbl_dir) if f.endswith(".txt")]) if os.path.exists(val_lbl_dir) else 0
        total_saved_imgs = saved_train + saved_val

        total_boxes_in_dataset = 0
        for d in [train_lbl_dir, val_lbl_dir]:
            if os.path.exists(d):
                for lf in os.listdir(d):
                    if lf.endswith(".txt"):
                        with open(os.path.join(d, lf), "r", encoding="utf-8") as tf:
                            total_boxes_in_dataset += len([line for line in tf if line.strip()])

        yaml_exists = os.path.exists(os.path.join(target_dataset_dir, "college_data.yaml"))

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("💾 Total Saved Images", f"{total_saved_imgs:,} images")
        d_col2.metric("🚂 Train / Val Split", f"{saved_train} train / {saved_val} val")
        d_col3.metric("🎯 Total Student Bounding Boxes", f"{total_boxes_in_dataset:,} boxes")
        d_col4.metric("📄 Dataset YAML Config", "college_data.yaml ✅" if yaml_exists else "Missing")

        # ─────────────────────────────────────────
        # STEP 6: 1-CLICK MODEL TRAINING ON GPU
        # ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🏋️ 5. Train Custom Model on Saved Dataset")
        st.markdown("Fine-tune a custom YOLOv8 model directly on your newly labeled dataset using your GPU/CPU.")

        tc1, tc2, tc3 = st.columns(3)
        train_epochs = tc1.slider("Training Epochs", 10, 100, 30, 5, help="30-40 epochs is ideal for custom fine-tuning.")
        train_batch = tc2.selectbox("Batch Size", [2, 4, 8, 16], index=1)
        train_base = tc3.selectbox("Base YOLO Architecture", ["yolov8l.pt (Large - Best Accuracy)", "yolov8m.pt (Medium)", "yolov8n.pt (Nano - Fastest)"], index=0)

        if st.button("🚀 Start Training Custom YOLOv8 Model Now", type="primary", use_container_width=True):
            yaml_file = os.path.join(target_dataset_dir, "college_data.yaml")
            if total_saved_imgs == 0 or not os.path.exists(yaml_file):
                st.error("❌ No saved images in target dataset yet. Please label and click 'Save to Dataset' for some images first.")
            else:
                with st.spinner("🏋️ Fine-tuning YOLOv8 model on custom dataset... Please monitor progress..."):
                    try:
                        from ultralytics import YOLO
                        base_m_name = train_base.split()[0]
                        t_model = YOLO(base_m_name)
                        res = t_model.train(
                            data=yaml_file,
                            epochs=int(train_epochs),
                            imgsz=1280,
                            batch=int(train_batch),
                            device=DEVICE,
                            name="college_custom_model",
                            save=True
                        )
                        best_w = os.path.join("runs", "detect", "college_custom_model", "weights", "best.pt")
                        custom_dest = os.path.join(MODELS_DIR, "yolov8l_college_custom.pt")
                        if os.path.exists(best_w):
                            shutil.copy2(best_w, custom_dest)
                            st.success(f"🎉 Model training complete! Fine-tuned weights saved to: `{custom_dest}`")
                        else:
                            st.success("🎉 Training complete! Check `runs/detect/college_custom_model/` for weights.")
                    except Exception as te:
                        st.error(f"Training error: {te}")
