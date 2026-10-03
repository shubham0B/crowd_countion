import os, sys, time, io
# Prevent Streamlit inspection error with torch.classes on Windows
try:
    import torch
    if hasattr(torch, "classes"):
        torch.classes.__path__ = []
except Exception:
    pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    BASE_DIR, MODELS_DIR, YOLO_MODELS, DEFAULT_MODEL,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    INFERENCE_RESOLUTIONS, DEFAULT_RESOLUTION, DEVICE
)
from src.detector import process_image, _sahi_available, PERSON_CLASS_ID
from src.face_recovery import load_face_net
from src.utils import pil_to_bgr, bgr_to_rgb
from src.reporter import save_report_entry, load_all_reports, generate_html_report
from src.studio_tab import render_studio_tab

# ─────────────────────────────────────────
# VERSION-SAFE IMAGE DISPLAY
# Works on both old and new Streamlit versions
# ─────────────────────────────────────────
import inspect as _inspect
_IMG_PARAMS = set(_inspect.signature(st.image).parameters.keys())
def show_image(container, img, **kwargs):
    caption = kwargs.get("caption", None)
    if "use_container_width" in _IMG_PARAMS:
        container.image(img, use_container_width=True, caption=caption)
    elif "use_column_width" in _IMG_PARAMS:
        container.image(img, use_column_width=True, caption=caption)
    else:
        container.image(img, caption=caption)

st.set_page_config(
    page_title="Crowd Counter — VisionAI",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
div[data-testid="stMetric"] {
    background: rgba(15,20,35,0.6);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px 20px; border-radius: 14px;
}
div[data-testid="stMetricValue"] > div {
    font-size: 2rem !important; font-weight: 700 !important;
    background: linear-gradient(135deg,#00ff99,#00d2ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.density-warning {
    background: rgba(255,165,0,0.15); border-left: 3px solid #ffa500;
    padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #ffd580;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading face detection model...")
def get_face_net():
    return load_face_net()

face_net = get_face_net()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 VisionAI Crowd Counter")
    st.markdown(f"**Device:** `{DEVICE.upper()}`")
    st.divider()

    st.markdown("### 🎯 Detection Mode")
    detection_mode = st.radio(
        "Choose mode based on crowd size",
        options=["🔍 YOLO (0–150 people)", "🌊 Dense Crowd (150–2000+ people)"],
        help="YOLO: Draws boxes. Dense Crowd: Ultra-fine SAHI tiling for large crowds."
    )
    use_dense = detection_mode.startswith("🌊")

    if use_dense:
        st.info("🌊 Uses ultra-fine 320px SAHI tiles. No extra downloads needed.")
        st.divider()

    st.markdown("### 🧠 Detection Model")
    model_label = st.selectbox("YOLOv8 Architecture",
        options=list(YOLO_MODELS.keys()),
        index=list(YOLO_MODELS.keys()).index(DEFAULT_MODEL))

    st.markdown("### 🖼️ Resolution")
    resolution_label = st.selectbox("Inference Resolution",
        options=list(INFERENCE_RESOLUTIONS.keys()),
        index=list(INFERENCE_RESOLUTIONS.keys()).index(DEFAULT_RESOLUTION))
    imgsz = INFERENCE_RESOLUTIONS[resolution_label]

    st.markdown("### 🎛️ Detection Tuning")
    conf_threshold = st.slider("Confidence Threshold",
        0.05, 0.95, DEFAULT_CONF_THRESHOLD, 0.05)
    iou_threshold = st.slider("NMS IoU Threshold",
        0.10, 0.90, DEFAULT_IOU_THRESHOLD, 0.05)

    st.markdown("### 🚀 Crowd Enhancement")
    use_sahi = st.checkbox("Enable SAHI (Tiled Inference)", value=True)
    if use_sahi and not _sahi_available:
        st.error("SAHI not installed: pip install sahi")
        use_sahi = False

    use_face_recovery = st.checkbox("Enable Face Recovery", value=True)
    face_sensitivity = 9.0
    if use_face_recovery:
        face_sensitivity = st.slider("Face AI Sensitivity",
            1.0, 10.0, 9.0, 1.0)

    st.markdown("### 👁️ Visualization")
    vis_bodies_only = st.checkbox("Show Body Boxes Only", value=False)
    st.divider()
    st.markdown("🟢 Green = YOLO body\n\n🟠 Orange = Face recovery")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
st.title("👥 VisionAI — Crowd Counting System")
st.markdown("Upload an image to detect and count people using YOLOv8 + Face Recovery.")

tab_main, tab_video, tab_webcam, tab_labeler, tab_reports, tab_compare, tab_guide = st.tabs([
    "📸 Image Analytics",
    "🎬 Video File",
    "📷 Live Webcam",
    "🏷️ Live AI Auto-Labeler & Studio",
    "📑 Crowd Intelligence & Reports Dashboard",
    "⚖️ Model Comparison",
    "📚 System Guide"
])

# ══════════════════════════════════════════
# TAB 1 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════
with tab_main:
    uploaded_file = st.file_uploader(
        "Upload an image (JPG, PNG, JPEG)",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        frame     = pil_to_bgr(pil_image)

        col_orig, col_info = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            show_image(st, pil_image)

        if use_dense:
            with st.spinner("🌊 Running Dense Crowd mode..."):
                import time as _t
                _t0 = _t.time()
                from src.detector import load_yolo
                from src.utils import apply_clahe
                _model_name  = YOLO_MODELS[model_label]
                _dense_model = load_yolo(_model_name)
                _dframe      = apply_clahe(frame.copy())
                _fh, _fw     = _dframe.shape[:2]
                _dense_count = 0
                result_frame = _dframe.copy()

                if _sahi_available:
                    from sahi import AutoDetectionModel
                    from sahi.predict import get_sliced_prediction
                    _sm = AutoDetectionModel.from_pretrained(
                        model_type='yolov8', model_path=_model_name,
                        confidence_threshold=0.18, device=DEVICE)
                    _res = get_sliced_prediction(
                        cv2.cvtColor(_dframe, cv2.COLOR_BGR2RGB), _sm,
                        slice_height=320, slice_width=320,
                        overlap_height_ratio=0.40, overlap_width_ratio=0.40)
                    _dense_count = sum(1 for o in _res.object_prediction_list
                                       if o.category.id == PERSON_CLASS_ID)
                    _heat = np.zeros((_fh, _fw), dtype=np.float32)
                    for _obj in _res.object_prediction_list:
                        if _obj.category.id != PERSON_CLASS_ID:
                            continue
                        _x1,_y1,_x2,_y2 = map(int,_obj.bbox.to_xyxy())
                        _cx,_cy = (_x1+_x2)//2, (_y1+_y2)//2
                        cv2.circle(_heat, (_cx,_cy), 25, 3.0, -1)
                    _norm = cv2.normalize(_heat, None, 0, 255,
                                          cv2.NORM_MINMAX).astype(np.uint8)
                    _colored = cv2.applyColorMap(_norm, cv2.COLORMAP_JET)
                    result_frame = cv2.addWeighted(_dframe, 0.45, _colored, 0.55, 0)
                else:
                    _res2 = _dense_model(_dframe, conf=0.18, iou=0.40,
                                         classes=[PERSON_CLASS_ID], imgsz=1280, verbose=False)
                    _dense_count = sum(1 for r in _res2 for b in r.boxes
                                       if int(b.cls[0]) == PERSON_CLASS_ID)
                _t1 = _t.time()

            st.markdown("---")
            st.markdown("### 📊 Dense Crowd Results")
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Estimated Count", f"~{_dense_count:,}")
            dc2.metric("Mode", "Dense SAHI (320px)")
            dc3.metric("Time", f"{_t1-_t0:.1f}s")
            st.markdown("**Density Heatmap** — 🔴 Red = packed   🔵 Blue = sparse")
            show_image(st, bgr_to_rgb(result_frame),
                       caption=f"~{_dense_count:,} people estimated")
            _dl = io.BytesIO()
            Image.fromarray(bgr_to_rgb(result_frame)).save(_dl, format="PNG")
            _, _dcol, _ = st.columns([2,1,2])
            with _dcol:
                st.download_button("⬇️ Download Heatmap", data=_dl.getvalue(),
                    file_name=f"dense_{_dense_count}_people.png", mime="image/png")

        else:
            with st.spinner("🔍 Running AI detection pipeline..."):
                result = process_image(
                    frame=frame, model_label=model_label,
                    conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                    imgsz=imgsz, use_sahi=use_sahi,
                    use_face_recovery=use_face_recovery,
                    face_sensitivity=face_sensitivity,
                    face_net=face_net if use_face_recovery else None,
                    vis_bodies_only=vis_bodies_only)

            if result.crowd_density == "High" and not use_sahi:
                st.markdown('<div class="density-warning">⚠️ High density crowd! '
                    'Enable SAHI or switch to Dense Crowd mode.</div>',
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Detection Results")
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total People", result.total_count)
            c2.metric("YOLO Detections", result.yolo_count)
            c3.metric("Face Recovered", result.face_count)
            c4.metric("Avg Confidence", f"{result.avg_confidence:.1f}%")
            c5.metric("Process Time", f"{result.proc_time_sec:.2f}s")

            cs1,cs2,cs3 = st.columns(3)
            cs1.metric("Raw YOLO", result.raw_yolo_count,
                       delta=f"-{result.rejected_count} filtered")
            cs2.metric("Crowd Density", result.crowd_density)
            cs3.metric("Total Rejected", result.rejected_count)

            with st.expander("🔍 Why Were Detections Rejected?"):
                rej = result.rejection_stats.to_dict()
                rcols = st.columns(5)
                for col,(label,count) in zip(rcols, rej.items()):
                    col.metric(label, count)

            st.markdown("---")
            st.markdown("**Detection Result** — 🖱️ Scroll to zoom · drag to pan · double-click reset")
            import base64
            _buf = io.BytesIO()
            Image.fromarray(bgr_to_rgb(result.annotated_frame)).save(_buf, format="PNG")
            _b64 = base64.b64encode(_buf.getvalue()).decode()
            _zh = f"""
            <style>
              #zw{{width:100%;height:540px;overflow:hidden;background:#0e1117;
                border-radius:10px;border:1px solid #333;cursor:grab;position:relative;}}
              #zw:active{{cursor:grabbing;}}
              #zi{{position:absolute;transform-origin:0 0;max-width:none;user-select:none;}}
              #zinfo{{position:absolute;bottom:10px;right:14px;color:#aaa;font-size:12px;
                font-family:monospace;background:rgba(0,0,0,0.5);padding:3px 8px;border-radius:4px;}}
              #zreset{{position:absolute;top:10px;right:12px;background:rgba(30,30,30,0.85);
                color:#ccc;border:1px solid #555;border-radius:5px;padding:4px 10px;
                font-size:12px;cursor:pointer;z-index:10;}}
            </style>
            <div id="zw">
              <img id="zi" src="data:image/png;base64,{_b64}"/>
              <button id="zreset" onclick="zReset()">⟳ Reset</button>
              <div id="zinfo">100%</div>
            </div>
            <script>
              const zw=document.getElementById('zw'),zi=document.getElementById('zi'),
                    zinf=document.getElementById('zinfo');
              let s=1,tx=0,ty=0,drag=false,sx,sy,stx,sty;
              function fit(){{const wr=zw.clientWidth/zi.naturalWidth,
                hr=zw.clientHeight/zi.naturalHeight;s=Math.min(wr,hr);
                tx=(zw.clientWidth-zi.naturalWidth*s)/2;
                ty=(zw.clientHeight-zi.naturalHeight*s)/2;apply();}}
              function apply(){{zi.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{s}})`;
                zinf.textContent=Math.round(s*100)+'%';}}
              function zReset(){{fit();}}
              zi.onload=fit;
              zw.addEventListener('wheel',e=>{{e.preventDefault();
                const r=zw.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
                const d=e.deltaY>0?0.85:1.18,ns=Math.max(0.3,Math.min(12,s*d));
                tx=mx-(mx-tx)*(ns/s);ty=my-(my-ty)*(ns/s);s=ns;apply();}},{{passive:false}});
              zw.addEventListener('mousedown',e=>{{drag=true;sx=e.clientX;sy=e.clientY;stx=tx;sty=ty;}});
              document.addEventListener('mousemove',e=>{{if(!drag)return;tx=stx+(e.clientX-sx);ty=sty+(e.clientY-sy);apply();}});
              document.addEventListener('mouseup',()=>{{drag=false;}});
              zw.addEventListener('dblclick',zReset);
            </script>"""
            import streamlit.components.v1 as _comp
            _comp.html(_zh, height=560)
            st.caption(f"🟢 Green = YOLO body   🟠 Orange = Face recovery   |   Total: **{result.total_count} people**")
            _dl2 = io.BytesIO()
            Image.fromarray(bgr_to_rgb(result.annotated_frame)).save(_dl2, format="PNG")
            _, dl_col, _ = st.columns([2,1,2])
            with dl_col:
                st.download_button("⬇️ Download Result", data=_dl2.getvalue(),
                    file_name=f"crowd_{result.total_count}_people.png", mime="image/png")
    else:
        st.info("⬆️ Upload an image above to begin counting people.")

# ══════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════
with tab_video:
    st.markdown("### 🎬 Video File — People Counting")
    st.markdown("Upload a video — frames play directly on this page with live count.")
    vid_file = st.file_uploader("Upload Video", type=["mp4","avi","mov","mkv"])
    vc1, vc2, vc3, vc4 = st.columns(4)
    vid_conf  = vc1.slider("Confidence", 0.10, 0.80, 0.25, 0.05, key="vid_conf")
    vid_skip  = vc2.selectbox("Frame Skip", [1,2,3,4], index=1, key="vid_skip")
    vid_model = vc3.selectbox("Model", list(YOLO_MODELS.keys()), index=0, key="vid_model")
    vid_imgsz = vc4.selectbox("Resolution", [416,640,960], index=1, key="vid_imgsz")

    if vid_file:
        if st.button("▶ Start Processing Video", type="primary"):
            import tempfile, time as _vt
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name
            st.markdown("---")
            vid_frame_ph = st.empty()
            vcols = st.columns(3)
            vcount_ph, vfps_ph, vframe_ph = vcols[0].empty(), vcols[1].empty(), vcols[2].empty()
            progress_bar = st.progress(0)
            vstatus = st.empty()
            try:
                from src.detector import load_yolo
                from src.live_processor import detect_frame
                _vm = load_yolo(YOLO_MODELS[vid_model])
                cap = cv2.VideoCapture(tmp_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
                _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                import tempfile as _tf2
                out_tmp = _tf2.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = out_tmp.name; out_tmp.close()
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps_in / int(vid_skip), (_w, _h))
                import pandas as pd
                frame_history = []
                counts_list = []
                timestamps_list = []
                idx = 0; t0 = _vt.time()

                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    idx += 1
                    if idx % int(vid_skip) != 0: continue
                    out_frame, count = detect_frame(frame, _vm, conf=vid_conf, imgsz=int(vid_imgsz))
                    writer.write(out_frame)
                    elapsed = _vt.time() - t0
                    current_time_sec = idx / fps_in
                    fps_out = (idx // int(vid_skip)) / max(elapsed, 0.001)

                    # Track analytics
                    counts_list.append(count)
                    timestamps_list.append(current_time_sec)
                    frame_history.append({
                        "Frame": idx,
                        "Time (s)": round(current_time_sec, 2),
                        "People Count": count,
                        "FPS": round(fps_out, 1)
                    })

                    # Live metrics
                    current_avg = sum(counts_list) / len(counts_list)
                    show_image(vid_frame_ph, bgr_to_rgb(out_frame),
                               caption=f"Frame {idx}/{total} | People in Frame: {count} | Avg: {current_avg:.1f}")
                    vcount_ph.metric("👥 People in Frame", count)
                    vfps_ph.metric("⚡ FPS", f"{fps_out:.1f}")
                    vframe_ph.metric("🎞️ Frame", f"{idx}/{total}")
                    progress_bar.progress(min(idx / max(total, 1), 1.0))

                cap.release(); writer.release()

                # ─────────────────────────────────────────
                # POST-VIDEO ANALYTICS DASHBOARD
                # ─────────────────────────────────────────
                if counts_list:
                    avg_count = sum(counts_list) / len(counts_list)
                    max_count = max(counts_list)
                    min_count = min(counts_list)
                    max_idx = counts_list.index(max_count)
                    peak_time = timestamps_list[max_idx]
                    total_time_str = f"{int(total / fps_in // 60):02d}:{int(total / fps_in % 60):02d}"
                    peak_time_str = f"{int(peak_time // 60):02d}:{int(peak_time % 60):02d}"

                    st.markdown("---")
                    st.markdown("### 📊 Post-Video Crowd Analytics Summary")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("📈 Average Crowd", f"{avg_count:.1f} people", help="Average number of people per frame throughout the entire video")
                    m2.metric("🚀 Peak Crowd", f"{max_count} people", f"at {peak_time_str}", help="Maximum crowd density observed in a single frame")
                    m3.metric("📉 Minimum Crowd", f"{min_count} people", help="Lowest crowd count observed")
                    m4.metric("⏱️ Total Duration", total_time_str, f"{len(counts_list)} frames")

                    # Time Series Chart
                    st.markdown("#### 📉 Crowd Fluctuation Over Time")
                    df_chart = pd.DataFrame({
                        "Time (seconds)": [round(t, 1) for t in timestamps_list],
                        "People in Frame": counts_list,
                        "Video Average": [round(avg_count, 1)] * len(counts_list)
                    }).set_index("Time (seconds)")
                    st.line_chart(df_chart, color=["#00ff99", "#ff9900"])

                    # Save to central report system
                    rep_entry = save_report_entry("Video", getattr(vid_file, "name", "Video"), int(sum(counts_list)), avg_count, max_count, min_count, total / fps_in, frame_history)
                    st.session_state['latest_report'] = rep_entry
                    st.session_state['latest_chart_df'] = df_chart

                    # Download Options
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        with open(out_path, "rb") as f:
                            st.download_button("⬇️ Download Annotated Video", data=f.read(),
                                file_name="crowd_annotated_video.mp4", mime="video/mp4", use_container_width=True)
                    with col_dl2:
                        csv_data = pd.DataFrame(frame_history).to_csv(index=False).encode('utf-8')
                        st.download_button("📊 Download Analytics CSV Report", data=csv_data,
                            file_name="crowd_analytics_report.csv", mime="text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Video error: {e}")
            finally:
                try: os.unlink(tmp_path)
                except Exception: pass
    else:
        st.info("⬆️ Upload a video file to begin.")

# ══════════════════════════════════════════
# TAB 3 — WEBCAM
# ══════════════════════════════════════════
with tab_webcam:
    st.markdown("### 📷 Live Webcam — Real-Time People Counting")
    wc1, wc2, wc3, wc4 = st.columns(4)
    cam_model = wc1.selectbox("Model",
        ["Nano  — Fastest  (0–30 people)",
         "Small — Fast     (0–60 people)",
         "Medium — Balanced (30–100 people)"], index=0, key="cam_model")
    cam_conf  = wc2.slider("Confidence", 0.15, 0.70, 0.30, 0.05, key="cam_conf")
    cam_src   = wc3.text_input("Camera Source", value="0", key="cam_src")
    cam_frames = wc4.number_input("Run for N frames", min_value=30,
                                   max_value=3000, value=300, step=30)
    st.markdown("---")
    cam_frame_ph = st.empty()
    cm = st.columns(3)
    cam_count_ph, cam_fps_ph, cam_fr_ph = cm[0].empty(), cm[1].empty(), cm[2].empty()
    cstart, cinfo = st.columns([1,3])
    start_cam = cstart.button("▶ Start Webcam", type="primary", key="start_cam2")
    cinfo.markdown("Camera captures **N frames** then auto-stops. Click **Start** again to continue.")

    if start_cam:
        import time as _ct
        src2 = int(cam_src) if cam_src.strip().isdigit() else cam_src.strip()
        try:
            from src.detector import load_yolo
            from src.live_processor import detect_frame
            _map = {
                "Nano  — Fastest  (0–30 people)"    : YOLO_MODELS["Nano  — Fastest  (0–30 people)"],
                "Small — Fast     (0–60 people)"    : YOLO_MODELS["Small — Fast     (0–60 people)"],
                "Medium — Balanced (30–100 people)" : YOLO_MODELS["Medium — Balanced (30–100 people)"],
            }
            _cm = load_yolo(_map.get(cam_model))
            cap2 = cv2.VideoCapture(src2)
            if not cap2.isOpened():
                st.error(f"Cannot open camera: {src2}")
            else:
                cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                t0 = _ct.time(); frames = 0; N = int(cam_frames)
                cam_frame_ph.info(f"📷 Camera started — running {N} frames...")
                while frames < N:
                    ret2, frame2 = cap2.read()
                    if not ret2: break
                    cap2.read()
                    out2, count2 = detect_frame(frame2, _cm, conf=cam_conf, iou=0.45, imgsz=416)
                    frames += 1
                    elapsed2 = _ct.time() - t0
                    fps2 = frames / max(elapsed2, 0.001)
                    show_image(cam_frame_ph, bgr_to_rgb(out2),
                               caption=f"People: {count2} | FPS: {fps2:.1f}")
                    cam_count_ph.metric("👥 People", count2)
                    cam_fps_ph.metric("⚡ FPS", f"{fps2:.1f}")
                    cam_fr_ph.metric("🎞 Frame", f"{frames}/{N}")
                cap2.release()
                cam_frame_ph.success("✅ Capture finished.")
        except Exception as e:
            st.error(f"Webcam error: {e}")

# ══════════════════════════════════════════
# TAB 4 — LIVE AI AUTO-LABELER & STUDIO
# ══════════════════════════════════════════
render_studio_tab(tab_labeler, model_label, face_net)

# ══════════════════════════════════════════
# TAB 5 — CROWD INTELLIGENCE & REPORTS DASHBOARD
# ══════════════════════════════════════════
with tab_reports:
    st.markdown("### 📑 Crowd Intelligence & Project Executive Reports")
    st.markdown("Automated analytics dashboard summarizing **crowd occupancy, timeline fluctuations, peak congestions, and project reports**.")

    all_reports = load_all_reports()
    
    if not all_reports and 'latest_report' not in st.session_state:
        st.info("💡 No analytics recorded yet. Process an image in **📸 Image Analytics**, a video in **🎬 Video File**, or run **🏷️ Live AI Auto-Labeler** to generate reports.")
    else:
        latest = st.session_state.get('latest_report', all_reports[0] if all_reports else {})
        
        # 1. Global KPIs
        total_people_seen = sum(r.get('total_count', 0) for r in all_reports) if all_reports else latest.get('total_count', 0)
        avg_people_overall = sum(r.get('avg_count', 0) for r in all_reports) / max(len(all_reports), 1) if all_reports else latest.get('avg_count', 0)
        global_peak = max(r.get('peak_count', 0) for r in all_reports) if all_reports else latest.get('peak_count', 0)
        total_sessions = len(all_reports)

        rk1, rk2, rk3, rk4 = st.columns(4)
        rk1.metric("👥 Total People Detected", f"{total_people_seen:,}", help="Total count of people across all analyzed frames")
        rk2.metric("📈 Average Crowd / Frame", f"{avg_people_overall:.1f} people", help="Average density per frame")
        rk3.metric("🚀 Max Density Peak", f"{global_peak} people", help="Highest crowd density spike recorded")
        rk4.metric("📁 Media Sessions", f"{total_sessions} runs", help="Total videos/datasets analyzed")

        st.markdown("---")

        # 2. Executive Report Generation
        st.markdown("#### 📄 Downloadable Executive Project Report")
        col_rep1, col_rep2 = st.columns([2, 1])
        with col_rep1:
            st.markdown(f"""
            **Latest Analysis Report:** `{latest.get('filename', 'Session')}`  
            - **Timestamp:** {latest.get('timestamp', 'N/A')}  
            - **Average Occupancy:** **{latest.get('avg_count', 0)} people / frame**  
            - **Peak Congestion:** **{latest.get('peak_count', 0)} people**
            """)
        with col_rep2:
            html_report_code = generate_html_report(latest, all_reports)
            st.download_button(
                "⬇️ Download Official Executive Report (HTML/PDF)",
                data=html_report_code,
                file_name=f"VisionAI_Crowd_Report_{int(time.time())}.html",
                mime="text/html",
                use_container_width=True
            )

        # 3. Interactive Chart
        if 'latest_chart_df' in st.session_state:
            st.markdown("---")
            st.markdown("#### 📉 Recent Video Crowd Fluctuation Curve")
            st.line_chart(st.session_state['latest_chart_df'], color=["#00ff99", "#ffaa00"])

        # 4. Historical Logs Data Table
        st.markdown("---")
        st.markdown("#### 🕒 Full Detection & Analytics History Table")
        if all_reports:
            df_hist = pd.DataFrame(all_reports)
            st.dataframe(df_hist, use_container_width=True)
            
            csv_hist = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button("📊 Download Complete History (CSV)", data=csv_hist,
                file_name="crowd_all_sessions_history.csv", mime="text/csv")

# ══════════════════════════════════════════
# TAB 6 — COMPARISON
# ══════════════════════════════════════════
with tab_compare:
    st.markdown("### ⚖️ Compare Two YOLOv8 Models")
    if 'pil_image' not in dir() or uploaded_file is None:
        st.info("Upload an image in the Image Analytics tab first.")
    else:
        ca, cb = st.columns(2)
        mk = list(YOLO_MODELS.keys())
        with ca: model_a = st.selectbox("Model A", mk, index=0)
        with cb: model_b = st.selectbox("Model B", mk, index=3)
        if st.button("▶ Run Comparison"):
            with st.spinner("Running both models..."):
                ra = process_image(pil_to_bgr(pil_image), model_a, conf_threshold,
                    iou_threshold, 640, False, use_face_recovery, face_sensitivity, face_net, vis_bodies_only)
                rb = process_image(pil_to_bgr(pil_image), model_b, conf_threshold,
                    iou_threshold, 1280, False, use_face_recovery, face_sensitivity, face_net, vis_bodies_only)
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**{model_a.split('—')[0].strip()}** (640p)")
                st.metric("Count", ra.total_count)
                st.metric("Time", f"{ra.proc_time_sec:.2f}s")
                show_image(st, bgr_to_rgb(ra.annotated_frame))
            with colB:
                st.markdown(f"**{model_b.split('—')[0].strip()}** (1280p)")
                st.metric("Count", rb.total_count)
                st.metric("Time", f"{rb.proc_time_sec:.2f}s")
                show_image(st, bgr_to_rgb(rb.annotated_frame))

# ══════════════════════════════════════════
# TAB 7 — GUIDE
# ══════════════════════════════════════════
with tab_guide:
    st.markdown("""
### 📚 How the System Works

#### Detection Modes
| Mode | Use When | How |
|---|---|---|
| 🔍 YOLO | 0–150 people | Draws boxes per person |
| 🌊 Dense Crowd | 150–2000+ people | 320px SAHI tiles + heatmap |

#### Recommended Settings
| Crowd Size | Model | Conf | SAHI | Face |
|---|---|---|---|---|
| 0–30 | Nano | 0.40 | OFF | OFF |
| 30–100 | Medium | 0.35 | ON | ON |
| 100–150 | Large | 0.30 | ON | ON |
| 150+ | Dense Crowd mode | – | – | – |

#### Box Colors
- 🟢 **Green** — YOLO body detection
- 🟠 **Orange** — Face recovery (body occluded)
    """)
