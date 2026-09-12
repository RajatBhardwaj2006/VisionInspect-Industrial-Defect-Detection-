import os
import sys
import io
import time
import base64
import hashlib
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any

import streamlit as st
import requests

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_category_config
from src.detection.category_checker import CategoryCompatibilityChecker

st.set_page_config(
    page_title="VisionInspect — Industrial Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.0rem;
    }
    .model-change-alert {
        background-color: #EFF6FF;
        border-left: 6px solid #3B82F6;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #1E40AF;
        margin-bottom: 1.0rem;
    }
    .status-card-defective {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .status-card-normal {
        background-color: #DCFCE7;
        border-left: 6px solid #22C55E;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .status-card-error {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .mismatch-warning-box {
        background-color: #FFFBEB;
        border: 2px solid #F59E0B;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        color: #92400E;
        margin: 0.8rem 0;
    }
    .summary-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "uploaded_files_data" not in st.session_state:
    st.session_state["uploaded_files_data"] = [] # list of (filename, bytes, hash)
if "batch_files_hash" not in st.session_state:
    st.session_state["batch_files_hash"] = ""
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = "bottle"
if "previous_category" not in st.session_state:
    st.session_state["previous_category"] = "bottle"
if "category_changed_notice" not in st.session_state:
    st.session_state["category_changed_notice"] = None
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("⚙️ Inspection Setup")

categories = ["bottle", "leather", "transistor", "zipper", "screw"]
category_labels = [c.capitalize() for c in categories]

current_cat_idx = categories.index(st.session_state["selected_category"]) if st.session_state["selected_category"] in categories else 0
chosen_label = st.sidebar.selectbox("Select Product Category", category_labels, index=current_cat_idx)
new_category = chosen_label.lower()

# Model change detection
if new_category != st.session_state["selected_category"]:
    prev = st.session_state["selected_category"]
    st.session_state["category_changed_notice"] = (
        f"MODEL CHANGED: {prev.capitalize()} → {new_category.capitalize()}.\n"
        f"The inspection will now use PatchCore v2.3 — {new_category.capitalize()} "
        f"with the {new_category.capitalize()} memory bank and calibrated thresholds."
    )
    st.session_state["previous_category"] = prev
    st.session_state["selected_category"] = new_category
    st.session_state["batch_results"] = None # Clear old results!
    st.rerun()

cfg = load_config()
cat_cfg = get_category_config(new_category, cfg)

st.sidebar.markdown(f"### Selected Model: **PatchCore v2.3 — {new_category.capitalize()}**")
st.sidebar.markdown(f"- **Category:** `{new_category}`")
st.sidebar.markdown(f"- **Architecture:** PatchCore v2.3 (ResNet18 64x64 Grid)")
st.sidebar.markdown(f"- **Memory Bank:** `models/{new_category}/patchcore_v23/`")
st.sidebar.markdown(f"- **Image Threshold:** `{cat_cfg.get('image_threshold', 1.50):.2f}`")
st.sidebar.markdown(f"- **Pixel Threshold:** `{cat_cfg.get('pixel_threshold', 1.40):.2f}`")
st.sidebar.markdown(f"- **Spatial Prior:** {'Enabled' if cat_cfg.get('use_spatial_prior', True) else 'Disabled (Texture Mode)'}")

backend_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")

backend_online = False
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=1.2)
    if health_resp.status_code == 200:
        backend_online = True
        st.sidebar.success("● FastAPI Backend Online")
    else:
        st.sidebar.warning("▲ Backend reachable with warning")
except Exception:
    st.sidebar.info("○ Backend offline (Direct fallback mode)")

st.sidebar.markdown("---")
st.sidebar.caption("VisionInspect Phase 3.2 — Batch Industrial Quality Assurance")

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">VisionInspect — Industrial Defect Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Active Model: <b>PatchCore v2.3 — {new_category.capitalize()}</b></div>', unsafe_allow_html=True)

# Display Model Changed Alert if triggered
if st.session_state["category_changed_notice"]:
    st.markdown(f"""
    <div class="model-change-alert">
        <h4 style="margin:0 0 5px 0;">🔄 MODEL CHANGED</h4>
        <p style="margin:0; font-size:0.95rem;">{st.session_state['category_changed_notice'].replace(chr(10), '<br>')}</p>
    </div>
    """, unsafe_allow_html=True)

# ----------------- SECTION 1: BATCH IMAGE UPLOAD (PRIMARY) -----------------
st.subheader("1. Upload Product Images (Batch 1–5)")

uploaded_files = st.file_uploader(
    "Drag & Drop 1–5 product images here, or click to browse files",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Upload between 1 and 5 product images for simultaneous independent inspection."
)

# Validate batch size
if not uploaded_files:
    st.session_state["uploaded_files_data"] = []
    st.session_state["batch_files_hash"] = ""
    st.session_state["batch_results"] = None
    st.info("👈 **Upload at least one image to begin inspection.** (Maximum 5 images)")
elif len(uploaded_files) > 5:
    st.error(f"⚠️ **Please upload a maximum of 5 images.** You uploaded {len(uploaded_files)} images. Please remove {len(uploaded_files) - 5} image(s).")
    st.session_state["uploaded_files_data"] = []
    st.session_state["batch_files_hash"] = ""
    st.session_state["batch_results"] = None
else:
    # 1 to 5 images uploaded: build batch state
    curr_batch = []
    hasher = hashlib.sha256()
    for f in uploaded_files:
        f_bytes = f.read()
        f_hash = hashlib.sha256(f_bytes).hexdigest()
        curr_batch.append((f.name, f_bytes, f_hash))
        hasher.update(f_bytes)
        
    combined_hash = hasher.hexdigest()
    if st.session_state["batch_files_hash"] != combined_hash:
        st.session_state["uploaded_files_data"] = curr_batch
        st.session_state["batch_files_hash"] = combined_hash
        st.session_state["batch_results"] = None  # Clear previous results!
        # Clear model change notice once user interacts with files
        st.session_state["category_changed_notice"] = None

# ----------------- SECTION 2: OPTIONAL SAMPLE TESTER -----------------
with st.expander("📁 Optional: Test with MVTec Dataset Samples", expanded=False):
    dataset_dir = Path("dataset/mvtec_anomaly_detection") / new_category / "test"
    if dataset_dir.exists():
        defect_folders = sorted([f.name for f in dataset_dir.iterdir() if f.is_dir()])
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s1:
            sample_defect = st.selectbox("Sample Defect Class", defect_folders, key="mvtec_defect_select")
        with col_s2:
            sample_files = sorted(list((dataset_dir / sample_defect).glob("*.png")))
            sample_names = [f.name for f in sample_files]
            chosen_sample_name = st.selectbox("Sample Image", sample_names, key="mvtec_file_select")
        with col_s3:
            st.write("")
            st.write("")
            if st.button("Load Dataset Sample into Batch"):
                sample_p = dataset_dir / sample_defect / chosen_sample_name
                with open(sample_p, "rb") as sf:
                    s_bytes = sf.read()
                st.session_state["uploaded_files_data"] = [(f"MVTec_{new_category}_{chosen_sample_name}", s_bytes, hashlib.sha256(s_bytes).hexdigest())]
                st.session_state["batch_files_hash"] = hashlib.sha256(s_bytes).hexdigest()
                st.session_state["batch_results"] = None
                st.session_state["category_changed_notice"] = None
                st.rerun()

st.markdown("---")

# ----------------- SECTION 3: BATCH EXECUTION -----------------
active_batch = st.session_state["uploaded_files_data"]

if len(active_batch) > 0 and len(active_batch) <= 5:
    st.subheader(f"2. Inspect Product Batch ({len(active_batch)} Image{'s' if len(active_batch) > 1 else ''})")
    
    # Thumbnails preview row
    thumb_cols = st.columns(min(5, len(active_batch)))
    for idx, (fname, fbytes, _) in enumerate(active_batch):
        with thumb_cols[idx]:
            try:
                pil_im = Image.open(io.BytesIO(fbytes))
                st.image(pil_im, caption=f"{idx+1}. {fname}", use_container_width=True)
            except Exception:
                st.warning(f"Corrupt: {fname}")
                
    button_label = f"🔍 INSPECT ALL ({len(active_batch)} IMAGES)" if len(active_batch) > 1 else "🔍 INSPECT IMAGE"
    if st.button(button_label, type="primary", use_container_width=True):
        st.session_state["category_changed_notice"] = None
        with st.spinner(f"Running PatchCore v2.3 ({new_category}) on {len(active_batch)} image(s)..."):
            batch_output = []
            
            # Attempt FastAPI batch endpoint first if available
            if backend_online:
                try:
                    files_payload = [
                        ("files", (fname, io.BytesIO(fbytes), "image/png"))
                        for (fname, fbytes, _) in active_batch
                    ]
                    data_payload = {"category": new_category, "return_visualizations": "true"}
                    r = requests.post(f"{backend_url}/predict/batch", files=files_payload, data=data_payload, timeout=60)
                    if r.status_code == 200:
                        batch_output = r.json().get("results", [])
                    else:
                        st.warning(f"Backend returned status {r.status_code}. Using local processing.")
                except Exception as be:
                    st.warning(f"Backend connection error ({be}). Using local processing.")
                    
            # Fallback to local processing if backend not used
            if not batch_output:
                from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
                detector = PatchCoreDetectorV23(category=new_category)
                checker = CategoryCompatibilityChecker()
                
                for fname, fbytes, _ in active_batch:
                    try:
                        pil_img = Image.open(io.BytesIO(fbytes)).convert("RGB")
                        t0 = time.perf_counter()
                        res = detector.inspect(pil_img)
                        latency = (time.perf_counter() - t0) * 1000.0
                        
                        comp_res = checker.check_compatibility(pil_img, new_category)
                        
                        defect_regions = []
                        for ridx, reg in enumerate(res.get("localized_regions", [])):
                            w = reg["width"]
                            h = reg["height"]
                            defect_regions.append({
                                "id": ridx + 1,
                                "bbox": [reg["x"], reg["y"], w, h],
                                "area": reg["area"],
                                "score": round(float(reg.get("score", 0.0)), 4),
                                "aspect_ratio": round(float(w / max(1, h)), 2)
                            })
                            
                        vis_b64 = None
                        if "saved_path" in res and Path(res["saved_path"]).exists():
                            with open(res["saved_path"], "rb") as sf:
                                vis_b64 = base64.b64encode(sf.read()).decode("utf-8")
                                
                        batch_output.append({
                            "filename": fname,
                            "category": new_category,
                            "model": "PatchCore v2.3",
                            "status": res["status"],
                            "is_defective": res["status"] == "DEFECTIVE",
                            "anomaly_score": round(float(res["score"]), 4),
                            "image_threshold": round(float(res["image_threshold"]), 4),
                            "pixel_threshold": round(float(res["pixel_threshold"]), 4),
                            "num_defects": len(defect_regions),
                            "localized_regions": defect_regions,
                            "defect_regions": defect_regions,
                            "visualization_base64": vis_b64,
                            "inference_time_ms": round(latency, 2),
                            "explanation": res["explanation"],
                            "compatibility": comp_res,
                            "category_warning": comp_res.get("warning_message")
                        })
                    except Exception as err:
                        batch_output.append({
                            "filename": fname,
                            "category": new_category,
                            "status": "ERROR",
                            "is_defective": False,
                            "error": str(err)
                        })
                        
            st.session_state["batch_results"] = batch_output
            st.rerun()

# ----------------- SECTION 4: DISPLAY BATCH RESULTS -----------------
results = st.session_state.get("batch_results")

if results is not None:
    st.markdown("---")
    st.subheader("3. Inspection Results")
    
    # Calculate Summary Statistics
    total_imgs = len(results)
    normal_count = sum(1 for r in results if r.get("status") == "NORMAL")
    defective_count = sum(1 for r in results if r.get("status") == "DEFECTIVE")
    failed_count = sum(1 for r in results if r.get("status") == "ERROR")
    warning_count = sum(1 for r in results if r.get("category_warning"))
    
    # Summary Bar
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("Total Images", f"{total_imgs}")
    with s2:
        st.metric("Normal", f"{normal_count}")
    with s3:
        st.metric("Defective", f"{defective_count}")
    with s4:
        st.metric("Category Warnings", f"{warning_count}")
    with s5:
        st.metric("Failed", f"{failed_count}")
        
    st.markdown("---")
    
    # Render Per-Image Cards
    for idx, r in enumerate(results):
        fname = r.get("filename", f"Image {idx+1}")
        status = r.get("status", "UNKNOWN")
        is_defective = r.get("is_defective", False)
        
        st.markdown(f"### Image {idx+1}: `{fname}`")
        
        # Handle Processing Errors
        if status == "ERROR":
            st.markdown(f"""
            <div class="status-card-error">
                <h4 style="color:#B45309; margin:0 0 4px 0;">⚠️ INSPECTION FAILED</h4>
                <p style="margin:0; color:#78350F;">Unable to decode or inspect image: {r.get('error', 'Unknown error')}</p>
            </div>
            """, unsafe_allow_html=True)
            continue
            
        # Category Compatibility Mismatch Warning Card
        cat_warning = r.get("category_warning")
        comp = r.get("compatibility", {})
        if cat_warning:
            suggested = comp.get("best_compatible_category", "").capitalize()
            st.markdown(f"""
            <div class="mismatch-warning-box">
                <h4 style="margin:0 0 5px 0;">⚠️ CATEGORY MISMATCH WARNING</h4>
                <p style="margin:0 0 8px 0; font-size:1.0rem;">
                    You selected: <b>{new_category.capitalize()}</b>.<br>
                    However, this image appears more compatible with: <b>{suggested}</b>.<br>
                    <i>The selected {new_category.capitalize()} model was used.</i>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            btn_col1, btn_col2, _ = st.columns([1, 1, 2])
            with btn_col1:
                if st.button(f"Switch to {suggested}", key=f"switch_{idx}_{suggested}"):
                    st.session_state["selected_category"] = suggested.lower()
                    st.session_state["category_changed_notice"] = f"Switched model to PatchCore v2.3 — {suggested}."
                    st.session_state["batch_results"] = None
                    st.rerun()
            with btn_col2:
                if st.button(f"Continue with {new_category.capitalize()}", key=f"continue_{idx}_{new_category}"):
                    st.info(f"Proceeding with {new_category.capitalize()} model.")
                    
        # Status Card
        if is_defective:
            st.markdown(f"""
            <div class="status-card-defective">
                <h3 style="color:#B91C1C; margin:0 0 4px 0;">🚨 DEFECTIVE</h3>
                <p style="margin:0; color:#7F1D1D;">{r.get('explanation', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-normal">
                <h3 style="color:#15803D; margin:0 0 4px 0;">✅ NORMAL (PASS)</h3>
                <p style="margin:0; color:#14532D;">{r.get('explanation', '')}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Metrics row
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Anomaly Score", f"{r.get('anomaly_score', 0.0):.4f}", delta=f"{r.get('anomaly_score', 0.0) - r.get('image_threshold', 0.0):+.4f} vs threshold", delta_color="inverse")
        with col_m2:
            st.metric("Calibrated Threshold", f"{r.get('image_threshold', 0.0):.2f}")
        with col_m3:
            st.metric("Localized Defects", f"{r.get('num_defects', 0)}")
        with col_m4:
            st.metric("Inference Latency", f"{r.get('inference_time_ms', 0.0):.1f} ms")
            
        # Visuals
        if r.get("visualization_base64"):
            img_data = base64.b64decode(r["visualization_base64"])
            overlay_pil = Image.open(io.BytesIO(img_data))
            
            # Find original image from active batch
            orig_img_bytes = next((fb for (fn, fb, _) in active_batch if fn == fname), None)
            if orig_img_bytes:
                vcol1, vcol2 = st.columns([1, 1])
                with vcol1:
                    st.image(Image.open(io.BytesIO(orig_img_bytes)), caption=f"Original: {fname}", use_container_width=True)
                with vcol2:
                    st.image(overlay_pil, caption=f"Defect Localization Overlay ({status})", use_container_width=True)
            else:
                st.image(overlay_pil, caption=f"Inspection Overlay ({status})", use_container_width=True)
                
        # Defect Regions Table
        if is_defective and r.get("localized_regions"):
            with st.expander(f"📍 View Defect Bounding Boxes ({len(r['localized_regions'])} regions)", expanded=False):
                st.table(r["localized_regions"])
                
        st.markdown("---")
