import os
import sys
import io
import time
import base64
import hashlib
from pathlib import Path
from PIL import Image

import streamlit as st
import requests

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_category_config

st.set_page_config(
    page_title="VisionInspect — Industrial Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
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
        margin-bottom: 1.2rem;
    }
    .status-card-defective {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1.1rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    .status-card-normal {
        background-color: #DCFCE7;
        border-left: 6px solid #22C55E;
        padding: 1.1rem;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    .category-notice {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE INITIALIZATION -----------------
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None
if "image_name" not in st.session_state:
    st.session_state["image_name"] = None
if "image_hash" not in st.session_state:
    st.session_state["image_hash"] = None
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = "bottle"
if "inspection_result" not in st.session_state:
    st.session_state["inspection_result"] = None

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ Inspection Setup")

categories = ["bottle", "leather", "transistor", "zipper", "screw"]
category_labels = [c.capitalize() for c in categories]

current_cat_idx = categories.index(st.session_state["selected_category"]) if st.session_state["selected_category"] in categories else 0
chosen_label = st.sidebar.selectbox("Product Category", category_labels, index=current_cat_idx)
new_category = chosen_label.lower()

# Clear results if user switches category
if new_category != st.session_state["selected_category"]:
    st.session_state["selected_category"] = new_category
    st.session_state["inspection_result"] = None
    st.rerun()

cfg = load_config()
cat_cfg = get_category_config(new_category, cfg)

st.sidebar.markdown("### Category Parameters")
st.sidebar.markdown(f"- **Architecture:** PatchCore v2.3")
st.sidebar.markdown(f"- **Backbone:** ResNet18 (64x64 Grid)")
st.sidebar.markdown(f"- **Spatial Prior:** {'Enabled' if cat_cfg.get('use_spatial_prior', True) else 'Disabled (Texture Mode)'}")
st.sidebar.markdown(f"- **Image Threshold:** `{cat_cfg.get('image_threshold', 1.50):.2f}`")
st.sidebar.markdown(f"- **Pixel Threshold:** `{cat_cfg.get('pixel_threshold', 1.40):.2f}`")

backend_url = st.sidebar.text_input("FastAPI Backend URL", value="http://127.0.0.1:8000")

# Check backend connectivity
backend_online = False
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=1.5)
    if health_resp.status_code == 200:
        backend_online = True
        st.sidebar.success("● FastAPI Backend Online")
    else:
        st.sidebar.warning("▲ Backend responded with warning")
except Exception:
    st.sidebar.info("○ Backend offline (Direct mode fallback)")

st.sidebar.markdown("---")
st.sidebar.markdown("**VisionInspect Industrial Inspection**")
st.sidebar.caption("Unsupervised Anomaly Localization & Defect Detection")

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">VisionInspect — Industrial Defect Detection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Active Model: <b>PatchCore v2.3</b> | Category: <b>{new_category.upper()}</b></div>', unsafe_allow_html=True)

# Important User Warning on Category Matching
st.markdown(f"""
<div class="category-notice">
    ⚠️ <b>Category Matching Notice:</b> The selected category (<b>{chosen_label}</b>) must match the physical product shown in the image.
    VisionInspect runs specialized unsupervised feature memory banks per product type.
</div>
""", unsafe_allow_html=True)

# ----------------- SECTION 1: UPLOAD CUSTOM IMAGE (PRIMARY) -----------------
st.subheader("1. Upload Product Image")
uploaded_file = st.file_uploader(
    "Drag and drop your product image here, or click to browse files",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload an arbitrary product image for inspection."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    file_hash = hashlib.sha256(file_bytes).hexdigest()
    
    # Check if a new file was uploaded
    if st.session_state["image_hash"] != file_hash:
        st.session_state["image_bytes"] = file_bytes
        st.session_state["image_name"] = uploaded_file.name
        st.session_state["image_hash"] = file_hash
        st.session_state["inspection_result"] = None  # Clear previous results!

# ----------------- SECTION 2: OPTIONAL SAMPLE BROWSER (SECONDARY) -----------------
with st.expander("📁 Optional: Test with an MVTec Dataset Sample", expanded=False):
    dataset_dir = Path("dataset/mvtec_anomaly_detection") / new_category / "test"
    if dataset_dir.exists():
        defect_folders = sorted([f.name for f in dataset_dir.iterdir() if f.is_dir()])
        col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
        with col_s1:
            sample_defect = st.selectbox("Sample Defect Class", defect_folders, key="sample_defect_select")
        with col_s2:
            sample_files = sorted(list((dataset_dir / sample_defect).glob("*.png")))
            sample_names = [f.name for f in sample_files]
            chosen_sample_name = st.selectbox("Sample Image", sample_names, key="sample_img_select")
        with col_s3:
            st.write("")
            st.write("")
            if st.button("Load Dataset Sample"):
                sample_p = dataset_dir / sample_defect / chosen_sample_name
                with open(sample_p, "rb") as sf:
                    s_bytes = sf.read()
                st.session_state["image_bytes"] = s_bytes
                st.session_state["image_name"] = f"MVTec: {new_category}/{sample_defect}/{chosen_sample_name}"
                st.session_state["image_hash"] = hashlib.sha256(s_bytes).hexdigest()
                st.session_state["inspection_result"] = None
                st.rerun()
    else:
        st.info("MVTec dataset directory not found locally.")

st.markdown("---")

# ----------------- SECTION 3: INSPECTION WORKFLOW -----------------
if st.session_state["image_bytes"] is None:
    st.info("👈 **Upload an image to begin inspection.** (Drag & drop or browse above)")
else:
    # Load PIL image from session bytes
    active_image = Image.open(io.BytesIO(st.session_state["image_bytes"])).convert("RGB")
    
    col_img, col_act = st.columns([1, 1])
    with col_img:
        st.subheader("Selected Image Preview")
        st.image(active_image, caption=f"Loaded: {st.session_state['image_name']}", use_container_width=True)
        
    with col_act:
        st.subheader("Run Inspection")
        st.write(f"Product Category: **{new_category.upper()}**")
        st.write(f"Image Dimensions: **{active_image.width} × {active_image.height} px**")
        st.write(f"Inspection Model: **PatchCore v2.3 (ResNet18)**")
        
        inspect_button = st.button("🔍 INSPECT IMAGE", type="primary", use_container_width=True)
        if inspect_button:
            with st.spinner(f"Inspecting image against {new_category} normal feature space..."):
                pred_data = None
                
                # 1. Attempt call to FastAPI backend
                if backend_online:
                    try:
                        files = {'file': (st.session_state['image_name'] or 'image.png', io.BytesIO(st.session_state['image_bytes']), 'image/png')}
                        data = {'category': new_category, 'return_visualizations': 'true'}
                        response = requests.post(f"{backend_url}/predict", files=files, data=data, timeout=30)
                        if response.status_code == 200:
                            pred_data = response.json()
                        else:
                            st.warning(f"Backend returned status {response.status_code}. Falling back to local detector.")
                    except Exception as e:
                        st.warning(f"Backend communication failed ({e}). Falling back to local detector.")
                        
                # 2. Local Fallback if backend offline
                if pred_data is None:
                    from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
                    t0 = time.perf_counter()
                    detector = PatchCoreDetectorV23(category=new_category)
                    res = detector.inspect(active_image)
                    latency = (time.perf_counter() - t0) * 1000.0
                    
                    defect_regions = []
                    for idx, reg in enumerate(res.get("localized_regions", [])):
                        w = reg["width"]
                        h = reg["height"]
                        defect_regions.append({
                            "id": idx + 1,
                            "bbox": [reg["x"], reg["y"], w, h],
                            "area": reg["area"],
                            "score": round(float(reg.get("score", 0.0)), 4),
                            "aspect_ratio": round(float(w / max(1, h)), 2)
                        })
                        
                    vis_b64 = None
                    if "saved_path" in res and Path(res["saved_path"]).exists():
                        with open(res["saved_path"], "rb") as f:
                            vis_b64 = base64.b64encode(f.read()).decode("utf-8")
                            
                    pred_data = {
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
                        "explanation": res["explanation"]
                    }
                    
                st.session_state["inspection_result"] = pred_data
                st.rerun()

    # ----------------- SECTION 4: DISPLAY RESULTS -----------------
    result = st.session_state["inspection_result"]
    if result is not None:
        st.markdown("---")
        is_defective = result.get("is_defective", False)
        status_label = "DEFECTIVE" if is_defective else "NORMAL"
        
        # Result Banner
        if is_defective:
            st.markdown(f"""
            <div class="status-card-defective">
                <h2 style="color:#B91C1C; margin:0 0 5px 0;">🚨 DEFECTIVE</h2>
                <p style="margin:0; color:#7F1D1D; font-size:1.05rem;">{result['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-card-normal">
                <h2 style="color:#15803D; margin:0 0 5px 0;">✅ NORMAL (PASS)</h2>
                <p style="margin:0; color:#14532D; font-size:1.05rem;">{result['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Metric Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Anomaly Score",
                f"{result['anomaly_score']:.4f}",
                delta=f"{result['anomaly_score'] - result['image_threshold']:+.4f} vs threshold",
                delta_color="inverse"
            )
        with m2:
            st.metric("Calibrated Threshold", f"{result['image_threshold']:.2f}")
        with m3:
            st.metric("Localized Defect Regions", f"{result['num_defects']}")
        with m4:
            st.metric("Inference Latency", f"{result['inference_time_ms']:.1f} ms")
            
        # Visual Comparison
        st.subheader("Inspection Visuals")
        if is_defective:
            vis_col1, vis_col2 = st.columns([1, 1])
            with vis_col1:
                st.image(active_image, caption="Original Product Image", use_container_width=True)
            with vis_col2:
                if result.get("visualization_base64"):
                    img_data = base64.b64decode(result["visualization_base64"])
                    overlay_pil = Image.open(io.BytesIO(img_data))
                    st.image(overlay_pil, caption="Defect Localization Overlay (Bounding Boxes & Heatmap)", use_container_width=True)
                else:
                    st.info("Overlay visualization not available.")
                    
            # Localized Defect Regions Table
            if result.get("localized_regions"):
                st.subheader("📍 Localized Defect Regions")
                st.table(result["localized_regions"])
        else:
            vis_col1, vis_col2 = st.columns([1, 1])
            with vis_col1:
                st.image(active_image, caption="Original Product Image (No Defects)", use_container_width=True)
            with vis_col2:
                if result.get("visualization_base64"):
                    img_data = base64.b64decode(result["visualization_base64"])
                    overlay_pil = Image.open(io.BytesIO(img_data))
                    st.image(overlay_pil, caption="Normal Heatmap Verification (Clean Surface)", use_container_width=True)
                else:
                    st.success("Clean surface verified. Zero defect regions detected.")
