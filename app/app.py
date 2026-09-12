import os
import sys
import io
import time
import base64
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

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-card-defective {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .status-card-normal {
        background-color: #DCFCE7;
        border-left: 6px solid #22C55E;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.title("⚙️ Inspection Controls")

categories = ["bottle", "leather", "transistor", "zipper", "screw"]
selected_category = st.sidebar.selectbox("Select Target Category", categories, index=0)

cfg = load_config()
cat_cfg = get_category_config(selected_category, cfg)

st.sidebar.markdown("### Category Parameters")
st.sidebar.markdown(f"- **Architecture:** PatchCore v2.3")
st.sidebar.markdown(f"- **Backbone:** ResNet18 (Multi-scale 64x64)")
st.sidebar.markdown(f"- **Spatial Prior:** {'Enabled' if cat_cfg.get('use_spatial_prior', True) else 'Disabled (Texture Mode)'}")
st.sidebar.markdown(f"- **Image Threshold:** `{cat_cfg.get('image_threshold', 1.50):.2f}`")
st.sidebar.markdown(f"- **Pixel Threshold:** `{cat_cfg.get('pixel_threshold', 1.40):.2f}`")

backend_url = st.sidebar.text_input("Backend API URL", value="http://127.0.0.1:8000")

# Check backend health
backend_online = False
try:
    health_resp = requests.get(f"{backend_url}/health", timeout=1.5)
    if health_resp.status_code == 200:
        backend_online = True
        st.sidebar.success("● FastAPI Backend Online")
    else:
        st.sidebar.warning("▲ Backend reachable with warning")
except Exception:
    st.sidebar.info("○ Backend offline (Direct mode fallback)")

st.sidebar.markdown("---")
input_source = st.sidebar.radio("Select Input Source", ["Test Dataset Sample", "Upload Custom Image"])

selected_image_pil = None
sample_label = "Custom Image"

if input_source == "Test Dataset Sample":
    dataset_dir = Path("dataset/mvtec_anomaly_detection") / selected_category / "test"
    if dataset_dir.exists():
        defect_folders = sorted([f.name for f in dataset_dir.iterdir() if f.is_dir()])
        selected_defect = st.sidebar.selectbox("Defect Type", defect_folders)
        
        sample_files = sorted(list((dataset_dir / selected_defect).glob("*.png")))
        sample_names = [f.name for f in sample_files]
        if sample_names:
            selected_file_name = st.sidebar.selectbox("Sample Image", sample_names)
            chosen_path = dataset_dir / selected_defect / selected_file_name
            selected_image_pil = Image.open(chosen_path).convert("RGB")
            sample_label = f"{selected_category}/{selected_defect}/{selected_file_name}"
    else:
        st.sidebar.error(f"Dataset path {dataset_dir} not found.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        selected_image_pil = Image.open(uploaded_file).convert("RGB")
        sample_label = uploaded_file.name

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">VisionInspect: Unsupervised Industrial Defect Inspection</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Category: <b>{selected_category.upper()}</b> | Powered by High-Resolution PatchCore v2.3</div>', unsafe_allow_html=True)

if selected_image_pil is None:
    st.info("👈 Please select a sample image from the test set or upload your own image in the sidebar to begin inspection.")
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Input Product Image")
        st.image(selected_image_pil, caption=f"Selected: {sample_label}", use_container_width=True)
        inspect_button = st.button("🚀 Run Visual Inspection", type="primary", use_container_width=True)

    if inspect_button:
        with st.spinner("Analyzing image features & computing anomaly localization..."):
            pred_data = None
            
            # 1. Attempt API call
            if backend_online:
                try:
                    img_byte_arr = io.BytesIO()
                    selected_image_pil.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    files = {'file': ('image.png', img_byte_arr, 'image/png')}
                    data = {'category': selected_category, 'return_visualizations': 'true'}
                    response = requests.post(f"{backend_url}/predict", files=files, data=data, timeout=30)
                    if response.status_code == 200:
                        pred_data = response.json()
                    else:
                        st.warning(f"Backend returned status {response.status_code}. Falling back to direct detector.")
                except Exception as e:
                    st.warning(f"Backend communication failed ({e}). Falling back to direct detector.")
                    
            # 2. Fallback to direct detector
            if pred_data is None:
                from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
                t0 = time.perf_counter()
                detector = PatchCoreDetectorV23(category=selected_category)
                res = detector.inspect(selected_image_pil)
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
                    "category": selected_category,
                    "is_defective": res["status"] == "DEFECTIVE",
                    "status": res["status"],
                    "anomaly_score": round(float(res["score"]), 4),
                    "image_threshold": round(float(res["image_threshold"]), 4),
                    "pixel_threshold": round(float(res["pixel_threshold"]), 4),
                    "num_defects": len(defect_regions),
                    "defect_regions": defect_regions,
                    "visualization_base64": vis_b64,
                    "inference_time_ms": round(latency, 2),
                    "explanation": res["explanation"]
                }
                
            # ----------------- DISPLAY RESULTS -----------------
            is_defective = pred_data["is_defective"]
            
            with col2:
                st.subheader("Inspection Overlay & Heatmap")
                if pred_data.get("visualization_base64"):
                    img_data = base64.b64decode(pred_data["visualization_base64"])
                    overlay_pil = Image.open(io.BytesIO(img_data))
                    st.image(overlay_pil, caption=f"Defect Localization Overlay (Status: {pred_data['status']})", use_container_width=True)
                else:
                    st.info("No visualization overlay generated.")
                    
            st.markdown("---")
            
            # Status Banner
            if is_defective:
                st.markdown(f"""
                <div class="status-card-defective">
                    <h3 style="color:#B91C1C; margin:0;">🚨 DEFECT DETECTED ({pred_data['num_defects']} Region{'s' if pred_data['num_defects'] != 1 else ''})</h3>
                    <p style="margin:5px 0 0 0; color:#450A0A;">{pred_data['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card-normal">
                    <h3 style="color:#15803D; margin:0;">✅ PRODUCT NORMAL (PASS)</h3>
                    <p style="margin:5px 0 0 0; color:#052E16;">{pred_data['explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Key Metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Anomaly Score", f"{pred_data['anomaly_score']:.4f}", delta=f"{pred_data['anomaly_score'] - pred_data['image_threshold']:+.4f} vs threshold", delta_color="inverse")
            with m2:
                st.metric("Calibrated Threshold", f"{pred_data['image_threshold']:.2f}")
            with m3:
                st.metric("Defect Regions", f"{pred_data['num_defects']}")
            with m4:
                st.metric("Inference Latency", f"{pred_data['inference_time_ms']:.1f} ms")
                
            # Defect Regions Table
            if is_defective and pred_data.get("defect_regions"):
                st.markdown("### 📍 Localized Defect Regions")
                st.table(pred_data["defect_regions"])
