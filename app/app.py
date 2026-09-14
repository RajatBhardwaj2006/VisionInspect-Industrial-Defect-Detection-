"""
VisionInspect — AI-Powered Industrial Defect Detection & Localization
======================================================================
Frontend Web Application matching the reference UI design language:
- Color Palette: Navy (#2F4156), Teal (#567C8D), Sky Blue (#C8D9E6), Beige (#F5EFEB), White (#FFFFFF)
- Strict Image/Result/Category State Association (Request-ID Bound, Zero Stale Pairing)
- Interactive 4-Step Inspection Workflow with Realistic Industrial Red Laser Scanning Animation
- 5 Product Categories: Bottle, Leather, Transistor, Zipper, Screw
- 5 Pages: Home, Inspect, Analytics, Model Info, Documentation
"""
import os
import sys
import io
import time
import base64
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

import streamlit as st
import requests

# Set project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_category_config

# Page Configuration
st.set_page_config(
    page_title="VisionInspect — Industrial AI Visual Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS & METADATA
# -----------------------------------------------------------------------------
CATEGORIES = ["bottle", "leather", "transistor", "zipper", "screw"]
CATEGORY_META = {
    "bottle": {
        "title": "Bottle",
        "desc": "Cracks, contamination and structural defects.",
        "focus": "Mouth cracks, surface contamination, broken base.",
        "type": "Rigid Container",
        "sample_good": "dataset/mvtec_anomaly_detection/bottle/test/good/000.png",
        "sample_defect": "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png",
        "sample_defect_label": "Broken Large Defect"
    },
    "leather": {
        "title": "Leather",
        "desc": "Surface defects and texture irregularities.",
        "focus": "Cuts, color patches, folds and surface stains.",
        "type": "Natural Texture",
        "sample_good": "dataset/mvtec_anomaly_detection/leather/test/good/000.png",
        "sample_defect": "dataset/mvtec_anomaly_detection/leather/test/cut/000.png",
        "sample_defect_label": "Surface Cut Defect"
    },
    "transistor": {
        "title": "Transistor",
        "desc": "Fine leads, casing and placement anomalies.",
        "focus": "Bent pins, cut leads, damaged casing, misalignment.",
        "type": "Electronic Component",
        "sample_good": "dataset/mvtec_anomaly_detection/transistor/test/good/000.png",
        "sample_defect": "dataset/mvtec_anomaly_detection/transistor/test/cut_lead/000.png",
        "sample_defect_label": "Cut Lead Defect"
    },
    "zipper": {
        "title": "Zipper",
        "desc": "Broken teeth, fabric and structural defects.",
        "focus": "Missing teeth, fabric border fraying, slider defects.",
        "type": "Fastener Assembly",
        "sample_good": "dataset/mvtec_anomaly_detection/zipper/test/good/000.png",
        "sample_defect": "dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png",
        "sample_defect_label": "Broken Teeth Defect"
    },
    "screw": {
        "title": "Screw",
        "desc": "Thread, head and surface defects.",
        "focus": "Thread damage, scratch head, tip deformation.",
        "type": "Threaded Fastener",
        "sample_good": "dataset/mvtec_anomaly_detection/screw/test/good/000.png",
        "sample_defect": "dataset/mvtec_anomaly_detection/screw/test/scratch_head/000.png",
        "sample_defect_label": "Scratch Head Defect"
    }
}

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Inspect"
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = "screw"
if "uploaded_image_data" not in st.session_state:
    st.session_state["uploaded_image_data"] = None  # (filename, bytes, (w, h))
if "current_inspection" not in st.session_state:
    st.session_state["current_inspection"] = None
if "recent_inspections" not in st.session_state:
    st.session_state["recent_inspections"] = []
if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = DEFAULT_BACKEND_URL
if "is_scanning" not in st.session_state:
    st.session_state["is_scanning"] = False

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (NAVY / TEAL / SKY BLUE / BEIGE / WHITE THEME)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #2F4156;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }

    /* Sidebar Styling (Deep Navy) */
    section[data-testid="stSidebar"] {
        background-color: #2F4156 !important;
        border-right: 1px solid #1E2D3D;
    }
    
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FFFFFF !important;
    }
    
    /* Top Bar & Cards */
    .top-system-bar {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 0.9rem 1.4rem;
        box-shadow: 0 2px 10px rgba(47, 65, 86, 0.05);
        border: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #EFF6FB 0%, #FFFFFF 100%);
        border: 1px solid #C8D9E6;
        border-radius: 16px;
        padding: 1.6rem 2.0rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(47, 65, 86, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #2F4156;
        line-height: 1.15;
        margin: 0.3rem 0;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #567C8D;
        max-width: 600px;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    
    .badge-ai {
        display: inline-flex;
        align-items: center;
        background: #E0F2FE;
        color: #0369A1;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 20px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
        border: 1px solid #BAE6FD;
    }
    
    .feature-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.45rem 0.85rem;
        font-size: 0.82rem;
        color: #2F4156;
        font-weight: 600;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    
    /* White Content Cards */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.3rem;
        box-shadow: 0 3px 12px rgba(47, 65, 86, 0.04);
        margin-bottom: 1.2rem;
    }
    
    .step-header {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #2F4156;
        margin-bottom: 0.9rem;
    }
    
    .step-num {
        background: #2F4156;
        color: #FFFFFF;
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.82rem;
        font-weight: 800;
    }

    /* Category Cards */
    .category-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
        position: relative;
    }
    .category-card:hover {
        border-color: #567C8D;
        box-shadow: 0 4px 12px rgba(86, 124, 141, 0.15);
        transform: translateY(-2px);
    }
    .category-card.selected {
        border-color: #567C8D !important;
        background: #F0F7FB !important;
        box-shadow: 0 4px 14px rgba(86, 124, 141, 0.25) !important;
    }

    /* Laser Scanning Frame Animation */
    .scan-container {
        position: relative;
        width: 100%;
        max-width: 280px;
        margin: 0 auto;
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #567C8D;
        background: #000;
    }
    
    .scan-corner-tl {
        position: absolute; top: 6px; left: 6px; width: 18px; height: 18px;
        border-top: 3px solid #06B6D4; border-left: 3px solid #06B6D4; z-index: 10;
    }
    .scan-corner-tr {
        position: absolute; top: 6px; right: 6px; width: 18px; height: 18px;
        border-top: 3px solid #06B6D4; border-right: 3px solid #06B6D4; z-index: 10;
    }
    .scan-corner-bl {
        position: absolute; bottom: 6px; left: 6px; width: 18px; height: 18px;
        border-bottom: 3px solid #06B6D4; border-left: 3px solid #06B6D4; z-index: 10;
    }
    .scan-corner-br {
        position: absolute; bottom: 6px; right: 6px; width: 18px; height: 18px;
        border-bottom: 3px solid #06B6D4; border-right: 3px solid #06B6D4; z-index: 10;
    }

    .red-laser-line {
        position: absolute;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, rgba(255,0,0,0) 0%, rgba(255,50,50,0.9) 15%, #FF0000 50%, rgba(255,50,50,0.9) 85%, rgba(255,0,0,0) 100%);
        box-shadow: 0 0 14px 2px #FF0000, 0 0 4px #FFFFFF;
        top: 0%;
        left: 0;
        z-index: 8;
        animation: laserScan 2.2s ease-in-out infinite alternate;
    }
    
    @keyframes laserScan {
        0% { top: 5%; opacity: 0.9; }
        50% { opacity: 1.0; }
        100% { top: 92%; opacity: 0.9; }
    }

    /* Result Status Badges */
    .badge-normal {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        padding: 0.75rem 1.4rem;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.8rem;
    }
    
    .badge-defective {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        padding: 0.75rem 1.4rem;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.8rem;
    }
    
    .badge-signal-normal {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FCD34D;
        padding: 0.75rem 1.4rem;
        border-radius: 10px;
        font-weight: 800;
        font-size: 1.15rem;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.8rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        text-align: left;
    }
    .metric-val {
        font-size: 1.35rem;
        font-weight: 800;
        color: #2F4156;
        line-height: 1.2;
    }
    .metric-lbl {
        font-size: 0.75rem;
        font-weight: 700;
        color: #567C8D;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .metric-sub {
        font-size: 0.70rem;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    /* Why Result Callout */
    .why-card {
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        margin-top: 0.8rem;
    }
    .why-card-defective {
        background: #FEF2F2;
        border: 1px solid #FECACA;
        border-radius: 10px;
        padding: 0.8rem 1.1rem;
        margin-top: 0.8rem;
    }
    
    /* Checklist Items */
    .chk-item {
        font-size: 0.85rem;
        color: #334155;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS (IN-MEMORY IMAGE & INFERENCE PIPELINE)
# -----------------------------------------------------------------------------
def get_image_bytes(image_path: Path) -> bytes:
    with open(image_path, "rb") as f:
        return f.read()

def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def execute_inspection(category: str, img_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Executes inspection via FastAPI backend if available, or local PatchCore fallback.
    Guarantees that request_id, original image, heatmap, and overlay belong strictly
    to the same inference run.
    """
    backend_url = st.session_state.get("backend_url", DEFAULT_BACKEND_URL)
    req_id = f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
    
    # 1. Try FastAPI backend
    try:
        files_payload = {"file": (filename, io.BytesIO(img_bytes), "image/png")}
        data_payload = {"category": category, "return_visualizations": "true"}
        r = requests.post(f"{backend_url}/predict", files=files_payload, data=data_payload, timeout=25)
        if r.status_code == 200:
            data = r.json()
            data["request_id"] = req_id
            return data
    except Exception as e:
        pass

    # 2. Local Fallback Execution
    from src.detection.patchcore_v23_detector import PatchCoreDetectorV23
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    t0 = time.perf_counter()
    detector = PatchCoreDetectorV23(category=category)
    res = detector.inspect(pil_img)
    t_elapsed = time.perf_counter() - t0

    return {
        "request_id": req_id,
        "filename": filename,
        "category": category,
        "status": res["status"],
        "is_defective": res["status"] == "DEFECTIVE",
        "anomaly_score": round(float(res["score"]), 4),
        "image_threshold": round(float(res["image_threshold"]), 4),
        "pixel_threshold": round(float(res["pixel_threshold"]), 4),
        "decision_margin": round(float(res.get("decision_margin", res["score"] - res["image_threshold"])), 4),
        "num_defects": len(res.get("localized_regions", [])),
        "localized_regions": res.get("localized_regions", []),
        "heatmap_base64": res.get("heatmap_base64"),
        "visualization_base64": res.get("visualization_base64"),
        "inference_time_s": round(t_elapsed, 2),
        "inference_time_ms": round(t_elapsed * 1000, 1),
        "explanation": res["explanation"],
        "why_explanation": res.get("why_explanation", "")
    }


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    # Logo & Brand Header
    st.markdown("""
    <div style="padding: 0.8rem 0 1.2rem 0; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.8rem;">⚙️</span>
            <div>
                <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">VisionInspect</div>
                <div style="font-size: 0.72rem; color: #C8D9E6; font-weight: 500;">AI for a defect-free tomorrow</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation Menu
    nav_options = ["Home", "Inspect", "Analytics", "Model Info", "Documentation", "Settings"]
    nav_icons = {"Home": "🏠", "Inspect": "🔍", "Analytics": "📊", "Model Info": "🧠", "Documentation": "📖", "Settings": "⚙️"}
    
    st.markdown("<p style='font-size: 0.75rem; color: #C8D9E6; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.4rem;'>Main Navigation</p>", unsafe_allow_html=True)
    for opt in nav_options:
        icon = nav_icons[opt]
        is_active = st.session_state["nav_page"] == opt
        label = f"{icon}  {opt}"
        if st.button(label, key=f"nav_btn_{opt}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state["nav_page"] = opt
            st.rerun()

    # Sidebar Footer
    st.markdown("---")
    st.markdown("""
    <div style="padding-top: 1.0rem; color: #C8D9E6; font-size: 0.78rem;">
        <div style="font-weight: 700; color: #FFFFFF; margin-bottom: 3px;">VisionInspect v3.0</div>
        <div>Multi-Category PatchCore AI</div>
        <div style="margin-top: 8px; font-style: italic; color: #C8D9E6;">"Better Products, Brighter Tomorrows"</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP SYSTEM BAR (COMMON TO ALL PAGES)
# -----------------------------------------------------------------------------
backend_online = False
try:
    health_r = requests.get(f"{st.session_state['backend_url']}/health", timeout=1.0)
    if health_r.status_code == 200:
        backend_online = True
except Exception:
    backend_online = False

now_str = datetime.datetime.now().strftime("%b %d, %Y %H:%M")

st.markdown(f"""
<div class="top-system-bar">
    <div style="display: flex; align-items: center; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: {'#10B981' if backend_online else '#F59E0B'};"></span>
            <span style="font-size: 0.85rem; font-weight: 700; color: {'#065F46' if backend_online else '#92400E'};">
                {'Inspection Service Online' if backend_online else 'Standby Mode (Direct Engine)'}
            </span>
        </div>
        <span style="color: #CBD5E1;">|</span>
        <span style="font-size: 0.82rem; color: #64748B;">{now_str}</span>
    </div>
    <div style="font-size: 0.84rem; font-style: italic; color: #567C8D;">
        "Quality is intelligence in action."
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: HOME / LANDING PAGE
# =============================================================================
if st.session_state["nav_page"] == "Home":
    st.markdown("""
    <div class="hero-card">
        <div class="badge-ai">✨ AI-POWERED INDUSTRIAL INSPECTION</div>
        <div class="hero-title">Detect Defects.<br>Ensure Quality.</div>
        <div class="hero-subtitle">
            VisionInspect uses unsupervised deep-learning anomaly detection to identify microscopic, structural, and surface manufacturing defects across multiple industrial categories without labeled defect training data.
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1.0rem;">
            <span class="feature-pill">🎯 <b>High Accuracy</b> Reliable PatchCore</span>
            <span class="feature-pill">⚡ <b>Real-time</b> ~2.0s Inference</span>
            <span class="feature-pill">📦 <b>Multi-Category</b> 5 Product Domains</span>
            <span class="feature-pill">📊 <b>Precise Localization</b> Anomaly Heatmaps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([2, 1])
    with h_col1:
        st.markdown("### How VisionInspect Works")
        st.markdown("""
        ```
        [1. Upload Product] ➔ [2. Select Category] ➔ [3. Feature Extraction] ➔ [4. Memory Bank Match] ➔ [5. Heatmap & Localize]
        ```
        - **1. Normal Pattern Representation**: VisionInspect learns nominal visual feature distributions from purely defect-free reference components.
        - **2. ResNet18 Multi-Scale Features**: High-resolution patch embeddings (64×64 feature grid) preserve microscopic lead, thread, and texture details.
        - **3. Coreset Memory Bank**: Compact, representative normal embeddings enable ultra-fast nearest-neighbor anomaly scoring.
        - **4. Dual-Gated Decision**: An anomaly is confirmed only when both the image score exceeds threshold AND localized connected components pass strict area criteria.
        """)

        if st.button("🚀 Start Product Inspection Now →", type="primary", use_container_width=True):
            st.session_state["nav_page"] = "Inspect"
            st.rerun()

    with h_col2:
        st.markdown("### Supported Categories")
        for cat_k, cat_v in CATEGORY_META.items():
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.6rem 0.9rem; margin-bottom:8px;">
                <div style="font-weight:700; color:#2F4156; font-size:0.92rem;">{cat_v['title']} <span style="font-size:0.75rem; color:#567C8D; font-weight:500;">({cat_v['type']})</span></div>
                <div style="font-size:0.78rem; color:#64748B;">{cat_v['desc']}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: INSPECT WORKSPACE (PRIMARY WORKFLOW MATCHING REFERENCE UI)
# =============================================================================
elif st.session_state["nav_page"] == "Inspect":

    # Hero Banner
    st.markdown("""
    <div class="hero-card" style="padding: 1.2rem 1.6rem; margin-bottom: 1.0rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div class="badge-ai" style="margin-bottom:0.2rem;">AI-POWERED INDUSTRIAL INSPECTION</div>
                <div style="font-size: 1.6rem; font-weight: 800; color: #2F4156; line-height: 1.15;">
                    Detect Defects. Ensure Quality.
                </div>
                <div style="font-size: 0.88rem; color: #567C8D; margin-top: 0.2rem;">
                    Advanced AI to detect and localize manufacturing defects with high accuracy and speed.
                </div>
            </div>
            <div style="display:flex; gap:8px;">
                <span class="feature-pill">🎯 High Accuracy</span>
                <span class="feature-pill">⚡ Real-time</span>
                <span class="feature-pill">📦 Multi-Category</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-STEP GRID: Two Major Columns (Left: Steps 1, 2, 3 | Right: Step 4 Inspection Result)
    main_col_left, main_col_right = st.columns([1.15, 1.05], gap="medium")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: STEPS 1, 2, 3
    # -------------------------------------------------------------------------
    with main_col_left:

        # -----------------------------
        # STEP 1: SELECT PRODUCT CATEGORY
        # -----------------------------
        st.markdown("""
        <div class="step-header">
            <span class="step-num">1</span>
            <span>Select Product Category</span>
        </div>
        """, unsafe_allow_html=True)

        cat_cols = st.columns(5)
        for idx, cat_name in enumerate(CATEGORIES):
            meta = CATEGORY_META[cat_name]
            is_sel = st.session_state["selected_category"] == cat_name
            with cat_cols[idx]:
                # Category Button
                label = f"✓ {meta['title']}" if is_sel else meta['title']
                if st.button(label, key=f"cat_select_{cat_name}", use_container_width=True, type="primary" if is_sel else "secondary"):
                    if st.session_state["selected_category"] != cat_name:
                        st.session_state["selected_category"] = cat_name
                        # RESET CRITICAL STATE (Zero cross-category contamination)
                        st.session_state["uploaded_image_data"] = None
                        st.session_state["current_inspection"] = None
                        st.rerun()

        active_cat = st.session_state["selected_category"]
        active_meta = CATEGORY_META[active_cat]

        st.markdown(f"""
        <div style="background:#F1F5F9; border-radius:8px; padding:0.5rem 0.9rem; margin-top:0.5rem; margin-bottom:1.0rem; font-size:0.82rem; color:#475569;">
            <b>Active Model:</b> PatchCore v2.3 — <b style="color:#2F4156;">{active_meta['title']}</b> | 
            <b>Inspection Focus:</b> <i>{active_meta['focus']}</i>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # STEP 2: UPLOAD INSPECTION IMAGE
        # -----------------------------
        st.markdown("""
        <div class="step-header">
            <span class="step-num">2</span>
            <span>Upload Inspection Image</span>
        </div>
        """, unsafe_allow_html=True)

        up_col1, up_col2 = st.columns([1.1, 0.9])
        
        with up_col1:
            uploaded_file = st.file_uploader(
                "Drop inspection image here (PNG, JPG, Max 10MB)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"uploader_{active_cat}",
                help="Upload a single product image to run the AI defect inspection."
            )
            
            # Fast One-Click Demo Sample Selector (Perfect for Teacher Presentation)
            with st.expander("⚡ Quick Demo Samples (MVTec Test Set)", expanded=False):
                s_col1, s_col2 = st.columns(2)
                with s_col1:
                    if st.button(f"Normal Sample ({active_meta['title']})", key=f"btn_norm_{active_cat}", use_container_width=True):
                        sample_path = project_root / active_meta["sample_good"]
                        if sample_path.exists():
                            s_bytes = get_image_bytes(sample_path)
                            pil_im = Image.open(io.BytesIO(s_bytes))
                            st.session_state["uploaded_image_data"] = (sample_path.name, s_bytes, pil_im.size)
                            st.session_state["current_inspection"] = None
                            st.rerun()
                with s_col2:
                    if st.button(f"Defect: {active_meta['sample_defect_label']}", key=f"btn_def_{active_cat}", use_container_width=True):
                        sample_path = project_root / active_meta["sample_defect"]
                        if sample_path.exists():
                            s_bytes = get_image_bytes(sample_path)
                            pil_im = Image.open(io.BytesIO(s_bytes))
                            st.session_state["uploaded_image_data"] = (f"{active_cat}_{sample_path.parent.name}_{sample_path.name}", s_bytes, pil_im.size)
                            st.session_state["current_inspection"] = None
                            st.rerun()

        # Handle user-uploaded file
        if uploaded_file is not None:
            f_bytes = uploaded_file.read()
            try:
                pil_im = Image.open(io.BytesIO(f_bytes))
                current_stored = st.session_state["uploaded_image_data"]
                if current_stored is None or current_stored[0] != uploaded_file.name or current_stored[1] != f_bytes:
                    st.session_state["uploaded_image_data"] = (uploaded_file.name, f_bytes, pil_im.size)
                    st.session_state["current_inspection"] = None  # Clear previous result on new upload!
            except Exception:
                st.error("Corrupted or unreadable image file.")

        with up_col2:
            up_data = st.session_state["uploaded_image_data"]
            if up_data is not None:
                fname, fbytes, dims = up_data
                st.image(Image.open(io.BytesIO(fbytes)), caption=f"{fname} ({dims[0]}×{dims[1]}px)", use_container_width=True)
                if st.button("🗑️ Remove Image", key="btn_remove_img", use_container_width=True):
                    st.session_state["uploaded_image_data"] = None
                    st.session_state["current_inspection"] = None
                    st.rerun()
            else:
                st.markdown("""
                <div style="border: 2px dashed #CBD5E1; border-radius: 12px; padding: 2.2rem 1.0rem; text-align: center; color: #94A3B8;">
                    <div style="font-size: 2.0rem; margin-bottom: 0.3rem;">☁️</div>
                    <div style="font-size: 0.86rem; font-weight: 600;">No Image Selected</div>
                    <div style="font-size: 0.75rem;">Upload an image or choose a demo sample.</div>
                </div>
                """, unsafe_allow_html=True)

        # -----------------------------
        # STEP 3: SCANNING ANIMATION
        # -----------------------------
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="step-header">
            <span class="step-num">3</span>
            <span>AI Scanning & Feature Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state["uploaded_image_data"] is not None:
            fname, fbytes, dims = st.session_state["uploaded_image_data"]
            
            # Primary Run Inspection Action
            if st.button("🔍 Run AI Inspection", key="btn_run_inspection", type="primary", use_container_width=True):
                # Trigger real inspection
                with st.spinner("Extracting features and comparing against PatchCore memory bank..."):
                    result_payload = execute_inspection(active_cat, fbytes, fname)
                    # Bind atomic result
                    st.session_state["current_inspection"] = result_payload
                    
                    # Log to session recent inspections
                    new_hist = {
                        "id": len(st.session_state["recent_inspections"]) + 1,
                        "filename": fname,
                        "category": active_cat,
                        "status": result_payload["status"],
                        "score": result_payload["anomaly_score"],
                        "threshold": result_payload["image_threshold"],
                        "regions": result_payload["num_defects"],
                        "time_s": result_payload.get("inference_time_s", 2.1),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state["recent_inspections"].insert(0, new_hist)
                    st.rerun()

            # Realistic Laser Scanning Box
            scan_col1, scan_col2 = st.columns([1, 1])
            with scan_col1:
                img_b64 = base64.b64encode(fbytes).decode("utf-8")
                st.markdown(f"""
                <div class="scan-container">
                    <div class="scan-corner-tl"></div>
                    <div class="scan-corner-tr"></div>
                    <div class="scan-corner-bl"></div>
                    <div class="scan-corner-br"></div>
                    <div class="red-laser-line"></div>
                    <img src="data:image/png;base64,{img_b64}" style="width: 100%; display: block; filter: brightness(0.95);" />
                </div>
                """, unsafe_allow_html=True)
                
            with scan_col2:
                insp_done = st.session_state["current_inspection"] is not None
                st.markdown(f"""
                <div style="padding-left: 0.5rem;">
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Loading image bytes</b></div>
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Preprocessing (256×256)</b></div>
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Extracting ResNet18 patch features</b></div>
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Matching memory bank ({active_cat})</b></div>
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Evaluating dual-gated threshold</b></div>
                    <div class="chk-item">{'✅' if insp_done else '⏳'} <b>Defect region localization</b></div>
                    <div style="margin-top: 0.8rem; font-size: 0.76rem; color: #567C8D; background: #F1F5F9; padding: 0.5rem; border-radius: 6px;">
                        💡 <b>Tip:</b> Multi-scale patch analysis ensures sub-millimeter lead and thread anomalies are captured.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👈 Upload an inspection image or click a quick demo sample above to begin.")

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: STEP 4 INSPECTION RESULT
    # -------------------------------------------------------------------------
    with main_col_right:
        st.markdown("""
        <div class="step-header">
            <span class="step-num">4</span>
            <span>Inspection Result</span>
        </div>
        """, unsafe_allow_html=True)

        res = st.session_state["current_inspection"]

        if res is not None:
            # Check for Image / Category match
            curr_cat = st.session_state["selected_category"]
            is_defective = res.get("is_defective", False)
            status = res.get("status", "NORMAL")
            score = res.get("anomaly_score", 0.0)
            th = res.get("image_threshold", 0.0)
            margin = res.get("decision_margin", score - th)
            n_defects = res.get("num_defects", 0)
            latency_s = res.get("inference_time_s", 2.0)
            
            # Status Badge
            if is_defective:
                st.markdown("""
                <div class="badge-defective">
                    <span>⚠️</span>
                    <span>DEFECTIVE (FAIL)</span>
                </div>
                """, unsafe_allow_html=True)
            elif score > th and n_defects == 0:
                st.markdown("""
                <div class="badge-signal-normal">
                    <span>⚡</span>
                    <span>HIGH ANOMALY SIGNAL — NO CONFIRMED DEFECT</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="badge-normal">
                    <span>✔</span>
                    <span>NORMAL (PASS)</span>
                </div>
                """, unsafe_allow_html=True)

            # 1-2 line plain-language explanation
            st.markdown(f"<p style='font-size: 0.92rem; color: #334155; margin-bottom: 0.8rem;'>{res.get('explanation', '')}</p>", unsafe_allow_html=True)

            # Separate Metric Cards Row
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Anomaly Score</div>
                    <div class="metric-val">{score:.4f}</div>
                    <div class="metric-sub">Overall visual strength</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Decision Boundary</div>
                    <div class="metric-val">{th:.4f}</div>
                    <div class="metric-sub">Calibrated threshold</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col3:
                margin_color = "#B91C1C" if margin > 0 else "#15803D"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Decision Margin</div>
                    <div class="metric-val" style="color:{margin_color};">{margin:+.4f}</div>
                    <div class="metric-sub">Score - Threshold</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
            m_col4, m_col5 = st.columns(2)
            with m_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Localized Regions</div>
                    <div class="metric-val">{n_defects}</div>
                    <div class="metric-sub">{'Defect clusters found' if n_defects > 0 else 'No regions detected'}</div>
                </div>
                """, unsafe_allow_html=True)
            with m_col5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">Inference Time</div>
                    <div class="metric-val">{latency_s:.2f} s</div>
                    <div class="metric-sub">Feature extraction + match</div>
                </div>
                """, unsafe_allow_html=True)

            # Image Visualization Panels: Original Image, Anomaly Heatmap, Defect Localization
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #567C8D; text-transform: uppercase;'>Visual Inspection Panels</p>", unsafe_allow_html=True)

            v_col1, v_col2 = st.columns(2)
            with v_col1:
                if st.session_state["uploaded_image_data"]:
                    _, f_bytes, _ = st.session_state["uploaded_image_data"]
                    st.image(Image.open(io.BytesIO(f_bytes)), caption="Original Product Image", use_container_width=True)

            with v_col2:
                hm_b64 = res.get("heatmap_base64")
                if hm_b64:
                    st.image(Image.open(io.BytesIO(base64.b64decode(hm_b64))), caption="Anomaly Heatmap (Jet)", use_container_width=True)

            # Defect Localization Overlay
            ov_b64 = res.get("visualization_base64")
            if is_defective and ov_b64:
                st.image(Image.open(io.BytesIO(base64.b64decode(ov_b64))), caption=f"Defect Localization ({n_defects} bounding box{'es' if n_defects > 1 else ''})", use_container_width=True)
            elif not is_defective:
                st.markdown("""
                <div style="background: #F0FDF4; border: 1px dashed #86EFAC; border-radius: 8px; padding: 0.8rem; text-align: center; color: #166534; font-size: 0.85rem; font-weight: 600;">
                    ✔ NO CONFIRMED DEFECT REGION (Clean Normal Component)
                </div>
                """, unsafe_allow_html=True)

            # "Why this result?" card
            why_txt = res.get("why_explanation", "")
            card_class = "why-card-defective" if is_defective else "why-card"
            st.markdown(f"""
            <div class="{card_class}">
                <div style="font-weight: 700; font-size: 0.84rem; color: {'#991B1B' if is_defective else '#166534'}; margin-bottom: 2px;">
                    💡 Why this result?
                </div>
                <div style="font-size: 0.80rem; color: #334155;">
                    {why_txt}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Technical Details Collapsible
            with st.expander("🛠️ Technical Details (Viva / Architecture Summary)", expanded=False):
                st.markdown(f"""
                - **Architecture:** PatchCore v2.3
                - **Backbone:** ResNet18 (Layer 1, 2, 3 Multi-Scale Embeddings)
                - **Memory Bank:** `models/{active_cat}/patchcore_v23/`
                - **Learning Strategy:** Unsupervised visual feature modeling
                - **Request ID:** `{res.get('request_id', 'N/A')}`
                - **Calibrated Image Threshold:** `{th:.4f}`
                - **Calibrated Pixel Threshold:** `{res.get('pixel_threshold', 0.0):.4f}`
                """)
        else:
            st.markdown("""
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 3.0rem 1.5rem; text-align: center; color: #94A3B8;">
                <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">📋</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #475569;">Inspection Report Standby</div>
                <div style="font-size: 0.82rem; margin-top: 0.3rem;">Upload an image and run inspection to view the detailed report, metrics, and anomaly localization.</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: RECENT INSPECTIONS & QUICK INSIGHTS
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 1.6rem 0 1.2rem 0; border-color: #E2E8F0;' />", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1.5, 1.0, 0.9])

    with b_col1:
        st.markdown("### 🕒 Recent Inspections (Current Session)")
        hist = st.session_state["recent_inspections"]
        if hist:
            rows_html = ""
            for item in hist[:5]:
                badge_bg = "#FEE2E2" if item["status"] == "DEFECTIVE" else "#DCFCE7"
                badge_fg = "#B91C1C" if item["status"] == "DEFECTIVE" else "#15803D"
                rows_html += f"""
                <tr style="border-bottom: 1px solid #F1F5F9; font-size: 0.80rem;">
                    <td style="padding: 6px 8px;"><b>#{item['id']}</b></td>
                    <td style="padding: 6px 8px;"><code>{item['filename'][:16]}</code></td>
                    <td style="padding: 6px 8px; text-transform: capitalize;">{item['category']}</td>
                    <td style="padding: 6px 8px;"><span style="background:{badge_bg}; color:{badge_fg}; padding:2px 8px; border-radius:12px; font-weight:700;">{item['status']}</span></td>
                    <td style="padding: 6px 8px;">{item['score']:.3f}</td>
                    <td style="padding: 6px 8px;">{item['threshold']:.3f}</td>
                    <td style="padding: 6px 8px;">{item['regions']}</td>
                    <td style="padding: 6px 8px;">{item['time_s']}s</td>
                </tr>
                """
            st.markdown(f"""
            <table style="width: 100%; border-collapse: collapse; background: #FFFFFF; border-radius: 10px; overflow: hidden; border: 1px solid #E2E8F0;">
                <thead>
                    <tr style="background: #F8FAFC; text-align: left; font-size: 0.74rem; color: #567C8D; border-bottom: 1px solid #E2E8F0;">
                        <th style="padding: 8px;">#</th><th>Filename</th><th>Category</th><th>Status</th><th>Score</th><th>Threshold</th><th>Regions</th><th>Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 0.82rem; color: #94A3B8;'>No inspections recorded yet in this session.</p>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("### 📊 Quick Insights")
        total_scanned = len(st.session_state["recent_inspections"])
        normal_scanned = sum(1 for x in st.session_state["recent_inspections"] if x["status"] == "NORMAL")
        defective_scanned = sum(1 for x in st.session_state["recent_inspections"] if x["status"] == "DEFECTIVE")
        acc_pct = int((normal_scanned / max(1, total_scanned)) * 100) if total_scanned > 0 else 100

        q1, q2 = st.columns(2)
        with q1:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px;">
                <div class="metric-lbl">Total Scanned</div>
                <div class="metric-val">{total_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Defective</div>
                <div class="metric-val" style="color:#B91C1C;">{defective_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
        with q2:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px;">
                <div class="metric-lbl">Normal (Pass)</div>
                <div class="metric-val" style="color:#15803D;">{normal_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Pass Rate</div>
                <div class="metric-val">{acc_pct}%</div>
            </div>
            """, unsafe_allow_html=True)

    with b_col3:
        st.markdown("### 🏆 VisionInspect")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2F4156 0%, #1E2D3D 100%); color: #FFFFFF; border-radius: 12px; padding: 1.2rem; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 1.1rem; font-weight: 800; line-height: 1.2;">Precision Through AI</div>
                <div style="font-size: 0.76rem; color: #C8D9E6; margin-top: 4px;">Inspect. Improve. Innovate.</div>
            </div>
            <div style="font-size: 0.72rem; color: #94A3B8;">
                Autonomous industrial defect detection for manufacturing excellence.
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: ANALYTICS PAGE
# =============================================================================
elif st.session_state["nav_page"] == "Analytics":
    st.markdown("## 📊 Session Analytics & Performance")
    st.markdown("Inspect performance trends, detection ratios, and inference speed metrics recorded during the current session.")

    hist = st.session_state["recent_inspections"]
    total_imgs = len(hist)
    norm_imgs = sum(1 for x in hist if x["status"] == "NORMAL")
    def_imgs = sum(1 for x in hist if x["status"] == "DEFECTIVE")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Total Inspections", f"{total_imgs}")
    with a2:
        st.metric("Normal Parts Passed", f"{norm_imgs}")
    with a3:
        st.metric("Defective Parts Flagged", f"{def_imgs}")
    with a4:
        avg_lat = round(sum(x["time_s"] for x in hist) / max(1, total_imgs), 2) if hist else 2.1
        st.metric("Avg Inference Latency", f"{avg_lat} s")

    st.markdown("---")
    st.markdown("### Category Breakdown")
    cat_counts = {c: sum(1 for x in hist if x["category"] == c) for c in CATEGORIES}
    
    c_cols = st.columns(5)
    for idx, c in enumerate(CATEGORIES):
        with c_cols[idx]:
            st.markdown(f"""
            <div class="content-box" style="text-align: center;">
                <div style="font-weight: 700; text-transform: capitalize; color: #2F4156;">{c}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: #567C8D; margin: 4px 0;">{cat_counts[c]}</div>
                <div style="font-size: 0.75rem; color: #94A3B8;">Inspections</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: MODEL INFO PAGE (VIVA / ARCHITECTURE GUIDE)
# =============================================================================
elif st.session_state["nav_page"] == "Model Info":
    st.markdown("## 🧠 Machine Learning Architecture & Methodology")
    st.markdown("Deep dive into the unsupervised PatchCore architecture, feature extractors, and dual decision boundaries.")

    st.markdown("""
    ### 🔬 PatchCore Pipeline Overview
    ```
    Input Image (256×256)
          │
          ▼
    ResNet18 Feature Extractor (Layers 1, 2, 3)
          │
          ▼
    Multi-Scale Patch Descriptors (64×64 grid, 448 dimensions)
          │
          ▼
    Coreset Subsampled Memory Bank (~87,000 normal vectors)
          │
          ▼
    k-Nearest Neighbor Euclidean Distance Computation
          │
          ▼
    Anomaly Heatmap + Gaussian Smoothing (σ=1.2)
          │
          ▼
    Dual Decision Rule: [Score > T_image] AND [Localized Connected Component Area ≥ min_area]
          │
          ▼
    Final Classification: NORMAL (PASS) or DEFECTIVE (FAIL)
    ```
    """)

    st.markdown("---")
    st.markdown("### 🎓 Viva Preparation & Key Questions")
    with st.expander("Q1: Why is PatchCore superior to Convolutional Autoencoders?", expanded=True):
        st.write("""
        Autoencoders attempt to reconstruct images through an information bottleneck. They often suffer from blurry reconstructions and 'shortcut learning', where subtle cracks or tiny pin cuts are accidentally reconstructed well, depressing pixel IoU.
        
        PatchCore uses mid-level representations from a pre-trained ResNet18 backbone. Instead of reconstructing pixels, it measures local feature discrepancy against an exact memory bank of nominal patterns, achieving >99% AUROC on rigid industrial parts.
        """)

    with st.expander("Q2: What is the Dual Decision Rule?", expanded=False):
        st.write("""
        An image is classified as DEFECTIVE if and only if:
        1. The overall image anomaly score exceeds the calibrated category threshold (T_image).
        2. At least one connected component in the pixel anomaly map exceeds the minimum defect area (min_area) after morphology closing/opening.
        
        This eliminates false alarms from single-pixel sensor noise or specular reflections.
        """)

    with st.expander("Q3: How was zero test set leakage guaranteed?", expanded=False):
        st.write("""
        All image thresholds, pixel thresholds, and morphology parameters were calibrated exclusively on a held-out 20% validation split of normal training images, constrained strictly to a false positive rate ≤ 3.5%. The MVTec test set was never accessed during calibration.
        """)


# =============================================================================
# PAGE 5: DOCUMENTATION PAGE
# =============================================================================
elif st.session_state["nav_page"] == "Documentation":
    st.markdown("## 📖 VisionInspect Documentation")
    st.markdown("""
    ### 1. Project Scope
    VisionInspect is an unsupervised industrial visual inspection platform engineered for automated manufacturing quality assurance. It detects surface cracks, component misalignments, cuts, and thread flaws without requiring defective training samples.

    ### 2. Supported Categories & Benchmarks
    - **Bottle**: AUROC 0.9929 | Pixel IoU 0.4319 | Detection Rate 98.4%
    - **Transistor**: AUROC 0.9154 | Pixel F1 0.3737 | Detection Rate 80.0%
    - **Leather**: AUROC 0.9236 | Pixel Precision 0.1748 | Detection Rate 73.9%
    - **Zipper**: AUROC 0.8978 | Detection Rate 68.1%
    - **Screw**: AUROC 0.8774 | Detection Rate 66.4%

    ### 3. Step-by-Step Usage Guide
    1. Navigate to **Inspect** in the sidebar.
    2. Click on the desired **Product Category** (Bottle, Leather, Transistor, Zipper, Screw).
    3. Drag and drop your product photo or click **Quick Demo Samples** to test verified MVTec samples.
    4. Click **Run AI Inspection** to view the real-time scanning animation and generate the inspection report.
    """)


# =============================================================================
# PAGE 6: SETTINGS PAGE
# =============================================================================
elif st.session_state["nav_page"] == "Settings":
    st.markdown("## ⚙️ System Settings & Environment")
    
    st.subheader("FastAPI Backend Configuration")
    new_url = st.text_input("Backend Endpoint URL", value=st.session_state["backend_url"])
    if st.button("Save & Test Connection"):
        st.session_state["backend_url"] = new_url
        st.rerun()

    st.markdown("---")
    st.subheader("Session Management")
    if st.button("Clear Session History & Reset All State"):
        st.session_state["recent_inspections"] = []
        st.session_state["current_inspection"] = None
        st.session_state["uploaded_image_data"] = None
        st.success("Session state successfully cleared.")
        st.rerun()
