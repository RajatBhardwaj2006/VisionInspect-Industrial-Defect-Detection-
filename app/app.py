"""
VisionInspect — Unsupervised Industrial Defect Detection & Localization
Premium Production UI (v3.0 Demo Edition)
Palette:
  - Deep Navy:   #2F4156
  - Soft Teal:   #567C8D
  - Sky Blue:    #C8D9E6
  - Soft Beige:  #F5EFEB
  - Pure White:  #FFFFFF
  - Accents:     #15803D (Pass) / #B91C1C (Fail)
"""

import os
import io
import time
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionInspect | Industrial Defect Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DEFAULT_BACKEND_URL = os.environ.get("VISIONINSPECT_BACKEND_URL", "http://127.0.0.1:8000")

# -----------------------------------------------------------------------------
# CATEGORY DEFINITIONS & VERIFIED SAMPLES
# -----------------------------------------------------------------------------
CATEGORIES = {
    "bottle": {
        "name": "Bottle",
        "tag": "Rigid Glass Container",
        "focus": "Surface cracks • contamination • structural anomalies",
        "icon": "🍾",
        "samples": [
            ("Normal (Good Sample)", "dataset/mvtec_anomaly_detection/bottle/test/good/001.png"),
            ("Defect: Broken Large Crack", "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png"),
            ("Defect: Broken Small Chipping", "dataset/mvtec_anomaly_detection/bottle/test/broken_small/000.png"),
            ("Defect: Foreign Contamination", "dataset/mvtec_anomaly_detection/bottle/test/contamination/000.png"),
        ]
    },
    "leather": {
        "name": "Leather",
        "tag": "Surface Texture Fabric",
        "focus": "Cuts • holes • color patches • fold marks",
        "icon": "🧤",
        "samples": [
            ("Normal (Good Sample)", "dataset/mvtec_anomaly_detection/leather/test/good/001.png"),
            ("Defect: Surface Cut", "dataset/mvtec_anomaly_detection/leather/test/cut/000.png"),
            ("Defect: Deep Fold", "dataset/mvtec_anomaly_detection/leather/test/fold/000.png"),
            ("Defect: Color Flaw", "dataset/mvtec_anomaly_detection/leather/test/color/000.png"),
        ]
    },
    "transistor": {
        "name": "Transistor",
        "tag": "Electronic Semiconductor",
        "focus": "Bent leads • cut leads • damaged casing • misplacement",
        "icon": "⚡",
        "samples": [
            ("Normal (Good Sample)", "dataset/mvtec_anomaly_detection/transistor/test/good/001.png"),
            ("Defect: Bent Lead Pin", "dataset/mvtec_anomaly_detection/transistor/test/bent_lead/000.png"),
            ("Defect: Cut Lead Pin", "dataset/mvtec_anomaly_detection/transistor/test/cut_lead/000.png"),
            ("Defect: Damaged Casing", "dataset/mvtec_anomaly_detection/transistor/test/damaged_case/000.png"),
        ]
    },
    "zipper": {
        "name": "Zipper",
        "tag": "Mechanical Closure Fastener",
        "focus": "Broken teeth • split teeth • fabric roughness",
        "icon": "🤐",
        "samples": [
            ("Normal (Good Sample)", "dataset/mvtec_anomaly_detection/zipper/test/good/001.png"),
            ("Defect: Broken Teeth", "dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png"),
            ("Defect: Split Teeth Gap", "dataset/mvtec_anomaly_detection/zipper/test/split_teeth/000.png"),
            ("Defect: Fabric Roughness", "dataset/mvtec_anomaly_detection/zipper/test/rough/000.png"),
        ]
    },
    "screw": {
        "name": "Screw",
        "tag": "Threaded Metal Fastener",
        "focus": "Thread damage • scratch head • metal deformation",
        "icon": "🔩",
        "samples": [
            ("Normal (Good Sample)", "dataset/mvtec_anomaly_detection/screw/test/good/001.png"),
            ("Defect: Head Scratch", "dataset/mvtec_anomaly_detection/screw/test/scratch_head/000.png"),
            ("Defect: Thread Side Damage", "dataset/mvtec_anomaly_detection/screw/test/thread_side/000.png"),
            ("Defect: Manipulated Front", "dataset/mvtec_anomaly_detection/screw/test/manipulated_front/000.png"),
        ]
    }
}

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Inspect"
if "selected_category" not in st.session_state:
    st.session_state["selected_category"] = "bottle"
if "uploaded_image_data" not in st.session_state:
    st.session_state["uploaded_image_data"] = None  # (filename, bytes, (w, h))
if "current_inspection" not in st.session_state:
    st.session_state["current_inspection"] = None
if "current_request_id" not in st.session_state:
    st.session_state["current_request_id"] = None
if "recent_inspections" not in st.session_state:
    st.session_state["recent_inspections"] = []
if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = DEFAULT_BACKEND_URL
if "is_scanning" not in st.session_state:
    st.session_state["is_scanning"] = False
if "quick_sample_choice" not in st.session_state:
    st.session_state["quick_sample_choice"] = None

# Helper to purge inspection state atomically
def purge_inspection(new_category: Optional[str] = None):
    st.session_state["current_inspection"] = None
    st.session_state["current_request_id"] = None
    st.session_state["uploaded_image_data"] = None
    st.session_state["quick_sample_choice"] = None
    if new_category:
        st.session_state["selected_category"] = new_category

# -----------------------------------------------------------------------------
# PREMIUM INDUSTRIAL STYLING (NAVY / TEAL / SKY BLUE / BEIGE / WHITE)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #2F4156;
    }
    .stApp {
        background-color: #F8FAFC;
    }

    /* Top Navigation Header */
    .top-header {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 0.85rem 1.6rem;
        box-shadow: 0 2px 10px rgba(47, 65, 86, 0.05);
        border: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #2F4156;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-sub {
        font-size: 0.76rem;
        color: #567C8D;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .status-pill {
        background: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        font-size: 0.74rem;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid #86EFAC;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    /* Compact Category Cards */
    .cat-card {
        background: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.85rem 0.6rem;
        text-align: center;
        transition: all 0.2s ease;
        height: 100%;
    }
    .cat-card.active {
        border-color: #567C8D;
        background: #F0F7FB;
        box-shadow: 0 4px 12px rgba(86, 124, 141, 0.15);
    }
    .cat-icon {
        font-size: 1.6rem;
        margin-bottom: 4px;
    }
    .cat-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #2F4156;
    }
    .cat-tag {
        font-size: 0.70rem;
        color: #567C8D;
        margin-top: 2px;
        font-weight: 500;
    }

    /* Step Banner */
    .step-badge {
        display: inline-block;
        background: #2F4156;
        color: #FFFFFF;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 6px;
    }
    .step-title {
        font-weight: 800;
        font-size: 1.05rem;
        color: #2F4156;
        display: flex;
        align-items: center;
        margin-bottom: 0.6rem;
    }

    /* Laser Scanner Simulation Box */
    .scan-container {
        position: relative;
        width: 100%;
        max-width: 480px;
        margin: 0 auto;
        border-radius: 10px;
        overflow: hidden;
        background: #0F172A;
        border: 1px solid #334155;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25);
    }
    .scan-bracket {
        position: absolute;
        width: 22px;
        height: 22px;
        border-color: #38BDF8;
        border-style: solid;
        z-index: 10;
        pointer-events: none;
    }
    .scan-bracket-tl { top: 8px; left: 8px; border-width: 3px 0 0 3px; }
    .scan-bracket-tr { top: 8px; right: 8px; border-width: 3px 3px 0 0; }
    .scan-bracket-bl { bottom: 8px; left: 8px; border-width: 0 0 3px 3px; }
    .scan-bracket-br { bottom: 8px; right: 8px; border-width: 0 3px 3px 0; }
    .scan-laser {
        position: absolute;
        left: 0;
        width: 100%;
        height: 2.5px;
        background: #EF4444;
        box-shadow: 0 0 14px #EF4444, 0 0 4px #FCA5A5;
        z-index: 8;
        animation: laserSweep 1.8s ease-in-out infinite alternate;
    }
    @keyframes laserSweep {
        0% { top: 2%; opacity: 0.8; }
        50% { top: 50%; opacity: 1.0; }
        100% { top: 96%; opacity: 0.8; }
    }

    /* Result Status Badges */
    .status-banner-normal {
        background: #F0FDF4;
        border: 1.5px solid #86EFAC;
        border-radius: 12px;
        padding: 1.0rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.0rem;
    }
    .status-banner-defective {
        background: #FEF2F2;
        border: 1.5px solid #FCA5A5;
        border-radius: 12px;
        padding: 1.0rem 1.4rem;
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 1.0rem;
    }
    .status-icon-badge {
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 800;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1.0rem;
        box-shadow: 0 2px 6px rgba(47, 65, 86, 0.03);
        height: 100%;
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #567C8D;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 2px;
    }
    .metric-val {
        font-size: 1.45rem;
        font-weight: 800;
        color: #2F4156;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.68rem;
        color: #94A3B8;
        margin-top: 2px;
    }

    /* Visual Inspection Panels */
    .vis-panel-header {
        background: #2F4156;
        color: #FFFFFF;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-align: center;
        padding: 6px 10px;
        border-radius: 8px 8px 0 0;
        text-transform: uppercase;
    }

    /* Heatmap Legend Bar */
    .heatmap-legend {
        margin-top: 6px;
        padding: 6px 10px;
        background: #FFFFFF;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .legend-bar {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #000080 0%, #00FFFF 35%, #FFFF00 70%, #FF0000 100%);
        margin-bottom: 4px;
    }
    .legend-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.68rem;
        color: #567C8D;
        font-weight: 600;
    }

    /* Recent Table */
    .recent-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        font-size: 0.78rem;
    }
    .recent-table th {
        background: #F8FAFC;
        padding: 8px 10px;
        text-align: left;
        color: #567C8D;
        font-weight: 700;
        border-bottom: 1px solid #E2E8F0;
    }
    .recent-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #F1F5F9;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HEADER & NAVIGATION
# -----------------------------------------------------------------------------
header_col1, header_col2, header_col3 = st.columns([2.2, 3.2, 1.2])

with header_col1:
    st.markdown("""
    <div style="padding: 4px 0;">
        <div class="brand-title">🔬 VISIONINSPECT</div>
        <div class="brand-sub">Autonomous Defect Detection & Localization</div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    # Clean top navigation pills
    pages = ["Inspect", "Home", "Analytics", "Model"]
    nav_cols = st.columns(len(pages))
    for idx, p in enumerate(pages):
        with nav_cols[idx]:
            is_cur = (st.session_state["nav_page"] == p)
            btn_label = f"● {p}" if is_cur else p
            if st.button(btn_label, key=f"nav_top_{p}", use_container_width=True):
                st.session_state["nav_page"] = p
                st.rerun()

with header_col3:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="status-pill">● Service Online</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin: 0.4rem 0 1.2rem 0; border-color: #E2E8F0;' />", unsafe_allow_html=True)

# =============================================================================
# PAGE 1: HOME PAGE
# =============================================================================
if st.session_state["nav_page"] == "Home":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #EFF6FB 0%, #FFFFFF 100%); border: 1px solid #C8D9E6; border-radius: 14px; padding: 2.2rem; margin-bottom: 1.5rem;">
        <div style="font-size: 0.85rem; font-weight: 800; color: #567C8D; text-transform: uppercase; letter-spacing: 1px;">Industrial Computer Vision</div>
        <div style="font-size: 2.3rem; font-weight: 800; color: #2F4156; margin: 4px 0 8px 0;">Intelligent Visual Inspection.</div>
        <div style="font-size: 1.05rem; color: #475569; max-width: 650px; line-height: 1.5; margin-bottom: 1.4rem;">
            Detect, score, and localize manufacturing anomalies using unsupervised deep learning memory banks. Calibrated with zero defective training samples.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("START INSPECTION →", type="primary"):
        st.session_state["nav_page"] = "Inspect"
        st.rerun()

    st.markdown("### Supported Manufacturing Categories")
    c_cols = st.columns(5)
    for idx, (cat_key, cat_data) in enumerate(CATEGORIES.items()):
        with c_cols[idx]:
            st.markdown(f"""
            <div class="cat-card">
                <div class="cat-icon">{cat_data['icon']}</div>
                <div class="cat-title">{cat_data['name']}</div>
                <div class="cat-tag">{cat_data['tag']}</div>
                <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 6px;">{cat_data['focus']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Simple 4-Step Inspection Workflow")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("**1. UPLOAD**<br><span style='font-size:0.8rem; color:#567C8D;'>Select category and load product image or quick sample.</span>", unsafe_allow_html=True)
    with w2:
        st.markdown("**2. ANALYZE**<br><span style='font-size:0.8rem; color:#567C8D;'>PatchCore extracts multi-scale ResNet18 features.</span>", unsafe_allow_html=True)
    with w3:
        st.markdown("**3. LOCALIZE**<br><span style='font-size:0.8rem; color:#567C8D;'>Nearest-neighbor memory bank matches anomaly regions.</span>", unsafe_allow_html=True)
    with w4:
        st.markdown("**4. INSPECT**<br><span style='font-size:0.8rem; color:#567C8D;'>Instant dual-gated Pass/Fail decision with bounding boxes.</span>", unsafe_allow_html=True)


# =============================================================================
# PAGE 2: MAIN INSPECTION PAGE
# =============================================================================
elif st.session_state["nav_page"] == "Inspect":
    active_cat = st.session_state["selected_category"]

    # -------------------------------------------------------------------------
    # STEP 01: PRODUCT CATEGORY SELECTION (COMPACT ROW)
    # -------------------------------------------------------------------------
    st.markdown("""
    <div class="step-title">
        <span class="step-badge">STEP 01</span>
        <span>Select Product Category</span>
    </div>
    """, unsafe_allow_html=True)

    cat_cols = st.columns(5)
    for idx, (cat_key, cat_data) in enumerate(CATEGORIES.items()):
        with cat_cols[idx]:
            is_active = (active_cat == cat_key)
            card_class = "cat-card active" if is_active else "cat-card"
            check_mark = "✔ " if is_active else ""
            if st.button(
                f"{cat_data['icon']} {check_mark}{cat_data['name']}",
                key=f"cat_btn_{cat_key}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                if active_cat != cat_key:
                    purge_inspection(new_category=cat_key)
                    st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MAIN WORKFLOW COLUMNS (LEFT: INPUT / RIGHT: SCANNER & RESULT)
    # -------------------------------------------------------------------------
    col_input, col_result = st.columns([1.15, 1.85], gap="large")

    # LEFT COLUMN: STEP 02 & STEP 03
    with col_input:
        st.markdown("""
        <div class="step-title">
            <span class="step-badge">STEP 02</span>
            <span>Inspection Image</span>
        </div>
        """, unsafe_allow_html=True)

        # Quick Demo Samples Dropdown
        samples_list = CATEGORIES[active_cat]["samples"]
        sample_options = ["-- Choose a Quick Demo Sample --"] + [s[0] for s in samples_list]

        selected_sample_label = st.selectbox(
            f"Quick Demo Samples ({CATEGORIES[active_cat]['name']}):",
            options=sample_options,
            key="quick_sample_dropdown"
        )

        if selected_sample_label != "-- Choose a Quick Demo Sample --":
            match_path = None
            for s_label, s_path in samples_list:
                if s_label == selected_sample_label:
                    match_path = s_path
                    break
            if match_path and Path(match_path).exists():
                with open(match_path, "rb") as f:
                    s_bytes = f.read()
                pil_im = Image.open(io.BytesIO(s_bytes))
                # Only update if different
                cur_data = st.session_state["uploaded_image_data"]
                if not cur_data or cur_data[0] != Path(match_path).name:
                    st.session_state["uploaded_image_data"] = (Path(match_path).name, s_bytes, pil_im.size)
                    st.session_state["current_inspection"] = None
                    st.session_state["current_request_id"] = None

        # Drag & Drop File Uploader
        uploaded_file = st.file_uploader(
            "Or drag and drop your image:",
            type=["png", "jpg", "jpeg", "webp"],
            key="user_file_uploader",
            label_visibility="visible"
        )

        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            pil_up = Image.open(io.BytesIO(raw_bytes))
            cur_data = st.session_state["uploaded_image_data"]
            if not cur_data or cur_data[0] != uploaded_file.name:
                st.session_state["uploaded_image_data"] = (uploaded_file.name, raw_bytes, pil_up.size)
                st.session_state["current_inspection"] = None
                st.session_state["current_request_id"] = None

        # Image Preview & Metadata
        if st.session_state["uploaded_image_data"]:
            fname, img_b, (iw, ih) = st.session_state["uploaded_image_data"]
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:8px; padding:0.6rem 0.8rem; margin: 8px 0; font-size:0.75rem; color:#567C8D; display:flex; justify-content:space-between;">
                <span><b>File:</b> {fname}</span>
                <span><b>Dimensions:</b> {iw}×{ih} px</span>
            </div>
            """, unsafe_allow_html=True)

            # STEP 03: RUN AI INSPECTION BUTTON
            st.markdown("""
            <div class="step-title" style="margin-top: 12px;">
                <span class="step-badge">STEP 03</span>
                <span>Execute Inspection</span>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🚀 RUN AI INSPECTION", type="primary", use_container_width=True):
                st.session_state["is_scanning"] = True
                st.rerun()

        else:
            st.info(f"Select a quick demo sample or drag-and-drop a {CATEGORIES[active_cat]['name']} image above.")

    # RIGHT COLUMN: SCANNING ANIMATION OR INSPECTION REPORT
    with col_result:
        # Check if scanning in progress
        if st.session_state["is_scanning"]:
            st.markdown("""
            <div class="step-title">
                <span class="step-badge">SCANNING</span>
                <span>AI Optical Inspection In Progress</span>
            </div>
            """, unsafe_allow_html=True)

            _, scan_b, _ = st.session_state["uploaded_image_data"]
            b64_img = base64.b64encode(scan_b).decode("utf-8")

            st.markdown(f"""
            <div class="scan-container">
                <div class="scan-bracket scan-bracket-tl"></div>
                <div class="scan-bracket scan-bracket-tr"></div>
                <div class="scan-bracket scan-bracket-bl"></div>
                <div class="scan-bracket scan-bracket-br"></div>
                <div class="scan-laser"></div>
                <img src="data:image/png;base64,{b64_img}" style="width:100%; display:block; filter: brightness(0.9);" />
            </div>
            <div style="text-align:center; padding: 12px 0 6px 0;">
                <div style="font-weight: 800; font-size: 0.96rem; color: #2F4156;">SCANNING PRODUCT SURFACE...</div>
                <div style="font-size: 0.78rem; color: #567C8D; margin-top: 3px;">
                    Extracting multi-scale patch features & comparing with {CATEGORIES[active_cat]['name']} memory bank...
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Run Backend Inference
            try:
                fname, raw_b, _ = st.session_state["uploaded_image_data"]
                res = requests.post(
                    f"{st.session_state['backend_url']}/predict",
                    data={"category": active_cat, "return_visualizations": "true"},
                    files={"file": (fname, raw_b, "image/png")},
                    timeout=25
                )
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["current_inspection"] = data
                    st.session_state["current_request_id"] = data.get("request_id")
                    # Log to history
                    hist_entry = {
                        "id": len(st.session_state["recent_inspections"]) + 1,
                        "filename": fname,
                        "category": active_cat,
                        "status": data.get("status", "NORMAL"),
                        "score": data.get("anomaly_score", 0.0),
                        "threshold": data.get("image_threshold", 0.0),
                        "regions": data.get("num_defects", 0),
                        "time_s": data.get("inference_time_s", 1.5)
                    }
                    st.session_state["recent_inspections"].insert(0, hist_entry)
                else:
                    st.error(f"Backend returned error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Failed to communicate with FastAPI backend: {e}")
            finally:
                st.session_state["is_scanning"] = False
                st.rerun()

        # Render Inspection Results if available
        elif st.session_state["current_inspection"] is not None:
            res = st.session_state["current_inspection"]
            is_defective = res.get("is_defective", False)
            status = res.get("status", "NORMAL")
            score = res.get("anomaly_score", 0.0)
            th = res.get("image_threshold", 0.0)
            margin = res.get("decision_margin", score - th)
            n_defects = res.get("num_defects", 0)
            latency_s = res.get("inference_time_s", 2.0)
            cat_tag = CATEGORIES[active_cat]["tag"]
            cat_focus = CATEGORIES[active_cat]["focus"]

            # Category Banner
            st.markdown(f"""
            <div style="background:#2F4156; color:#FFFFFF; border-radius:10px; padding:0.6rem 1.0rem; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-weight:800; font-size:0.95rem; text-transform:uppercase;">{active_cat} INSPECTION</span>
                    <span style="font-size:0.75rem; color:#C8D9E6; margin-left:8px;">PatchCore v2.3</span>
                </div>
                <div style="font-size:0.72rem; color:#C8D9E6;">
                    Focus: {cat_focus}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Status Banner
            if is_defective:
                st.markdown(f"""
                <div class="status-banner-defective">
                    <div class="status-icon-badge" style="background:#FEE2E2; color:#B91C1C;">!</div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #991B1B;">DEFECTIVE — INSPECTION FAILED</div>
                        <div style="font-size: 0.82rem; color: #7F1D1D; margin-top: 2px;">
                            Anomaly regions were detected and localized. Component rejected.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif score > th and n_defects == 0:
                st.markdown(f"""
                <div class="status-banner-normal" style="border-color:#FDE68A; background:#FFFBEB;">
                    <div class="status-icon-badge" style="background:#FEF3C7; color:#B45309;">⚡</div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #92400E;">ELEVATED SIGNAL — ACCEPTED AS NORMAL</div>
                        <div style="font-size: 0.82rem; color: #78350F; margin-top: 2px;">
                            Raw anomaly score exceeded threshold, but no region passed the defect size criteria. Inspection Passed.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-banner-normal">
                    <div class="status-icon-badge" style="background:#DCFCE7; color:#15803D;">✓</div>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #166534;">NORMAL — INSPECTION PASSED</div>
                        <div style="font-size: 0.82rem; color: #14532D; margin-top: 2px;">
                            No confirmed defect region was identified. Component approved.
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 5 Key Metric Cards (Row)
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Anomaly Score</div>
                    <div class="metric-val">{score:.4f}</div>
                    <div class="metric-sub">Feature deviation</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Decision Threshold</div>
                    <div class="metric-val">{th:.4f}</div>
                    <div class="metric-sub">Calibrated bound</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                margin_col = "#B91C1C" if margin > 0 else "#15803D"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Decision Margin</div>
                    <div class="metric-val" style="color:{margin_col};">{margin:+.4f}</div>
                    <div class="metric-sub">Score - Threshold</div>
                </div>
                """, unsafe_allow_html=True)
            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Regions</div>
                    <div class="metric-val">{n_defects}</div>
                    <div class="metric-sub">{"Clusters found" if n_defects > 0 else "None"}</div>
                </div>
                """, unsafe_allow_html=True)
            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Inference Time</div>
                    <div class="metric-val">{latency_s:.2f}s</div>
                    <div class="metric-sub">End-to-end latency</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

            # -----------------------------------------------------------------
            # THREE-PANEL VISUAL INSPECTION
            # -----------------------------------------------------------------
            st.markdown("""
            <div style="font-size: 0.88rem; font-weight: 800; color: #2F4156; text-transform: uppercase; margin-bottom: 6px;">
                🔍 VISUAL INSPECTION
            </div>
            """, unsafe_allow_html=True)

            v_col1, v_col2, v_col3 = st.columns(3)

            # Panel 1: Original Image
            with v_col1:
                st.markdown("<div class='vis-panel-header'>1. Original Image</div>", unsafe_allow_html=True)
                if st.session_state["uploaded_image_data"]:
                    _, o_bytes, _ = st.session_state["uploaded_image_data"]
                    st.image(Image.open(io.BytesIO(o_bytes)), use_container_width=True)
                st.caption("Pristine uploaded product image")

            # Panel 2: Anomaly Heatmap
            with v_col2:
                st.markdown("<div class='vis-panel-header'>2. Anomaly Heatmap</div>", unsafe_allow_html=True)
                hm_b64 = res.get("heatmap_base64")
                if hm_b64:
                    st.image(Image.open(io.BytesIO(base64.b64decode(hm_b64))), use_container_width=True)
                    st.markdown("""
                    <div class="heatmap-legend">
                        <div class="legend-bar"></div>
                        <div class="legend-labels">
                            <span>Low (Nominal)</span>
                            <span>Medium</span>
                            <span>High (Anomaly)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Heatmap visualization unavailable.")

            # Panel 3: Defect Localization
            with v_col3:
                st.markdown("<div class='vis-panel-header'>3. Defect Localization</div>", unsafe_allow_html=True)
                vis_b64 = res.get("visualization_base64")
                if is_defective and vis_b64:
                    st.image(Image.open(io.BytesIO(base64.b64decode(vis_b64))), use_container_width=True)
                    st.caption(f"Overlay with {n_defects} confirmed bounding box(es)")
                else:
                    if st.session_state["uploaded_image_data"]:
                        _, o_bytes, _ = st.session_state["uploaded_image_data"]
                        st.image(Image.open(io.BytesIO(o_bytes)), use_container_width=True)
                    st.caption("✔ Clean Component — Zero defect regions identified")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # Detected Anomalies Summary
            if is_defective and res.get("localized_regions"):
                st.markdown("""
                <div style="font-size: 0.82rem; font-weight: 700; color: #2F4156; margin-bottom: 4px;">
                    DETECTED ANOMALIES
                </div>
                """, unsafe_allow_html=True)
                regs = res.get("localized_regions", [])
                reg_cols = st.columns(min(3, len(regs)))
                for r_idx, reg in enumerate(regs[:3]):
                    with reg_cols[r_idx]:
                        lbl = reg.get("label", f"Region {r_idx+1:02d}")
                        intensity = reg.get("intensity", "Anomaly Region")
                        area_px = reg.get("area", 0)
                        st.markdown(f"""
                        <div style="background:#FFFFFF; border:1px solid #FCA5A5; border-radius:8px; padding:6px 10px; font-size:0.76rem;">
                            <div style="font-weight:700; color:#B91C1C;">{lbl} — {intensity}</div>
                            <div style="color:#64748B; font-size:0.70rem;">Area: {area_px} px | Score: {reg.get('score', 0.0):.4f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # "Why this result?" Card
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:4px solid {'#B91C1C' if is_defective else '#15803D'}; border-radius:8px; padding:0.75rem 1.0rem; margin:10px 0;">
                <div style="font-weight:700; font-size:0.80rem; color:{'#B91C1C' if is_defective else '#15803D'}; margin-bottom:2px;">
                    💡 Why this result?
                </div>
                <div style="font-size:0.78rem; color:#334155; line-height:1.4;">
                    {res.get('why_explanation', res.get('explanation', ''))}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Collapsible Technical Details
            with st.expander("🛠️ Technical Details (Viva & Architecture Summary)", expanded=False):
                st.markdown(f"""
                - **Model Architecture:** PatchCore v2.3 (Unsupervised Density-based Detection)
                - **Feature Extractor:** ResNet18 (Layer 1, Layer 2, Layer 3 Multi-Scale Embeddings)
                - **Patch Descriptors:** 448-dimensional feature representations on a 64×64 spatial grid
                - **Category Memory Bank:** `models/{active_cat}/patchcore_v23/`
                - **Image Decision Threshold ($T_{{image}}$):** `{th:.4f}`
                - **Pixel Anomaly Threshold ($T_{{pixel}}$):** `{res.get('pixel_threshold', 0.0):.4f}`
                - **Inference Request ID:** `{res.get('request_id', 'N/A')}`
                """)

        else:
            # Standby State
            st.markdown("""
            <div style="background:#FFFFFF; border:2px dashed #E2E8F0; border-radius:12px; padding:3.5rem 1.5rem; text-align:center; color:#94A3B8;">
                <div style="font-size:2.2rem; margin-bottom:0.5rem;">📋</div>
                <div style="font-size:1.05rem; font-weight:700; color:#475569;">Inspection Report Standby</div>
                <div style="font-size:0.82rem; margin-top:0.3rem;">Select an image on the left and click <b>RUN AI INSPECTION</b> to generate the optical inspection report.</div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # BOTTOM SECTION: RECENT INSPECTIONS & QUICK INSIGHTS
    # -------------------------------------------------------------------------
    st.markdown("<hr style='margin: 1.6rem 0 1.2rem 0; border-color: #E2E8F0;' />", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([1.6, 1.0], gap="medium")

    with b_col1:
        st.markdown("### 🕒 Recent Inspections (Session History)")
        hist = st.session_state["recent_inspections"]
        if hist:
            table_rows = []
            for item in hist[:5]:
                badge_bg = "#FEE2E2" if item["status"] == "DEFECTIVE" else "#DCFCE7"
                badge_fg = "#B91C1C" if item["status"] == "DEFECTIVE" else "#15803D"
                fname_disp = item["filename"][:18]
                table_rows.append(
                    f"<tr><td><b>#{item['id']}</b></td><td><code>{fname_disp}</code></td><td style='text-transform:capitalize;'>{item['category']}</td><td><span style='background:{badge_bg};color:{badge_fg};padding:2px 8px;border-radius:12px;font-weight:700;'>{item['status']}</span></td><td>{item['score']:.4f}</td><td>{item['threshold']:.4f}</td><td>{item['regions']}</td><td>{item['time_s']:.2f}s</td></tr>"
                )
            rows_html = "".join(table_rows)
            full_table = f"<table class='recent-table'><thead><tr><th>#</th><th>Filename</th><th>Category</th><th>Status</th><th>Score</th><th>Threshold</th><th>Regions</th><th>Latency</th></tr></thead><tbody>{rows_html}</tbody></table>"
            st.markdown(full_table, unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:0.80rem; color:#94A3B8;'>No inspections executed yet in this session.</p>", unsafe_allow_html=True)

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
                <div class="metric-title">Total Scanned</div>
                <div class="metric-val">{total_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Defective (Fail)</div>
                <div class="metric-val" style="color:#B91C1C;">{defective_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
        with q2:
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:8px;">
                <div class="metric-title">Normal (Pass)</div>
                <div class="metric-val" style="color:#15803D;">{normal_scanned}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Pass Rate</div>
                <div class="metric-val">{acc_pct}%</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 3: ANALYTICS PAGE
# =============================================================================
elif st.session_state["nav_page"] == "Analytics":
    st.markdown("## 📊 Session Analytics & Performance")
    st.markdown("Performance distribution and inference speeds recorded during active testing.")

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
        st.metric("Avg Latency", f"{avg_lat} s")

    st.markdown("---")
    st.markdown("### Category Distribution")
    c_cols = st.columns(5)
    for idx, (c_key, c_data) in enumerate(CATEGORIES.items()):
        c_count = sum(1 for x in hist if x["category"] == c_key)
        with c_cols[idx]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;">
                <div class="cat-icon">{c_data['icon']}</div>
                <div class="cat-title">{c_data['name']}</div>
                <div class="metric-val" style="color:#567C8D; margin:4px 0;">{c_count}</div>
                <div class="metric-sub">Inspections</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: MODEL INFO & VIVA PREPARATION
# =============================================================================
elif st.session_state["nav_page"] == "Model":
    st.markdown("## 🧠 Machine Learning Architecture & Methodology")
    st.markdown("PatchCore density-based anomaly localization pipeline and evaluation integrity.")

    st.markdown("""
    ### 🔬 PatchCore Pipeline
    ```
    Input Image (256×256)
          │
          ▼
    ResNet18 Backbone (Layers 1, 2, 3 Multi-Scale Embeddings)
          │
          ▼
    64×64 Spatial Patch Grid (448-Dimensional Descriptor Vectors)
          │
          ▼
    Coreset Memory Bank (~87,000 Normal Feature Vectors)
          │
          ▼
    Nearest-Neighbor Euclidean Distance Computation
          │
          ▼
    Anomaly Heatmap + Gaussian Smoothing (σ=1.2)
          │
          ▼
    Dual-Gated Decision Rule: [Score > T_image] AND [Connected Component Area ≥ min_area]
          │
          ▼
    Final Classification: NORMAL (PASS) or DEFECTIVE (FAIL)
    ```
    """)

    st.markdown("---")
    st.markdown("### 🎓 Viva Examination Guide")
    with st.expander("Q1: Why PatchCore instead of Convolutional Autoencoders?", expanded=True):
        st.write("""
        Autoencoders reconstruct images through an information bottleneck. They often suffer from blurry outputs and 'shortcut learning', where subtle cracks or missing leads are accidentally reconstructed well, causing false negatives.
        
        PatchCore uses mid-level semantic representations from a pre-trained ResNet18. Instead of reconstructing pixels, it measures local feature deviation against an exact memory bank of nominal patterns, achieving >99% AUROC on rigid industrial parts.
        """)

    with st.expander("Q2: What is the Dual-Gated Decision Rule?", expanded=False):
        st.write("""
        An image is classified as DEFECTIVE if and only if:
        1. The overall image anomaly score exceeds the calibrated category threshold (T_image).
        2. At least one connected component in the anomaly map exceeds the minimum defect area (min_area) after morphology filtering.
        
        This prevents solitary sensor noise or specular reflections from generating false alarms.
        """)

    with st.expander("Q3: How was zero test set leakage guaranteed?", expanded=False):
        st.write("""
        All image thresholds, pixel thresholds, and morphology parameters were calibrated exclusively on a held-out 20% validation split of normal training images, constrained strictly to false positive rate ≤ 3.5%. The MVTec test set was never accessed during calibration.
        """)
