"""
VisionInspect — Precision Industrial Optical Defect Inspection
Minimalist Product & Industrial Vision Experience (Apple-Style Design System)

Design System:
  - Background:     #F5F5F3 (Warm minimal light gray)
  - Surface:        #FFFFFF (Crisp white)
  - Primary Text:   #171717 (Deep black)
  - Secondary Text: #6B6B6B (Neutral mid-gray)
  - Borders:        #E5E5E2 (Subtle 1px border)
  - Primary Action: #171717 (Solid black button / white text)
  - Pass Status:    #15803D (Restrained forest green)
  - Fail Status:    #B91C1C (Restrained crimson red)
"""

import os
import io
import time
import base64
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List

import streamlit as st
import requests
from PIL import Image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CHROME REMOVAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionInspect",
    page_icon="○",
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
        "label": "Rigid Glass Container",
        "description": "Surface cracks, chipping, and particulate contamination.",
        "golden_sample": "dataset/mvtec_anomaly_detection/bottle/test/good/001.png",
        "samples": [
            ("Normal Baseline (Golden)", "dataset/mvtec_anomaly_detection/bottle/test/good/001.png"),
            ("Broken Large Crack", "dataset/mvtec_anomaly_detection/bottle/test/broken_large/000.png"),
            ("Broken Small Chipping", "dataset/mvtec_anomaly_detection/bottle/test/broken_small/000.png"),
            ("Foreign Contamination", "dataset/mvtec_anomaly_detection/bottle/test/contamination/000.png"),
        ]
    },
    "leather": {
        "name": "Leather",
        "label": "Surface Texture Fabric",
        "description": "Cuts, punctures, color flaws, and deep folds.",
        "golden_sample": "dataset/mvtec_anomaly_detection/leather/test/good/001.png",
        "samples": [
            ("Normal Baseline (Golden)", "dataset/mvtec_anomaly_detection/leather/test/good/001.png"),
            ("Surface Cut", "dataset/mvtec_anomaly_detection/leather/test/cut/000.png"),
            ("Deep Fold Mark", "dataset/mvtec_anomaly_detection/leather/test/fold/000.png"),
            ("Color Flaw Patch", "dataset/mvtec_anomaly_detection/leather/test/color/000.png"),
        ]
    },
    "transistor": {
        "name": "Transistor",
        "label": "Semiconductor IC",
        "description": "Bent leads, cut pins, casing damage, and missing components.",
        "golden_sample": "dataset/mvtec_anomaly_detection/transistor/test/good/001.png",
        "samples": [
            ("Normal Baseline (Golden)", "dataset/mvtec_anomaly_detection/transistor/test/good/001.png"),
            ("Bent Terminal Lead", "dataset/mvtec_anomaly_detection/transistor/test/bent_lead/000.png"),
            ("Cut Terminal Lead", "dataset/mvtec_anomaly_detection/transistor/test/cut_lead/000.png"),
            ("Damaged Package Casing", "dataset/mvtec_anomaly_detection/transistor/test/damaged_case/000.png"),
            ("Missing Component Body", "dataset/mvtec_anomaly_detection/transistor/test/misplaced/002.png"),
        ]
    },
    "zipper": {
        "name": "Zipper",
        "label": "Mechanical Fastener",
        "description": "Broken teeth, split tooth gaps, and fabric weave roughness.",
        "golden_sample": "dataset/mvtec_anomaly_detection/zipper/test/good/001.png",
        "samples": [
            ("Normal Baseline (Golden)", "dataset/mvtec_anomaly_detection/zipper/test/good/001.png"),
            ("Broken Teeth Chain", "dataset/mvtec_anomaly_detection/zipper/test/broken_teeth/000.png"),
            ("Split Teeth Gap", "dataset/mvtec_anomaly_detection/zipper/test/split_teeth/000.png"),
            ("Fabric Edge Roughness", "dataset/mvtec_anomaly_detection/zipper/test/rough/000.png"),
        ]
    },
    "screw": {
        "name": "Screw",
        "label": "Threaded Metal Fastener",
        "description": "Thread deformation, drive head scratches, and metal flaws.",
        "golden_sample": "dataset/mvtec_anomaly_detection/screw/test/good/001.png",
        "samples": [
            ("Normal Baseline (Golden)", "dataset/mvtec_anomaly_detection/screw/test/good/001.png"),
            ("Drive Head Scratch", "dataset/mvtec_anomaly_detection/screw/test/scratch_head/000.png"),
            ("Thread Side Flaw", "dataset/mvtec_anomaly_detection/screw/test/thread_side/000.png"),
            ("Manipulated Front Face", "dataset/mvtec_anomaly_detection/screw/test/manipulated_front/000.png"),
        ]
    }
}

# -----------------------------------------------------------------------------
# SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home"
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

# Zero-indentation HTML renderer to prevent CommonMark code block leaks
def render_html(content: str):
    st.markdown(textwrap.dedent(content).strip(), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MINIMALIST INDUSTRIAL DESIGN SYSTEM (APPLE / INDUSTRIAL LAB AESTHETIC)
# -----------------------------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global resets */
    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }
    
    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "SF Pro Text", "Helvetica Neue", sans-serif;
        background-color: #F5F5F3;
        color: #171717;
        letter-spacing: -0.01em;
    }

    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
    }

    /* Minimal Navbar */
    .vi-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0 1.1rem 0;
        border-bottom: 1px solid #E5E5E2;
        margin-bottom: 1.6rem;
    }
    .vi-brand {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #171717;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .vi-brand-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #171717;
        display: inline-block;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.15s ease !important;
        border: 1px solid #E5E5E2 !important;
        background-color: #FFFFFF !important;
        color: #171717 !important;
        box-shadow: none !important;
    }
    div.stButton > button:hover {
        background-color: #ECECE9 !important;
        border-color: #D4D4D0 !important;
        color: #171717 !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: #171717 !important;
        color: #FFFFFF !important;
        border: 1px solid #171717 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #2E2E2E !important;
        border-color: #2E2E2E !important;
        color: #FFFFFF !important;
    }

    /* Hero Typography */
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6B6B6B;
        margin-bottom: 12px;
    }
    .hero-heading {
        font-size: 3.4rem;
        font-weight: 700;
        line-height: 1.06;
        letter-spacing: -0.035em;
        color: #171717;
        margin: 0 0 16px 0;
    }
    .hero-sub {
        font-size: 1.08rem;
        line-height: 1.55;
        color: #6B6B6B;
        max-width: 440px;
        margin-bottom: 24px;
        font-weight: 400;
    }

    /* Central Hero Product Visual Container */
    .hero-visual-frame {
        position: relative;
        background: #FFFFFF;
        border: 1px solid #E5E5E2;
        border-radius: 8px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 440px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
    }
    .corner-bracket {
        position: absolute;
        width: 14px;
        height: 14px;
        border-color: #A1A1AA;
        border-style: solid;
        pointer-events: none;
    }
    .corner-tl { top: 12px; left: 12px; border-width: 1.5px 0 0 1.5px; }
    .corner-tr { top: 12px; right: 12px; border-width: 1.5px 1.5px 0 0; }
    .corner-bl { bottom: 12px; left: 12px; border-width: 0 0 1.5px 1.5px; }
    .corner-br { bottom: 12px; right: 12px; border-width: 0 1.5px 1.5px 0; }

    .visual-top-meta {
        position: absolute;
        top: 14px;
        left: 32px;
        right: 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #8E8E93;
        text-transform: uppercase;
    }
    .visual-bottom-meta {
        position: absolute;
        bottom: 14px;
        left: 32px;
        right: 32px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.68rem;
        color: #8E8E93;
    }

    /* Scanning Container */
    .scan-container {
        position: relative;
        width: 100%;
        max-width: 480px;
        margin: 0 auto;
        background: #FFFFFF;
        border: 1px solid #E5E5E2;
        border-radius: 8px;
        overflow: hidden;
        padding: 16px;
    }
    .scan-laser {
        position: absolute;
        left: 0;
        width: 100%;
        height: 2px;
        background: #EF4444;
        box-shadow: 0 0 8px rgba(239, 68, 68, 0.7);
        z-index: 10;
        animation: laserScanSweep 1.8s ease-in-out infinite alternate;
    }
    @keyframes laserScanSweep {
        0% { top: 4%; opacity: 0.8; }
        50% { top: 50%; opacity: 1.0; }
        100% { top: 96%; opacity: 0.8; }
    }

    /* Category Pill Selector */
    .cat-selector-row {
        display: flex;
        align-items: center;
        gap: 18px;
        margin: 18px 0 24px 0;
        flex-wrap: wrap;
    }
    .cat-text-item {
        font-size: 0.88rem;
        font-weight: 500;
        color: #6B6B6B;
        cursor: pointer;
        padding-bottom: 2px;
        transition: color 0.15s ease;
    }
    .cat-text-item.active {
        color: #171717;
        font-weight: 600;
        border-bottom: 1.5px solid #171717;
    }

    /* Result Cards & Panels */
    .result-panel-card {
        background: #FFFFFF;
        border: 1px solid #E5E5E2;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .result-panel-header {
        padding: 8px 12px;
        border-bottom: 1px solid #F0F0EE;
        font-size: 0.70rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #6B6B6B;
        text-transform: uppercase;
        display: flex;
        justify-content: space-between;
    }

    /* Minimal Info Strip */
    .metric-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E5E5E2;
        border-radius: 6px;
        padding: 14px 20px;
        margin: 16px 0;
        flex-wrap: wrap;
        gap: 16px;
    }
    .metric-item {
        display: flex;
        flex-direction: column;
    }
    .metric-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #8E8E93;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #171717;
        line-height: 1.1;
    }

    /* Minimal Legend */
    .minimal-legend {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.68rem;
        color: #8E8E93;
        margin-top: 6px;
        padding: 0 4px;
    }
    .legend-gradient {
        flex: 1;
        height: 4px;
        margin: 0 10px;
        border-radius: 2px;
        background: linear-gradient(to right, #000080 0%, #00FFFF 35%, #FFFF00 70%, #FF0000 100%);
    }

    /* Minimalist upload zone */
    .upload-zone-frame {
        background: #FFFFFF;
        border: 1px dashed #D4D4D0;
        border-radius: 8px;
        padding: 3rem 1.5rem;
        text-align: center;
        transition: border-color 0.15s ease;
    }
</style>
""")

# -----------------------------------------------------------------------------
# MINIMALIST NAVBAR
# -----------------------------------------------------------------------------
nav_col1, nav_col2, nav_col3 = st.columns([1.8, 3.4, 1.4])

with nav_col1:
    render_html("""
    <div style="padding-top: 4px;">
        <span class="vi-brand"><span class="vi-brand-dot"></span> VisionInspect</span>
    </div>
    """)

with nav_col2:
    pages = ["Home", "Inspect", "Models", "About"]
    p_cols = st.columns(len(pages))
    for idx, p in enumerate(pages):
        with p_cols[idx]:
            is_active = (st.session_state["nav_page"] == p)
            btn_label = f"• {p}" if is_active else p
            if st.button(btn_label, key=f"nav_btn_{p}", use_container_width=True):
                st.session_state["nav_page"] = p
                st.rerun()

with nav_col3:
    if st.button("Start Inspection", key="nav_start_btn", type="primary", use_container_width=True):
        st.session_state["nav_page"] = "Inspect"
        st.rerun()

st.markdown("<hr style='margin: 0.2rem 0 1.4rem 0; border: none; border-bottom: 1px solid #E5E5E2;' />", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: HOME PAGE (MINIMALIST HERO COMPOSITION)
# =============================================================================
if st.session_state["nav_page"] == "Home":
    active_cat = st.session_state["selected_category"]
    cat_data = CATEGORIES[active_cat]

    hero_left, hero_right = st.columns([1.05, 1.25], gap="large")

    with hero_left:
        render_html("""
        <div class="hero-eyebrow">INDUSTRIAL AI INSPECTION</div>
        <h1 class="hero-heading">See Defects.<br>Understand Them.</h1>
        <p class="hero-sub">AI-powered visual inspection for detecting and localizing manufacturing defects with sub-pixel precision.</p>
        """)

        # Clean Hero Action Buttons
        act_col1, act_col2 = st.columns([2.0, 1.0])
        with act_col1:
            if st.button("Start Inspection", type="primary", key="home_hero_start", use_container_width=True):
                st.session_state["nav_page"] = "Inspect"
                st.rerun()
        with act_col2:
            if st.button("→", key="home_hero_arrow", use_container_width=True):
                st.session_state["nav_page"] = "Inspect"
                st.rerun()

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Minimal Category Selector
        render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #8E8E93; text-transform: uppercase;'>PRODUCT</div>")
        c_cols = st.columns(5)
        for idx, (ckey, cval) in enumerate(CATEGORIES.items()):
            with c_cols[idx]:
                is_selected = (active_cat == ckey)
                btn_name = f"✔ {cval['name']}" if is_selected else cval["name"]
                if st.button(btn_name, key=f"hero_cat_{ckey}", use_container_width=True):
                    if active_cat != ckey:
                        purge_inspection(new_category=ckey)
                        st.rerun()

        # Small Subtle Technical Information Area
        render_html("""
        <div style="margin-top: 36px; padding-top: 14px; border-top: 1px solid #E5E5E2; font-size: 0.74rem; color: #8E8E93; line-height: 1.6;">
            <b>PatchCore v2.3</b> &nbsp;•&nbsp; Unsupervised anomaly detection &nbsp;•&nbsp; ResNet-18 &nbsp;•&nbsp; 448D features<br>
            Trained on defect-free samples. Designed to detect deviations from normal structure.
        </div>
        """)

    with hero_right:
        # Dominant Hero Product Visual
        sample_path = Path(cat_data["golden_sample"])
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                img_bytes = f.read()
            b64_sample = base64.b64encode(img_bytes).decode("utf-8")
        else:
            b64_sample = ""

        render_html(f"""
        <div class="hero-visual-frame">
            <div class="corner-bracket corner-tl"></div>
            <div class="corner-bracket corner-tr"></div>
            <div class="corner-bracket corner-bl"></div>
            <div class="corner-bracket corner-br"></div>
            <div class="visual-top-meta">
                <span>VISIONINSPECT // OPTICAL ACQUISITION</span>
                <span>● READY</span>
            </div>
            <img src="data:image/png;base64,{b64_sample}" style="max-height: 380px; max-width: 90%; object-fit: contain; margin: 28px 0;" />
            <div class="visual-bottom-meta">
                <span>CATEGORY: {cat_data['name'].upper()}</span>
                <span>REF: NOMINAL GOLDEN SAMPLE</span>
            </div>
        </div>
        """)


# =============================================================================
# PAGE 2: INSPECT PAGE (CLEAN WORKFLOW: SELECT -> UPLOAD -> ANALYZE)
# =============================================================================
elif st.session_state["nav_page"] == "Inspect":
    active_cat = st.session_state["selected_category"]
    cat_data = CATEGORIES[active_cat]

    # Sub-header bar
    top_nav1, top_nav2 = st.columns([1.2, 3.8])
    with top_nav1:
        if st.button("← Back to Overview", key="inspect_back_btn"):
            st.session_state["nav_page"] = "Home"
            st.rerun()

    with top_nav2:
        # Category Selector in header
        c_cols = st.columns(5)
        for idx, (ckey, cval) in enumerate(CATEGORIES.items()):
            with c_cols[idx]:
                is_selected = (active_cat == ckey)
                btn_name = f"✔ {cval['name']}" if is_selected else cval["name"]
                if st.button(btn_name, key=f"inspect_cat_{ckey}", use_container_width=True):
                    if active_cat != ckey:
                        purge_inspection(new_category=ckey)
                        st.rerun()

    st.markdown("<hr style='margin: 0.8rem 0 1.2rem 0; border: none; border-bottom: 1px solid #E5E5E2;' />", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STATE A: SCANNING IN PROGRESS
    # -------------------------------------------------------------------------
    if st.session_state["is_scanning"] and st.session_state["uploaded_image_data"]:
        fname, scan_bytes, _ = st.session_state["uploaded_image_data"]
        b64_scan = base64.b64encode(scan_bytes).decode("utf-8")

        render_html("""
        <div style="text-align: center; margin-bottom: 14px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #EF4444; text-transform: uppercase;">SCANNING IN PROGRESS</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #171717; margin-top: 4px;">Analyzing Industrial Component</div>
        </div>
        """)

        render_html(f"""
        <div class="scan-container">
            <div class="corner-bracket corner-tl"></div>
            <div class="corner-bracket corner-tr"></div>
            <div class="corner-bracket corner-bl"></div>
            <div class="corner-bracket corner-br"></div>
            <div class="scan-laser"></div>
            <img src="data:image/png;base64,{b64_scan}" style="width: 100%; display: block; filter: contrast(1.02);" />
        </div>
        <div style="text-align: center; margin-top: 14px; font-size: 0.78rem; color: #6B6B6B;">
            Extracting ResNet-18 Layers 1–3 (448D) • Querying {cat_data['name']} Coreset Memory Bank • Localizing Anomaly Map
        </div>
        """)

        # Execute Backend Prediction
        try:
            res = requests.post(
                f"{st.session_state['backend_url']}/predict",
                data={"category": active_cat, "return_visualizations": "true"},
                files={"file": (fname, scan_bytes, "image/png")},
                timeout=25
            )
            if res.status_code == 200:
                data = res.json()
                st.session_state["current_inspection"] = data
                st.session_state["current_request_id"] = data.get("request_id")
                # Log history
                st.session_state["recent_inspections"].insert(0, {
                    "id": len(st.session_state["recent_inspections"]) + 1,
                    "filename": fname,
                    "category": active_cat,
                    "status": data.get("status", "NORMAL"),
                    "score": data.get("anomaly_score", 0.0),
                    "threshold": data.get("image_threshold", 0.0),
                    "regions": data.get("num_defects", 0),
                    "time_s": data.get("inference_time_s", 1.5)
                })
            else:
                st.error(f"Inference error: {res.status_code} - {res.text}")
        except Exception as e:
            st.error(f"Backend connection error: {e}")
        finally:
            st.session_state["is_scanning"] = False
            st.rerun()

    # -------------------------------------------------------------------------
    # STATE B: INSPECTION RESULT VIEW (FULL-WIDTH, HIGH-IMPACT)
    # -------------------------------------------------------------------------
    elif st.session_state["current_inspection"] is not None:
        res = st.session_state["current_inspection"]
        is_defective = res.get("is_defective", False)
        status = res.get("status", "NORMAL")
        score = res.get("anomaly_score", 0.0)
        th = res.get("image_threshold", 0.0)
        margin = res.get("decision_margin", score - th)
        n_defects = res.get("num_defects", 0)
        latency_ms = res.get("inference_time_ms", 180.0)
        explanation = res.get("explanation", "")
        why_explanation = res.get("why_explanation", "")

        # Status Header
        res_h1, res_h2 = st.columns([3.5, 1.2])
        with res_h1:
            render_html("<div style='font-size: 0.72rem; font-weight: 600; letter-spacing: 0.10em; color: #8E8E93; text-transform: uppercase;'>INSPECTION RESULT</div>")
            if is_defective:
                render_html(f"""
                <div style="font-size: 2.2rem; font-weight: 800; color: #B91C1C; letter-spacing: -0.02em; line-height: 1.1;">DEFECTIVE</div>
                <div style="font-size: 0.95rem; color: #404040; margin-top: 4px;">{explanation}</div>
                """)
            elif score > th and n_defects == 0:
                render_html(f"""
                <div style="font-size: 2.2rem; font-weight: 800; color: #B45309; letter-spacing: -0.02em; line-height: 1.1;">NORMAL (ELEVATED SIGNAL)</div>
                <div style="font-size: 0.95rem; color: #404040; margin-top: 4px;">{explanation}</div>
                """)
            else:
                render_html(f"""
                <div style="font-size: 2.2rem; font-weight: 800; color: #15803D; letter-spacing: -0.02em; line-height: 1.1;">NORMAL</div>
                <div style="font-size: 0.95rem; color: #404040; margin-top: 4px;">{explanation}</div>
                """)

        with res_h2:
            if st.button("Inspect Another Component", key="inspect_another_btn", use_container_width=True):
                purge_inspection()
                st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # MAIN RESULT VISUAL (HUGE THREE-PANEL SECTION)
        # ---------------------------------------------------------------------
        v_col1, v_col2, v_col3 = st.columns(3, gap="medium")

        # 01 ORIGINAL
        with v_col1:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; margin-bottom: 6px;'>01 &nbsp; ORIGINAL</div>")
            if st.session_state["uploaded_image_data"]:
                _, orig_bytes, _ = st.session_state["uploaded_image_data"]
                st.image(Image.open(io.BytesIO(orig_bytes)), use_container_width=True)
            render_html("<div style='font-size: 0.68rem; color: #8E8E93; margin-top: 4px;'>Pristine optical sensor acquisition</div>")

        # 02 ANOMALY MAP
        with v_col2:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; margin-bottom: 6px;'>02 &nbsp; ANOMALY MAP</div>")
            hm_b64 = res.get("heatmap_base64")
            if hm_b64:
                st.image(Image.open(io.BytesIO(base64.b64decode(hm_b64))), use_container_width=True)
                render_html("""
                <div class="minimal-legend">
                    <span>Nominal</span>
                    <div class="legend-gradient"></div>
                    <span>Anomaly</span>
                </div>
                """)
            else:
                st.info("Anomaly map not available.")

        # 03 LOCALIZATION
        with v_col3:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; margin-bottom: 6px;'>03 &nbsp; LOCALIZATION</div>")
            vis_b64 = res.get("visualization_base64")
            if is_defective and vis_b64:
                st.image(Image.open(io.BytesIO(base64.b64decode(vis_b64))), use_container_width=True)
                render_html(f"<div style='font-size: 0.68rem; color: #B91C1C; margin-top: 4px; font-weight: 600;'>{n_defects} confirmed defect region(s) localized</div>")
            else:
                if st.session_state["uploaded_image_data"]:
                    _, orig_bytes, _ = st.session_state["uploaded_image_data"]
                    st.image(Image.open(io.BytesIO(orig_bytes)), use_container_width=True)
                render_html("<div style='font-size: 0.68rem; color: #15803D; margin-top: 4px; font-weight: 600;'>✔ Defect-free — Zero anomaly clusters detected</div>")

        # ---------------------------------------------------------------------
        # MINIMAL INFORMATION STRIP
        # ---------------------------------------------------------------------
        margin_sign = f"+{margin:.4f}" if margin > 0 else f"{margin:.4f}"
        margin_color = "#B91C1C" if margin > 0 else "#15803D"

        render_html(f"""
        <div class="metric-strip">
            <div class="metric-item">
                <span class="metric-label">Anomaly Score</span>
                <span class="metric-value">{score:.4f}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Threshold</span>
                <span class="metric-value">{th:.4f}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Decision Margin</span>
                <span class="metric-value" style="color: {margin_color};">{margin_sign}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Defect Regions</span>
                <span class="metric-value">{n_defects}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">Inference Latency</span>
                <span class="metric-value">{int(latency_ms)} ms</span>
            </div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # WHY THIS RESULT? (ENGINEERING EXPLANATION)
        # ---------------------------------------------------------------------
        render_html(f"""
        <div style="background: #FFFFFF; border: 1px solid #E5E5E2; border-radius: 6px; padding: 16px 20px; margin-bottom: 14px;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; text-transform: uppercase; margin-bottom: 4px;">WHY THIS RESULT?</div>
            <div style="font-size: 0.88rem; color: #171717; line-height: 1.5;">{why_explanation}</div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # DETECTED REGIONS BREAKDOWN (IF DEFECTIVE)
        # ---------------------------------------------------------------------
        if is_defective and res.get("localized_regions"):
            regs = res.get("localized_regions", [])
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; text-transform: uppercase; margin: 12px 0 6px 0;'>LOCALIZED ANOMALY REGIONS</div>")
            reg_cols = st.columns(min(4, len(regs)))
            for r_idx, reg in enumerate(regs[:4]):
                with reg_cols[r_idx]:
                    lbl = reg.get("label", f"Region {r_idx+1:02d}")
                    intensity = reg.get("intensity", "Anomaly Region")
                    area_px = reg.get("area", 0)
                    r_score = reg.get("score", 0.0)
                    render_html(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E5E5E2; border-radius: 6px; padding: 10px 14px; font-size: 0.78rem;">
                        <div style="font-weight: 700; color: #B91C1C;">{lbl}</div>
                        <div style="color: #6B6B6B; margin-top: 2px;">{intensity}</div>
                        <div style="font-size: 0.72rem; color: #8E8E93; margin-top: 4px;">Area: {area_px} px • Score: {r_score:.4f}</div>
                    </div>
                    """)

        # ---------------------------------------------------------------------
        # COLLAPSIBLE TECHNICAL DETAILS (FOR EXAMINER / VIVA)
        # ---------------------------------------------------------------------
        with st.expander("Technical Details", expanded=False):
            st.markdown(f"""
            - **Architecture:** PatchCore v2.3 (Unsupervised Density-based Anomaly Localization)
            - **Backbone:** ResNet-18 (Layers 1, 2, 3 Multi-Scale Embeddings)
            - **Descriptor Dimension:** 448-dimensional patch representations on a 64×64 spatial grid
            - **Category Model Directory:** `models/{active_cat}/patchcore_v23/`
            - **Calibrated Image Threshold ($T_{{image}}$):** `{th:.4f}`
            - **Calibrated Pixel Threshold ($T_{{pixel}}$):** `{res.get('pixel_threshold', 0.0):.4f}`
            - **Decision Rule:** Dual-Gated $[Score > T_{{image}}] \\land [Defect Area \\ge min\\_area]$
            - **Unique Request ID:** `{res.get('request_id', 'N/A')}`
            """)

    # -------------------------------------------------------------------------
    # STATE C: IMAGE INPUT SELECTION (UPLOAD OR VERIFIED SAMPLE)
    # -------------------------------------------------------------------------
    else:
        # If an image has been selected/uploaded, display it prominently
        if st.session_state["uploaded_image_data"]:
            fname, img_bytes, (iw, ih) = st.session_state["uploaded_image_data"]
            b64_loaded = base64.b64encode(img_bytes).decode("utf-8")

            col_img, col_actions = st.columns([1.3, 1.0], gap="large")

            with col_img:
                render_html(f"""
                <div class="hero-visual-frame" style="min-height: 380px;">
                    <div class="corner-bracket corner-tl"></div>
                    <div class="corner-bracket corner-tr"></div>
                    <div class="corner-bracket corner-bl"></div>
                    <div class="corner-bracket corner-br"></div>
                    <div class="visual-top-meta">
                        <span>COMPONENT: {cat_data['name'].upper()}</span>
                        <span>{iw}×{ih} PX</span>
                    </div>
                    <img src="data:image/png;base64,{b64_loaded}" style="max-height: 320px; max-width: 90%; object-fit: contain; margin: 24px 0;" />
                    <div class="visual-bottom-meta">
                        <span>FILE: {fname}</span>
                        <span>READY TO INSPECT</span>
                    </div>
                </div>
                """)

            with col_actions:
                render_html("""
                <div style="padding-top: 24px;">
                    <div style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.10em; color: #6B6B6B; text-transform: uppercase;">READY FOR INSPECTION</div>
                    <h2 style="font-size: 1.8rem; font-weight: 700; color: #171717; margin: 4px 0 12px 0;">Optical Acquisition Ready</h2>
                    <p style="font-size: 0.88rem; color: #6B6B6B; line-height: 1.5; margin-bottom: 24px;">
                        The image has been loaded into memory. Click below to execute deep feature comparison against the nominal memory bank.
                    </p>
                </div>
                """)

                if st.button("ANALYZE IMAGE →", key="analyze_btn", type="primary", use_container_width=True):
                    st.session_state["is_scanning"] = True
                    st.rerun()

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                if st.button("Clear / Choose Different Image", key="clear_img_btn", use_container_width=True):
                    purge_inspection()
                    st.rerun()

        # No image selected yet: Show minimal input options
        else:
            in_col1, in_col2 = st.columns([1.1, 1.0], gap="large")

            with in_col1:
                render_html("""
                <div style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.10em; color: #6B6B6B; text-transform: uppercase; margin-bottom: 8px;">INPUT OPTION 01</div>
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #171717; margin: 0 0 14px 0;">Upload Component Image</h3>
                """)

                uploaded_file = st.file_uploader(
                    "Drop an image here or click to browse",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="clean_uploader",
                    label_visibility="visible"
                )

                if uploaded_file is not None:
                    raw_bytes = uploaded_file.getvalue()
                    pil_im = Image.open(io.BytesIO(raw_bytes))
                    st.session_state["uploaded_image_data"] = (uploaded_file.name, raw_bytes, pil_im.size)
                    st.session_state["current_inspection"] = None
                    st.session_state["current_request_id"] = None
                    st.rerun()

            with in_col2:
                render_html("""
                <div style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.10em; color: #6B6B6B; text-transform: uppercase; margin-bottom: 8px;">INPUT OPTION 02</div>
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #171717; margin: 0 0 14px 0;">Use Verified Demo Sample</h3>
                """)

                sample_labels = [s[0] for s in cat_data["samples"]]
                chosen_sample = st.selectbox(
                    f"Select verified {cat_data['name']} evaluation sample:",
                    options=["-- Select a sample --"] + sample_labels,
                    key="sample_select_box"
                )

                if chosen_sample != "-- Select a sample --":
                    matched_path = None
                    for s_label, s_path in cat_data["samples"]:
                        if s_label == chosen_sample:
                            matched_path = s_path
                            break
                    if matched_path and Path(matched_path).exists():
                        with open(matched_path, "rb") as f:
                            s_bytes = f.read()
                        pil_im = Image.open(io.BytesIO(s_bytes))
                        st.session_state["uploaded_image_data"] = (Path(matched_path).name, s_bytes, pil_im.size)
                        st.session_state["current_inspection"] = None
                        st.session_state["current_request_id"] = None
                        st.rerun()

    # -------------------------------------------------------------------------
    # RECENT INSPECTIONS (MINIMAL LIST)
    # -------------------------------------------------------------------------
    if st.session_state["recent_inspections"]:
        st.markdown("<hr style='margin: 2.0rem 0 1.2rem 0; border: none; border-bottom: 1px solid #E5E5E2;' />", unsafe_allow_html=True)
        render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #8E8E93; text-transform: uppercase; margin-bottom: 8px;'>RECENT INSPECTIONS (SESSION)</div>")
        
        hist_rows = []
        for item in st.session_state["recent_inspections"][:4]:
            badge_color = "#B91C1C" if item["status"] == "DEFECTIVE" else "#15803D"
            hist_rows.append(
                f"<div style='display:flex; justify-content:space-between; padding: 6px 0; border-bottom: 1px solid #F0F0EE; font-size: 0.80rem;'>"
                f"<span><b>#{item['id']}</b> &nbsp; {item['category'].title()} &nbsp;•&nbsp; <code>{item['filename'][:20]}</code></span>"
                f"<span style='color:{badge_color}; font-weight:700;'>{item['status']}</span>"
                f"<span>Score: {item['score']:.4f}</span>"
                f"<span>Latency: {item['time_s']:.2f}s</span>"
                f"</div>"
            )
        render_html(f"<div style='background:#FFFFFF; border:1px solid #E5E5E2; border-radius:6px; padding:10px 16px;'>{''.join(hist_rows)}</div>")


# =============================================================================
# PAGE 3: MODELS PAGE (MINIMALIST ARCHITECTURE & METHODOLOGY)
# =============================================================================
elif st.session_state["nav_page"] == "Models":
    render_html("""
    <div class="hero-eyebrow">SYSTEM ARCHITECTURE</div>
    <h1 style="font-size: 2.2rem; font-weight: 700; color: #171717; margin: 0 0 12px 0;">PatchCore v2.3 Architecture</h1>
    <p style="font-size: 0.95rem; color: #6B6B6B; max-width: 680px; line-height: 1.5; margin-bottom: 24px;">
        VisionInspect operates on an unsupervised memory bank density paradigm. Feature representations are extracted from deep ResNet-18 layers, sub-sampled via greedy coreset selection, and tested without training on defect images.
    </p>
    """)

    m_col1, m_col2 = st.columns(2, gap="large")

    with m_col1:
        render_html("""
        <div style="background: #FFFFFF; border: 1px solid #E5E5E2; border-radius: 6px; padding: 20px; height: 100%;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; text-transform: uppercase; margin-bottom: 10px;">CORE PIPELINE STAGES</div>
            <div style="font-size: 0.82rem; color: #171717; line-height: 1.8;">
                <b>1. Multi-Scale Feature Extraction</b><br>
                ResNet-18 Layers 1, 2, and 3 are extracted and spatially bilinearly interpolated onto a uniform 64×64 grid, producing 448-dimensional localized patch descriptors.<br><br>
                <b>2. Coreset Memory Bank Subsampling</b><br>
                Greedy minimax coreset selection retains 10% of nominal feature vectors while maintaining complete coverage of the normal representation manifold.<br><br>
                <b>3. Nearest-Neighbor Anomaly Scoring</b><br>
                Test image patches are scored via exact Euclidean nearest-neighbor distance against the category's nominal memory bank.<br><br>
                <b>4. Dual-Gated Decision Rule</b><br>
                A component is defective if and only if its anomaly score exceeds T_image AND a morphological connected component exceeds min_area.
            </div>
        </div>
        """)

    with m_col2:
        render_html("""
        <div style="background: #FFFFFF; border: 1px solid #E5E5E2; border-radius: 6px; padding: 20px; height: 100%;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: #6B6B6B; text-transform: uppercase; margin-bottom: 10px;">CATEGORY MODELS & THRESHOLDS</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.80rem; margin-top: 8px;">
                <thead>
                    <tr style="border-bottom: 1px solid #E5E5E2; text-align: left; color: #8E8E93;">
                        <th style="padding: 6px 0;">Category</th>
                        <th>T_image</th>
                        <th>T_pixel</th>
                        <th>Spatial Prior</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #F0F0EE;">
                        <td style="padding: 8px 0; font-weight: 600;">Bottle</td>
                        <td>1.50</td>
                        <td>1.40</td>
                        <td>Active (p75/mean)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F0F0EE;">
                        <td style="padding: 8px 0; font-weight: 600;">Leather</td>
                        <td>2.20</td>
                        <td>2.71</td>
                        <td>Disabled (Texture)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F0F0EE;">
                        <td style="padding: 8px 0; font-weight: 600;">Transistor</td>
                        <td>3.525</td>
                        <td>2.822</td>
                        <td>Presence Gate</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #F0F0EE;">
                        <td style="padding: 8px 0; font-weight: 600;">Zipper</td>
                        <td>1.47</td>
                        <td>0.91</td>
                        <td>Active (mean)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 600;">Screw</td>
                        <td>2.28</td>
                        <td>2.00</td>
                        <td>Active (mean)</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """)


# =============================================================================
# PAGE 4: ABOUT PAGE (ACADEMIC CONTEXT & VIVA PREPARATION)
# =============================================================================
elif st.session_state["nav_page"] == "About":
    render_html("""
    <div class="hero-eyebrow">ABOUT VISIONINSPECT</div>
    <h1 style="font-size: 2.2rem; font-weight: 700; color: #171717; margin: 0 0 12px 0;">Autonomous Quality Inspection</h1>
    <p style="font-size: 0.95rem; color: #6B6B6B; max-width: 680px; line-height: 1.5; margin-bottom: 24px;">
        Designed for industrial manufacturing lines where defective samples are scarce or unavailable. VisionInspect models normal product geometry and flags any statistical departure from perfection.
    </p>
    """)

    st.markdown("### Examination & Viva Guide")
    with st.expander("Q1: Why PatchCore over Convolutional Autoencoders?", expanded=True):
        st.write("""
        Autoencoders attempt to compress and reconstruct images through an information bottleneck. In practice, they often suffer from blurry reconstructions and 'shortcut learning', where subtle micro-cracks or missing pins are accidentally reconstructed, producing false negatives.
        
        PatchCore circumvents reconstruction entirely. It evaluates semantic patch representations directly against a golden coreset memory bank of nominal features, preserving high spatial resolution and achieving >99% AUROC on rigid industrial components.
        """)

    with st.expander("Q2: How was zero test leakage guaranteed?", expanded=False):
        st.write("""
        All image thresholds, pixel thresholds, and morphological post-processing parameters were calibrated exclusively on a held-out 20% validation split of nominal training images, constrained strictly to false positive rate ≤ 3.5%. The MVTec test set was never accessed during calibration.
        """)

    with st.expander("Q3: What is the Dual-Gated Decision Rule?", expanded=False):
        st.write("""
        An image is classified as DEFECTIVE if and only if:
        1. The overall image anomaly score exceeds the calibrated category threshold (T_image).
        2. At least one connected component in the anomaly map exceeds the minimum defect area (min_area) after morphology filtering.
        
        This prevents isolated sensor noise or dust particles from generating false alarms.
        """)
