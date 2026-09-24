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
if "home_hero_selection" not in st.session_state:
    st.session_state["home_hero_selection"] = "Carton Box"
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
    cleaned_lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    cleaned_content = "\n".join(cleaned_lines)
    st.markdown(cleaned_content, unsafe_allow_html=True)

# Cached thumbnail helper for instant component catalog rendering
@st.cache_data
def get_thumbnail_b64(img_path: str) -> str:
    p = Path(img_path)
    if not p.exists():
        return ""
    try:
        with Image.open(p) as im:
            im_rgb = im.convert("RGB")
            im_rgb.thumbnail((96, 96), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im_rgb.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""

# -----------------------------------------------------------------------------
# CENTRALIZED DESIGN SYSTEM (CSS VARIABLES & ATOMIC TOKENS)
# -----------------------------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F5F5F3;
        --surface: #FFFFFF;
        --surface-subtle: #FAFAF8;
        --text-primary: #171717;
        --text-secondary: #5F6368;
        --text-muted: #7A7A7A;
        --border: #E4E4E0;
        --border-strong: #D4D4CF;
        --dark: #171717;
        --dark-hover: #2A2A2A;
        --success: #15803D;
        --success-bg: #F0FDF4;
        --danger: #B91C1C;
        --danger-bg: #FEF2F2;
        --warning: #A16207;
        --warning-bg: #FFFBEB;
    }

    /* Global Chrome Removal & App Resets */
    #MainMenu, header, footer, .stDeployButton {
        display: none !important;
        visibility: hidden !important;
    }

    html, body, [class*="css"], .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
        background-color: var(--bg) !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em;
    }

    /* Centered Layout Container (1280-1400px with 32px desktop padding) */
    .block-container {
        max-width: 1320px !important;
        width: calc(100% - 64px) !important;
        padding-left: 32px !important;
        padding-right: 32px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 2.5rem !important;
        margin: 0 auto !important;
    }

    /* Standardized Button System */
    div.stButton > button {
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.84rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.15s ease !important;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        box-shadow: none !important;
        cursor: pointer !important;
    }
    div.stButton > button:hover {
        background-color: var(--surface-subtle) !important;
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--dark) !important;
        color: var(--surface) !important;
        border: 1px solid var(--dark) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--dark-hover) !important;
        border-color: var(--dark-hover) !important;
        color: var(--surface) !important;
    }

    /* Prevent Button / Pill Text Clipping */
    div.stButton > button,
    div.stButton > button p,
    div.stButton > button span,
    [data-testid="stPills"] button,
    [data-testid="stPills"] button p,
    [data-testid="stPills"] button span {
        white-space: nowrap !important;
        text-overflow: clip !important;
        overflow: visible !important;
    }

    /* Minimalist Apple-style Pills */
    [data-testid="stPills"] {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        margin-top: 4px !important;
        margin-bottom: 8px !important;
    }
    [data-testid="stPills"] button {
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
        color: var(--text-secondary) !important;
        padding: 5px 14px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stPills"] button:hover {
        border-color: var(--border-strong) !important;
        color: var(--text-primary) !important;
        background-color: var(--surface-subtle) !important;
    }
    [data-testid="stPills"] button[aria-selected="true"] {
        background-color: var(--dark) !important;
        color: var(--surface) !important;
        border-color: var(--dark) !important;
        font-weight: 600 !important;
    }

    /* Clean Selectbox & Popover Styling */
    div[data-testid="stSelectbox"] > div {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stSelectbox"] * {
        color: var(--text-primary) !important;
    }
    div[data-testid="stSelectbox"] svg {
        fill: var(--text-primary) !important;
    }
    div[data-baseweb="select"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }
    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06) !important;
    }
    li[role="option"] {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
    }
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {
        background-color: var(--surface-subtle) !important;
        color: var(--text-primary) !important;
    }

    /* File Uploader Clean Styling */
    div[data-testid="stFileUploader"] {
        background-color: var(--surface) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 6px !important;
        padding: 1rem !important;
    }
    div[data-testid="stFileUploader"] section {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] div,
    div[data-testid="stFileUploader"] p {
        color: var(--text-secondary) !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: var(--surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
    }

    /* Hero Typography */
    .hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }
    .hero-heading {
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1.06;
        letter-spacing: -0.035em;
        color: var(--text-primary);
        margin: 0 0 14px 0;
    }
    .hero-sub {
        font-size: 1.0rem;
        line-height: 1.55;
        color: var(--text-secondary);
        max-width: 440px;
        margin-bottom: 22px;
        font-weight: 400;
    }

    /* Central Hero Visual Container */
    .hero-visual-frame {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 400px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
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
        left: 28px;
        right: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    .visual-bottom-meta {
        position: absolute;
        bottom: 14px;
        left: 28px;
        right: 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.68rem;
        color: var(--text-secondary);
    }

    /* Scanning Container & Laser */
    .scan-container {
        position: relative;
        width: 100%;
        max-width: 480px;
        margin: 0 auto;
        background: var(--surface);
        border: 1px solid var(--border);
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

    /* Metric Strip */
    .metric-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 14px 24px;
        margin: 16px 0;
        flex-wrap: wrap;
        gap: 16px;
    }
    .metric-item {
        display: flex;
        flex-direction: column;
    }
    .metric-label {
        font-size: 0.70rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }

    /* Heatmap Legend */
    .minimal-legend {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.70rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-top: 6px;
        padding: 0 4px;
    }
    .legend-gradient {
        flex: 1;
        height: 6px;
        margin: 0 10px;
        border-radius: 3px;
        background: linear-gradient(to right, #000080 0%, #00FFFF 35%, #FFFF00 70%, #FF0000 100%);
    }

    /* Plain Text Navigation Links */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stHorizontalBlock"] button {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        border-bottom: 2px solid transparent !important;
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        padding: 6px 14px !important;
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stHorizontalBlock"] button:hover {
        color: var(--text-primary) !important;
        background: transparent !important;
        border-bottom: 2px solid var(--border-strong) !important;
    }
</style>
""")

# -----------------------------------------------------------------------------
# MINIMALIST NAVBAR (NO BULLETS, PLAIN TEXT WITH ACTIVE UNDERLINE)
# -----------------------------------------------------------------------------
nav_col1, nav_col2, nav_col3 = st.columns([1.8, 3.4, 1.4])

with nav_col1:
    render_html("""
    <div style="padding-top: 4px;">
        <span style="font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text-primary); display: flex; align-items: center; gap: 8px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: var(--text-primary); display: inline-block;"></span>
            VisionInspect
        </span>
    </div>
    """)

with nav_col2:
    pages = ["Home", "Inspect", "Models", "About"]
    p_cols = st.columns(len(pages))
    for idx, p in enumerate(pages):
        with p_cols[idx]:
            is_active = (st.session_state["nav_page"] == p)
            if is_active:
                render_html(f"""
                <style>
                    div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stHorizontalBlock"] > div:nth-child({idx+1}) button {{
                        border-bottom: 2px solid var(--dark) !important;
                        color: var(--text-primary) !important;
                        font-weight: 600 !important;
                    }}
                </style>
                """)
            if st.button(p, key=f"nav_btn_{p}", use_container_width=True):
                st.session_state["nav_page"] = p
                st.rerun()

with nav_col3:
    if st.button("New Inspection →", key="nav_start_btn", type="primary", use_container_width=True):
        st.session_state["nav_page"] = "Inspect"
        purge_inspection()
        st.rerun()

st.markdown("<hr style='margin: 0.2rem 0 1.2rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)


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

        # Clean Hero Action Button
        if st.button("Start Inspection →", type="primary", key="home_hero_start_btn", use_container_width=False):
            st.session_state["nav_page"] = "Inspect"
            st.rerun()

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Minimal Category Selector with FULL Names (No Truncation)
        render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 4px;'>PRODUCT CATEGORY</div>")
        home_product_options = ["Carton Box", "Bottle", "Leather", "Transistor", "Zipper", "Screw"]
        cur_selection = st.session_state.get("home_hero_selection", "Carton Box")
        if cur_selection not in home_product_options:
            cur_selection = "Carton Box"

        selected_pill = st.pills(
            "Product",
            options=home_product_options,
            default=cur_selection,
            key="home_hero_pills_widget",
            label_visibility="collapsed"
        )

        if selected_pill and selected_pill != cur_selection:
            st.session_state["home_hero_selection"] = selected_pill
            if selected_pill.lower() in CATEGORIES:
                st.session_state["selected_category"] = selected_pill.lower()
                purge_inspection(new_category=selected_pill.lower())
            st.rerun()

        # Small Subtle Technical Information Area
        render_html("""
        <div style="margin-top: 32px; padding-top: 14px; border-top: 1px solid var(--border); font-size: 0.75rem; color: var(--text-secondary); line-height: 1.6;">
            <span style="font-weight: 600; color: var(--text-primary);">PatchCore v2.3</span> &nbsp;•&nbsp; Unsupervised anomaly detection &nbsp;•&nbsp; ResNet-18 &nbsp;•&nbsp; 448D features<br>
            Trained exclusively on defect-free samples. Calibrated to detect structural and surface departures from normal manifold.
        </div>
        """)

    with hero_right:
        cur_sel = st.session_state.get("home_hero_selection", "Carton Box")

        if cur_sel == "Carton Box":
            carton_path = Path("assets/carton_box_scan.jpg")
            if carton_path.exists():
                with open(carton_path, "rb") as f:
                    b64_sample = base64.b64encode(f.read()).decode("utf-8")
                img_mime = "image/jpeg"
            else:
                b64_sample = ""
                img_mime = "image/png"

            render_html(f"""
            <div class="hero-visual-frame">
                <div class="corner-bracket corner-tl"></div>
                <div class="corner-bracket corner-tr"></div>
                <div class="corner-bracket corner-bl"></div>
                <div class="corner-bracket corner-br"></div>
                <div class="visual-top-meta">
                    <span>VISIONINSPECT // OPTICAL ACQUISITION</span>
                    <span style="color: var(--danger); font-weight: 700;">● DEFECT DETECTED</span>
                </div>
                <img src="data:{img_mime};base64,{b64_sample}" style="max-height: 380px; max-width: 90%; object-fit: contain; margin: 28px 0; border-radius: 4px;" />
                <div class="visual-bottom-meta">
                    <span>TARGET: INDUSTRIAL PACKAGING (CARTON BOX)</span>
                    <span style="color: var(--danger); font-weight: 600;">DEFECT: CRUSHED FLAP / DAMAGE</span>
                </div>
            </div>
            """)
        else:
            cat_key = cur_sel.lower()
            cat_info = CATEGORIES.get(cat_key, CATEGORIES["bottle"])
            sample_path = Path(cat_info["golden_sample"])
            if sample_path.exists():
                with open(sample_path, "rb") as f:
                    b64_sample = base64.b64encode(f.read()).decode("utf-8")
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
                    <span style="color: var(--success); font-weight: 600;">● READY</span>
                </div>
                <img src="data:image/png;base64,{b64_sample}" style="max-height: 380px; max-width: 90%; object-fit: contain; margin: 28px 0;" />
                <div class="visual-bottom-meta">
                    <span>CATEGORY: {cat_info['name'].upper()}</span>
                    <span>REF: NOMINAL GOLDEN SAMPLE</span>
                </div>
            </div>
            """)


# =============================================================================
# PAGE 2: INSPECT PAGE (3-COLUMN WORKFLOW + RESULT INSPECTION REPORT)
# =============================================================================
elif st.session_state["nav_page"] == "Inspect":
    active_cat = st.session_state["selected_category"]
    cat_data = CATEGORIES[active_cat]

    # -------------------------------------------------------------------------
    # STATE A: SCANNING IN PROGRESS
    # -------------------------------------------------------------------------
    if st.session_state["is_scanning"] and st.session_state["uploaded_image_data"]:
        fname, scan_bytes, _ = st.session_state["uploaded_image_data"]
        b64_scan = base64.b64encode(scan_bytes).decode("utf-8")

        render_html("""
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: var(--danger); text-transform: uppercase;">SCANNING IN PROGRESS</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: var(--text-primary); margin-top: 4px;">Analyzing Industrial Component</div>
            <div style="font-size: 0.84rem; color: var(--text-secondary); margin-top: 4px;">Precision optical anomaly detection with sub-pixel localization</div>
        </div>
        """)

        render_html(f"""
        <div class="scan-container" style="max-width: 520px; margin: 0 auto;">
            <div class="corner-bracket corner-tl"></div>
            <div class="corner-bracket corner-tr"></div>
            <div class="corner-bracket corner-bl"></div>
            <div class="corner-bracket corner-br"></div>
            <div class="scan-laser"></div>
            <img src="data:image/png;base64,{b64_scan}" style="width: 100%; display: block; filter: contrast(1.02); border-radius: 4px;" />
            <div style="position: absolute; bottom: 12px; left: 16px; right: 16px; background: rgba(23, 23, 23, 0.88); backdrop-filter: blur(4px); padding: 8px 14px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; color: #FFFFFF; font-size: 0.72rem; font-weight: 600;">
                <span>● SCANNING ACTIVE</span>
                <span>RESNET-18 (448D) // CORESET MEMORY</span>
            </div>
        </div>

        <div style="max-width: 520px; margin: 18px auto 0 auto; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; font-size: 0.76rem; color: var(--text-secondary); display: flex; justify-content: space-between;">
            <span><b style="color: var(--text-primary);">01</b> Scanning Component</span>
            <span><b style="color: var(--text-primary);">02</b> Extracting 448D Features</span>
            <span><b style="color: var(--text-primary);">03</b> Anomaly Distance</span>
            <span><b style="color: var(--text-primary);">04</b> Morphological Gate</span>
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
    # STATE B: INSPECTION RESULT VIEW (PROFESSIONAL REPORT)
    # -------------------------------------------------------------------------
    elif st.session_state["current_inspection"] is not None:
        res = st.session_state["current_inspection"]
        is_defective = res.get("is_defective", False)
        status = res.get("status", "NORMAL")
        score = res.get("anomaly_score", 0.0)
        th = res.get("image_threshold", 0.0)
        p_th = res.get("pixel_threshold", 0.0)
        margin = res.get("decision_margin", score - th)
        n_defects = res.get("num_defects", 0)
        latency_ms = res.get("inference_time_ms", 180.0)
        explanation = res.get("explanation", "")
        why_explanation = res.get("why_explanation", "")
        regs = res.get("localized_regions", [])

        # Navigation & Status Header
        top_h1, top_h2 = st.columns([3.5, 1.3])
        with top_h1:
            if st.button("← Back to Inspect", key="res_back_btn"):
                purge_inspection()
                st.rerun()
        with top_h2:
            if st.button("Inspect Another Image →", key="res_inspect_another_btn", type="primary", use_container_width=True):
                purge_inspection()
                st.rerun()

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.10em; color: var(--text-secondary); text-transform: uppercase;'>INSPECTION RESULT</div>")

        if is_defective:
            subtext = f"{n_defects} localized anomaly region(s) detected exceeding nominal inspection criteria (Decision margin: +{margin:.4f})." if n_defects > 0 else f"Anomaly score exceeded nominal threshold (Decision margin: +{margin:.4f})."
            render_html(f"""
            <div style="font-size: 2.3rem; font-weight: 800; color: var(--danger); letter-spacing: -0.02em; line-height: 1.1;">DEFECTIVE</div>
            <div style="font-size: 0.92rem; color: var(--text-secondary); margin-top: 4px;">{subtext}</div>
            """)
        else:
            render_html(f"""
            <div style="font-size: 2.3rem; font-weight: 800; color: var(--success); letter-spacing: -0.02em; line-height: 1.1;">NORMAL</div>
            <div style="font-size: 0.92rem; color: var(--text-secondary); margin-top: 4px;">Zero anomaly clusters detected. Component matches nominal distribution within calibrated thresholds (Decision margin: {margin:.4f}).</div>
            """)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # MAIN RESULT VISUAL (HUGE THREE-PANEL SECTION)
        # ---------------------------------------------------------------------
        v_col1, v_col2, v_col3 = st.columns(3, gap="medium")

        # 01 ORIGINAL IMAGE
        with v_col1:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); margin-bottom: 6px;'>01 &nbsp; ORIGINAL IMAGE</div>")
            if st.session_state["uploaded_image_data"]:
                _, orig_bytes, _ = st.session_state["uploaded_image_data"]
                st.image(Image.open(io.BytesIO(orig_bytes)), use_container_width=True)
            render_html("<div style='font-size: 0.68rem; color: var(--text-muted); margin-top: 4px;'>Pristine optical sensor acquisition</div>")

        # 02 ANOMALY MAP
        with v_col2:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); margin-bottom: 6px;'>02 &nbsp; ANOMALY MAP</div>")
            hm_b64 = res.get("heatmap_base64")
            if hm_b64:
                st.image(Image.open(io.BytesIO(base64.b64decode(hm_b64))), use_container_width=True)
                render_html("""
                <div class="minimal-legend">
                    <span>Low (Nominal)</span>
                    <div class="legend-gradient"></div>
                    <span>High (Anomaly)</span>
                </div>
                """)
            else:
                st.info("Anomaly map not available.")

        # 03 DEFECT LOCALIZATION
        with v_col3:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); margin-bottom: 6px;'>03 &nbsp; DEFECT LOCALIZATION</div>")
            vis_b64 = res.get("visualization_base64")
            if is_defective and vis_b64:
                st.image(Image.open(io.BytesIO(base64.b64decode(vis_b64))), use_container_width=True)
                render_html(f"<div style='font-size: 0.68rem; color: var(--danger); margin-top: 4px; font-weight: 600;'>{n_defects} confirmed defect region(s) localized</div>")
            else:
                if st.session_state["uploaded_image_data"]:
                    _, orig_bytes, _ = st.session_state["uploaded_image_data"]
                    st.image(Image.open(io.BytesIO(orig_bytes)), use_container_width=True)
                render_html("<div style='font-size: 0.68rem; color: var(--success); margin-top: 4px; font-weight: 600;'>✔ Defect-free — Zero anomaly clusters detected</div>")

        # ---------------------------------------------------------------------
        # METRIC STRIP
        # ---------------------------------------------------------------------
        margin_sign = f"+{margin:.4f}" if margin > 0 else f"{margin:.4f}"
        margin_color = "var(--danger)" if margin > 0 else "var(--success)"

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
        # SECTION 1: WHY IS THIS PRODUCT DEFECTIVE / NORMAL?
        # ---------------------------------------------------------------------
        sec1_header = "WHY IS THIS PRODUCT DEFECTIVE?" if is_defective else "WHY IS THIS PRODUCT NORMAL?"
        if is_defective:
            what_text = f"Anomaly score {score:.4f} exceeded calibrated threshold {th:.4f} (Decision margin: +{margin:.4f})."
            if regs:
                where_text = f"{n_defects} localized anomaly region(s) were detected across the component surface."
            else:
                where_text = "Surface anomaly pattern recorded with elevated pixel intensity across the component boundary."
            why_text = f"PatchCore extracts 448-dimensional multi-scale patch representations from ResNet-18 Layers 1–3. In anomalous regions, feature distance to the nearest nominal vectors in the {cat_data['name']} coreset memory bank departed significantly from the learned normal manifold."
        else:
            what_text = f"Optical scan yielded an anomaly score of {score:.4f}, safely below the calibrated nominal threshold of {th:.4f} (Decision margin: {margin:.4f})."
            where_text = "No anomalous regions detected. Multi-scale patch representations across all spatial grid positions conform to nominal reference geometry."
            why_text = f"All 448-dimensional patch representations map tightly within the high-density manifold of normal feature vectors established by defect-free {cat_data['name']} training units."

        render_html(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 12px;">{sec1_header}</div>
            <div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.85rem; color: var(--text-primary); line-height: 1.55;">
                <div><span style="font-weight: 600; color: var(--text-primary);">• What was detected:</span> {what_text}</div>
                <div><span style="font-weight: 600; color: var(--text-primary);">• Where was it detected:</span> {where_text}</div>
                <div><span style="font-weight: 600; color: var(--text-primary);">• Why the model triggered:</span> {why_text}</div>
            </div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # SECTION 2: IS THIS PRODUCT STILL USABLE? (USABILITY ASSESSMENT)
        # ---------------------------------------------------------------------
        if is_defective:
            usability_badge = '<span style="display:inline-block; padding: 4px 10px; border-radius: 4px; background: var(--danger-bg); color: var(--danger); font-weight: 700; font-size: 0.72rem; border: 1px solid #FECACA; letter-spacing: 0.04em;">REQUIRES REVIEW</span>'
            usability_desc = "VisionInspect detected a visual deviation that requires secondary quality assessment. The anomaly detector identifies deviations from the learned normal pattern; it does not independently certify physical safety or fitness for use."
            insp_status_html = '<span style="color: var(--danger); font-weight: 700;">DEFECTIVE</span>'
            usab_status_html = '<span style="color: var(--danger); font-weight: 700;">REQUIRES REVIEW</span>'
            action_html = '<span style="color: var(--text-primary); font-weight: 500;">Secondary QA disposition review against engineering specifications</span>'
        else:
            usability_badge = '<span style="display:inline-block; padding: 4px 10px; border-radius: 4px; background: var(--success-bg); color: var(--success); font-weight: 700; font-size: 0.72rem; border: 1px solid #BBF7D0; letter-spacing: 0.04em;">PASSES VISUAL INSPECTION</span>'
            usability_desc = "No significant visual deviation from the learned normal pattern was detected under the configured inspection criteria. Multi-scale patch representations conform to the calibrated memory bank."
            insp_status_html = '<span style="color: var(--success); font-weight: 700;">NORMAL</span>'
            usab_status_html = '<span style="color: var(--success); font-weight: 700;">PASSES VISUAL INSPECTION</span>'
            action_html = '<span style="color: var(--text-primary); font-weight: 500;">Component clears optical quality gate. Proceed to standard operational testing</span>'

        render_html(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 18px 22px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase;">IS THIS PRODUCT STILL USABLE? (USABILITY ASSESSMENT)</span>
                {usability_badge}
            </div>
            <div style="font-size: 0.85rem; color: var(--text-primary); line-height: 1.55; margin-bottom: 14px;">
                {usability_desc}
            </div>
            <div style="background: var(--surface-subtle); border: 1px solid var(--border); border-radius: 6px; padding: 10px 16px; display: flex; flex-direction: column; gap: 8px; font-size: 0.82rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
                    <span style="color: var(--text-secondary);">Inspection status:</span>
                    {insp_status_html}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 6px;">
                    <span style="color: var(--text-secondary);">Usability status:</span>
                    {usab_status_html}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-secondary);">Recommended action:</span>
                    {action_html}
                </div>
            </div>
            <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 12px; line-height: 1.4;">
                Engineering Notice: Optical inspection evaluates surface and geometric concordance against the nominal reference manifold. Physical fitness for service depends on application-specific mechanical and electrical specifications.
            </div>
        </div>
        """)

        # ---------------------------------------------------------------------
        # LOCALIZED REGIONS BREAKDOWN (IF DEFECTIVE)
        # ---------------------------------------------------------------------
        if is_defective and regs:
            render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin: 16px 0 8px 0;'>LOCALIZED ANOMALY REGIONS</div>")
            reg_cols = st.columns(min(4, len(regs)))
            for r_idx, reg in enumerate(regs[:4]):
                with reg_cols[r_idx]:
                    lbl = reg.get("label", f"Region {r_idx+1:02d}")
                    intensity = reg.get("intensity", "Anomaly Region")
                    area_px = reg.get("area", 0)
                    r_score = reg.get("score", 0.0)
                    bbox = reg.get("bbox", [0, 0, 0, 0])
                    render_html(f"""
                    <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; font-size: 0.78rem;">
                        <div style="font-weight: 700; color: var(--danger);">{lbl}</div>
                        <div style="color: var(--text-secondary); margin-top: 2px;">{intensity}</div>
                        <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">Area: {area_px} px • Score: {r_score:.4f}</div>
                        <div style="font-size: 0.70rem; color: var(--text-muted); margin-top: 2px;">Box: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</div>
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
            - **Calibrated Pixel Threshold ($T_{{pixel}}$):** `{p_th:.4f}`
            - **Decision Rule:** Dual-Gated $[Score > T_{{image}}] \\land [Defect Area \\ge min\\_area]$
            - **Unique Request ID:** `{res.get('request_id', 'N/A')}`
            """)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
        if st.button("Inspect Another Image →", key="res_bottom_inspect_btn", type="primary", use_container_width=True):
            purge_inspection()
            st.rerun()

    # -------------------------------------------------------------------------
    # STATE C: 3-COLUMN INSPECTION LAYOUT (LEFT GUIDE | CENTER DROPZONE | RIGHT CATALOG)
    # -------------------------------------------------------------------------
    else:
        left_col, center_col, right_col = st.columns([1.1, 1.85, 1.15], gap="large")

        # LEFT COLUMN: Process Guide & Introduction
        with left_col:
            render_html("""
            <div class="hero-eyebrow">INSPECT COMPONENT</div>
            <h2 style="font-size: 1.85rem; font-weight: 700; color: var(--text-primary); margin: 0 0 8px 0; letter-spacing: -0.025em; line-height: 1.15;">
                Upload Your Image
            </h2>
            <p style="font-size: 0.88rem; color: var(--text-secondary); line-height: 1.5; margin-bottom: 20px;">
                Upload an image of the component you want to inspect. VisionInspect analyzes it for visual deviations from the learned normal pattern.
            </p>
            
            <div style="display: flex; flex-direction: column; gap: 14px; border-top: 1px solid var(--border); padding-top: 16px;">
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: var(--text-primary); min-width: 20px;">01</span>
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-primary);">Select Component</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary); margin-top: 2px;">Choose target component from the catalog on the right.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="font-size: 0.72rem; font-weight: 700; color: var(--text-primary); min-width: 20px;">02</span>
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-primary);">Upload Image</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary); margin-top: 2px;">Acquire optical sensor image or select a verified test sample.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="font-size: 0.72rem; font-weight: 700; min-width: 20px; color: var(--text-primary);">03</span>
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-primary);">Analyze</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary); margin-top: 2px;">Deep multi-scale 448D feature comparison against nominal memory bank.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; align-items: flex-start;">
                    <span style="font-size: 0.72rem; font-weight: 700; min-width: 20px; color: var(--text-primary);">04</span>
                    <div>
                        <div style="font-size: 0.82rem; font-weight: 600; color: var(--text-primary);">View Results</div>
                        <div style="font-size: 0.74rem; color: var(--text-secondary); margin-top: 2px;">Review anomaly heatmap, localized bounding boxes, and usability review.</div>
                    </div>
                </div>
            </div>
            """)

        # CENTER COLUMN: Main Visual Focus (Dropzone or Loaded Image)
        with center_col:
            if st.session_state["uploaded_image_data"]:
                fname, img_bytes, (iw, ih) = st.session_state["uploaded_image_data"]
                b64_loaded = base64.b64encode(img_bytes).decode("utf-8")

                render_html(f"""
                <div class="hero-visual-frame" style="min-height: 360px; padding: 20px;">
                    <div class="corner-bracket corner-tl"></div>
                    <div class="corner-bracket corner-tr"></div>
                    <div class="corner-bracket corner-bl"></div>
                    <div class="corner-bracket corner-br"></div>
                    <div class="visual-top-meta">
                        <span>TARGET: {cat_data['name'].upper()} ({cat_data['label'].upper()})</span>
                        <span>{iw} × {ih} PX</span>
                    </div>
                    <img src="data:image/png;base64,{b64_loaded}" style="max-height: 290px; max-width: 90%; object-fit: contain; margin: 24px 0; border-radius: 4px;" />
                    <div class="visual-bottom-meta">
                        <span>FILE: {fname}</span>
                        <span>READY FOR INSPECTION</span>
                    </div>
                </div>
                
                <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 10px 14px; margin-top: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.68rem; font-weight: 700; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em;">Image Selected</div>
                        <div style="font-size: 0.84rem; font-weight: 600; color: var(--text-primary); margin-top: 1px;">{fname}</div>
                    </div>
                    <div style="font-size: 0.74rem; color: var(--text-muted);">{iw} × {ih} px</div>
                </div>
                """)

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                btn_c1, btn_c2 = st.columns([2.0, 1.2])
                with btn_c1:
                    if st.button("ANALYZE IMAGE →", key="inspect_center_analyze_btn", type="primary", use_container_width=True):
                        st.session_state["is_scanning"] = True
                        st.rerun()
                with btn_c2:
                    if st.button("Change Image", key="inspect_center_clear_btn", use_container_width=True):
                        st.session_state["uploaded_image_data"] = None
                        st.rerun()

            else:
                # Before upload: Large clean dropzone + demo sample option
                render_html("""
                <div style="background: var(--surface); border: 1.5px dashed var(--border-strong); border-radius: 8px; padding: 36px 20px; text-align: center; margin-bottom: 8px;">
                    <div style="font-size: 1.8rem; color: var(--text-primary); margin-bottom: 6px;">⌖</div>
                    <div style="font-size: 1.05rem; font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">Drop an image here</div>
                    <div style="font-size: 0.76rem; color: var(--text-secondary);">PNG, JPG, WEBP • Industrial component optical scan</div>
                </div>
                """)

                uploaded_file = st.file_uploader(
                    "Choose Image",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="center_file_uploader",
                    label_visibility="collapsed"
                )
                if uploaded_file is not None:
                    raw_bytes = uploaded_file.getvalue()
                    pil_im = Image.open(io.BytesIO(raw_bytes))
                    st.session_state["uploaded_image_data"] = (uploaded_file.name, raw_bytes, pil_im.size)
                    st.rerun()

                render_html("""
                <div style="display: flex; align-items: center; text-align: center; margin: 16px 0 12px 0;">
                    <div style="flex: 1; border-bottom: 1px solid var(--border);"></div>
                    <span style="padding: 0 10px; font-size: 0.70rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.06em;">Or Choose Verified Sample</span>
                    <div style="flex: 1; border-bottom: 1px solid var(--border);"></div>
                </div>
                """)

                sample_options = [s[0] for s in cat_data["samples"]]
                selected_sample_label = st.selectbox(
                    f"Verified {cat_data['name']} Samples:",
                    options=["-- Select evaluation sample --"] + sample_options,
                    key="center_sample_selectbox"
                )
                if selected_sample_label != "-- Select evaluation sample --":
                    target_path = None
                    for s_name, s_path in cat_data["samples"]:
                        if s_name == selected_sample_label:
                            target_path = s_path
                            break
                    if target_path and Path(target_path).exists():
                        with open(target_path, "rb") as f:
                            s_bytes = f.read()
                        pil_im = Image.open(io.BytesIO(s_bytes))
                        st.session_state["uploaded_image_data"] = (Path(target_path).name, s_bytes, pil_im.size)
                        st.rerun()

        # RIGHT COLUMN: Component Catalog with Visual Thumbnails
        with right_col:
            render_html("""
            <div class="hero-eyebrow">SELECT COMPONENT</div>
            <div style="font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 14px;">Active model catalog:</div>
            """)

            for c_key, c_info in CATEGORIES.items():
                is_active = (active_cat == c_key)
                thumb_b64 = get_thumbnail_b64(c_info["golden_sample"])

                c_thumb_col, c_btn_col = st.columns([1.0, 3.0], gap="small")
                with c_thumb_col:
                    border_style = "1.5px solid var(--dark)" if is_active else "1px solid var(--border)"
                    render_html(f"""
                    <div style="width: 44px; height: 44px; border-radius: 6px; overflow: hidden; border: {border_style}; background: var(--surface); display: flex; align-items: center; justify-content: center; margin-top: 2px;">
                        <img src="data:image/jpeg;base64,{thumb_b64}" style="width: 100%; height: 100%; object-fit: cover;" />
                    </div>
                    """)
                with c_btn_col:
                    label_text = f"✔ {c_info['name']}" if is_active else c_info['name']
                    if is_active:
                        render_html(f"""
                        <style>
                            div[data-testid="stHorizontalBlock"] button[key="right_col_cat_{c_key}"] {{
                                border: 1.5px solid var(--dark) !important;
                                background-color: var(--surface) !important;
                                color: var(--text-primary) !important;
                                font-weight: 600 !important;
                            }}
                        </style>
                        """)
                    if st.button(label_text, key=f"right_col_cat_{c_key}", type="secondary", use_container_width=True):
                        if c_key != active_cat:
                            purge_inspection(new_category=c_key)
                            st.rerun()
                    render_html(f"""
                    <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: -6px; margin-bottom: 12px; line-height: 1.2;">
                        {c_info['label']}
                    </div>
                    """)

    # -------------------------------------------------------------------------
    # RECENT INSPECTIONS (MINIMAL LIST)
    # -------------------------------------------------------------------------
    if st.session_state["recent_inspections"]:
        st.markdown("<hr style='margin: 1.8rem 0 1.0rem 0; border: none; border-bottom: 1px solid var(--border);' />", unsafe_allow_html=True)
        render_html("<div style='font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 8px;'>RECENT INSPECTIONS (SESSION)</div>")
        
        hist_rows = []
        for item in st.session_state["recent_inspections"][:4]:
            badge_color = "var(--danger)" if item["status"] == "DEFECTIVE" else "var(--success)"
            hist_rows.append(
                f"<div style='display:flex; justify-content:space-between; align-items:center; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 0.80rem;'>"
                f"<span><span style='font-weight:600; color:var(--text-primary);'>#{item['id']}</span> &nbsp; {item['category'].title()} &nbsp;•&nbsp; <code style='font-size:0.75rem; color:var(--text-secondary);'>{item['filename'][:20]}</code></span>"
                f"<span style='color:{badge_color}; font-weight:700;'>{item['status']}</span>"
                f"<span style='color:var(--text-secondary);'>Score: {item['score']:.4f}</span>"
                f"<span style='color:var(--text-muted);'>Latency: {item['time_s']:.2f}s</span>"
                f"</div>"
            )
        render_html(f"<div style='background:var(--surface); border:1px solid var(--border); border-radius:6px; padding:10px 16px;'>{''.join(hist_rows)}</div>")


# =============================================================================
# PAGE 3: MODELS PAGE (MINIMALIST ARCHITECTURE & METHODOLOGY)
# =============================================================================
elif st.session_state["nav_page"] == "Models":
    render_html("""
    <div class="hero-eyebrow">SYSTEM ARCHITECTURE</div>
    <h1 style="font-size: 2.2rem; font-weight: 700; color: var(--text-primary); margin: 0 0 12px 0;">PatchCore v2.3 Architecture</h1>
    <p style="font-size: 0.95rem; color: var(--text-secondary); max-width: 680px; line-height: 1.5; margin-bottom: 24px;">
        VisionInspect operates on an unsupervised memory bank density paradigm. Feature representations are extracted from deep ResNet-18 layers, sub-sampled via greedy coreset selection, and tested without training on defect images.
    </p>
    """)

    m_col1, m_col2 = st.columns(2, gap="large")

    with m_col1:
        render_html("""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 20px; height: 100%;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 10px;">CORE PIPELINE STAGES</div>
            <div style="font-size: 0.82rem; color: var(--text-primary); line-height: 1.8;">
                <b style="color: var(--text-primary);">1. Multi-Scale Feature Extraction</b><br>
                ResNet-18 Layers 1, 2, and 3 are extracted and spatially bilinearly interpolated onto a uniform 64×64 grid, producing 448-dimensional localized patch descriptors.<br><br>
                <b style="color: var(--text-primary);">2. Coreset Memory Bank Subsampling</b><br>
                Greedy minimax coreset selection retains 10% of nominal feature vectors while maintaining complete coverage of the normal representation manifold.<br><br>
                <b style="color: var(--text-primary);">3. Nearest-Neighbor Anomaly Scoring</b><br>
                Test image patches are scored via exact Euclidean nearest-neighbor distance against the category's nominal memory bank.<br><br>
                <b style="color: var(--text-primary);">4. Dual-Gated Decision Rule</b><br>
                A component is defective if and only if its anomaly score exceeds T_image AND a morphological connected component exceeds min_area.
            </div>
        </div>
        """)

    with m_col2:
        render_html("""
        <div style="background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 20px; height: 100%;">
            <div style="font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 10px;">CATEGORY MODELS & THRESHOLDS</div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.80rem; margin-top: 8px;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border); text-align: left; color: var(--text-secondary);">
                        <th style="padding: 6px 0;">Category</th>
                        <th>T_image</th>
                        <th>T_pixel</th>
                        <th>Spatial Prior</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 8px 0; font-weight: 600; color: var(--text-primary);">Bottle</td>
                        <td>1.50</td>
                        <td>1.40</td>
                        <td>Active (p75/mean)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 8px 0; font-weight: 600; color: var(--text-primary);">Leather</td>
                        <td>2.20</td>
                        <td>2.71</td>
                        <td>Disabled (Texture)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 8px 0; font-weight: 600; color: var(--text-primary);">Transistor</td>
                        <td>3.525</td>
                        <td>2.822</td>
                        <td>Presence Gate</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 8px 0; font-weight: 600; color: var(--text-primary);">Zipper</td>
                        <td>1.47</td>
                        <td>0.91</td>
                        <td>Active (mean)</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 0; font-weight: 600; color: var(--text-primary);">Screw</td>
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
    <h1 style="font-size: 2.2rem; font-weight: 700; color: var(--text-primary); margin: 0 0 12px 0;">Autonomous Quality Inspection</h1>
    <p style="font-size: 0.95rem; color: var(--text-secondary); max-width: 680px; line-height: 1.5; margin-bottom: 24px;">
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
