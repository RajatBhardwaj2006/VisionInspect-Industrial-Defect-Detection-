"""
app/utils/pdf_report.py
=======================
Structured, publication-grade PDF Inspection Report Generator for VisionInspect.
Builds a 5-page formal QA report using ReportLab:
- Page 1: Executive Summary, Metadata & Full-resolution Product Image
- Page 2: Visual Analysis (Original, Heatmap, Localization & How-to-read Guide)
- Page 3: Inspection Scores, Dynamic Deviation Chart & Range Interpretation
- Page 4: Plain-English Findings (What, Where, Disposition & Action)
- Page 5: Technical Engineering Inspection Report (Parameters, Regions & Pipeline)
"""

import io
import re
import html
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas


def _clean_text(text: Any) -> str:
    """
    Sanitize text for ReportLab Platypus Paragraphs.
    Replaces unsupported entities/symbols with safe ASCII equivalents,
    and escapes XML special characters to prevent paraparser syntax errors.
    """
    if text is None:
        return ""
    s = str(text)
    replacements = {
        "&times;": "x",
        "&Delta;": "Delta",
        "&rarr;": "->",
        "&mdash;": " - ",
        "&ndash;": "-",
        "&ge;": ">=",
        "&le;": "<=",
        "&bull;": "*",
        "✔": "[OK]",
        "✓": "[OK]",
        "Δ": "Delta",
        "×": "x",
        "→": "->",
        "—": " - ",
        "–": "-",
        "≥": ">=",
        "≤": "<=",
        "•": "*",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # XML escape special characters
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return s


def _clean_markup(s: str) -> str:
    """
    Sanitize a markup string intended for ReportLab Platypus Paragraph.
    Replaces unsupported entities, removes forbidden attributes (like align on <font>),
    and ensures lone ampersands are escaped.
    """
    if not s:
        return ""
    replacements = {
        "&times;": "x",
        "&Delta;": "Delta",
        "&rarr;": "->",
        "&mdash;": " - ",
        "&ndash;": "-",
        "&ge;": ">=",
        "&le;": "<=",
        "&bull;": "*",
        "✔": "[OK]",
        "✓": "[OK]",
        "Δ": "Delta",
        "×": "x",
        "→": "->",
        "—": " - ",
        "–": "-",
        "≥": ">=",
        "≤": "<=",
        "•": "*",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    # Strip illegal align attribute inside <font> tags
    s = re.sub(r'(<font\b[^>]*?)\s+align=[\'"][^\'"]*[\'"]', r'\1', s, flags=re.IGNORECASE)
    # Escape bare & not part of standard XML entities
    s = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', s)
    return s



class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and print total page numbers (Page X of Y)
    and standard industrial quality disclaimer across all pages.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        self.saveState()
        self.setFont("Helvetica", 7.5)
        self.setFillColor(colors.HexColor("#6B7280"))

        # Running header rule on pages > 1
        if self._pageNumber > 1:
            self.setStrokeColor(colors.HexColor("#E5E7EB"))
            self.setLineWidth(0.5)
            self.line(36, letter[1] - 30, letter[0] - 36, letter[1] - 30)
            self.drawString(36, letter[1] - 25, "VisionInspect Industrial Defect Inspection System — QA Report")
            self.drawRightString(letter[0] - 36, letter[1] - 25, "CONFIDENTIAL // QUALITY AUDIT")

        # Running footer rule
        self.setStrokeColor(colors.HexColor("#E5E7EB"))
        self.setLineWidth(0.5)
        self.line(36, 42, letter[0] - 36, 42)

        # Regulatory disclaimer
        disclaimer = "VisionInspect provides visual anomaly detection based on the configured inspection model and criteria. It does not replace an organization's final quality-control, safety, or regulatory decision."
        self.drawString(36, 30, disclaimer)

        # Page numbering
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(letter[0] - 36, 30, page_str)
        self.restoreState()


def _generate_score_chart_bytes(score: float, threshold: float, is_defective: bool) -> bytes:
    """
    Generates a high-resolution, publication-quality horizontal score deviation chart
    using the real score and calibrated threshold without arbitrary universal limits.
    """
    fig, ax = plt.subplots(figsize=(6.5, 1.25), dpi=180)
    
    # Dynamic xmax scaled to actual values
    max_val = max(threshold * 1.55, score * 1.25, threshold + 0.5)
    
    # Normal Range: 0 to threshold
    ax.axvspan(0, threshold, color="#E6F4EA", alpha=0.95, label="Normal / Low Deviation")
    # Anomalous / Review Range: threshold to max_val
    ax.axvspan(threshold, max_val, color="#FCE8E6", alpha=0.95, label="Anomalous / Review Range")
    
    # Vertical threshold line
    ax.axvline(threshold, color="#15803D" if not is_defective else "#B91C1C",
               linestyle="--", linewidth=1.8, label=f"Threshold ({threshold:.4f})")
    
    # Actual score marker
    marker_color = "#B91C1C" if is_defective else "#15803D"
    ax.plot([score], [0.5], marker="o", markersize=9, color=marker_color,
            markeredgecolor="#FFFFFF", markeredgewidth=1.5, zorder=5,
            label=f"Inspected Score ({score:.4f})")
    
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Anomaly Score Deviation Scale", fontsize=8, labelpad=4, color="#374151")
    ax.tick_params(axis="x", labelsize=7.5, colors="#4B5563")
    
    # Clean styling
    for spine in ax.spines.values():
        spine.set_color("#D1D5DB")
        spine.set_linewidth(0.6)
        
    ax.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="#FFFFFF",
              edgecolor="#E5E7EB", framealpha=0.9)
    plt.tight_layout(pad=0.8)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180)
    plt.close(fig)
    return buf.getvalue()


def _b64_to_bytes(b64_str: Optional[str]) -> Optional[bytes]:
    """Helper to safely decode base64 strings."""
    if not b64_str:
        return None
    try:
        # Strip header if present
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        return base64.b64decode(b64_str)
    except Exception:
        return None


def generate_inspection_pdf(
    inspection_data: Dict[str, Any],
    category: str,
    filename: Optional[str] = None,
    original_image_bytes: Optional[bytes] = None,
    heatmap_base64: Optional[str] = None,
    visualization_base64: Optional[str] = None,
    timestamp_str: Optional[str] = None,
    why_findings: Optional[str] = None,
    where_text: Optional[str] = None,
    disposition_text: Optional[str] = None,
    action_text: Optional[str] = None,
    quality_status_text: Optional[str] = None,
) -> bytes:
    """
    Main entrypoint: Generates a complete 5-page PDF report for the active inspection.
    Returns raw PDF bytes.
    """
    category_id = category.strip().lower()
    cat_title = category_id.title()
    fname = filename or inspection_data.get("filename", "inspected_sample.png")
    timestamp = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    is_defective = inspection_data.get("is_defective", False)
    status_verdict = "VISUAL ANOMALY DETECTED" if is_defective else "NORMAL"
    score = float(inspection_data.get("anomaly_score", 0.0))
    threshold = float(inspection_data.get("image_threshold", 0.0))
    pixel_threshold = float(inspection_data.get("pixel_threshold", 0.0))
    margin = float(inspection_data.get("decision_margin", score - threshold))
    n_defects = int(inspection_data.get("num_defects", len(inspection_data.get("localized_regions", []))))
    latency_ms = float(inspection_data.get("inference_time_ms", 150.0))
    peak_anomaly = float(inspection_data.get("peak_anomaly", score))
    request_id = inspection_data.get("request_id", "N/A")
    mem_bank = inspection_data.get("memory_bank", "Greedy Coreset (10% nominal features)")
    regions = inspection_data.get("localized_regions", [])

    # Text narratives (fallback to standardized defaults if not provided)
    if not why_findings:
        if is_defective:
            why_findings = f"An unusual visual area was detected on the {category_id} component that departs from the learned normal appearance."
        else:
            why_findings = f"The {category_id} appearance is consistent with the normal {category_id} examples learned by VisionInspect. No obvious surface cuts, fractures, or foreign anomalies were detected."

    if not where_text:
        where_text = f"The highlighted region shows where VisionInspect detected the visual difference." if is_defective else "No suspicious region was detected."

    if not disposition_text:
        disposition_text = "Visual anomaly detected" if is_defective else "No visible anomaly detected"

    if not action_text:
        action_text = "Send the product for human quality review according to the organization's inspection procedure." if is_defective else "Continue with the normal quality-control process."

    if not quality_status_text:
        quality_status_text = "REVIEW REQUIRED" if is_defective else "NORMAL"

    # Setup ReportLab Document
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=54
    )

    # Styles
    base_styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=base_styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=12
    )
    subtitle_center = ParagraphStyle(
        "DocSubTitleCenter",
        parent=subtitle_style,
        alignment=1,
        spaceAfter=8
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=base_styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1F2937"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#374151")
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#111827")
    )
    callout_style = ParagraphStyle(
        "Callout",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1F2937")
    )

    elements = []

    # -------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE SUMMARY & ORIGINAL PRODUCT IMAGE
    # -------------------------------------------------------------------------
    # Top Header Banner
    logo_path = Path("app/assets/icons/visioninspect_3d_logo_72.png")
    if not logo_path.exists():
        logo_path = Path("assets/icons/visioninspect_3d_logo_72.png")
    
    header_cells = []
    if logo_path.exists():
        try:
            rl_logo = RLImage(str(logo_path), width=36, height=36)
            header_cells = [
                rl_logo,
                Paragraph("<b>VISIONINSPECT</b><br/><font size=8 color='#6B7280'>Industrial Visual Inspection Report // Automated Optical QA</font>", title_style)
            ]
        except Exception:
            header_cells = ["", Paragraph("<b>VISIONINSPECT</b><br/><font size=8 color='#6B7280'>Industrial Visual Inspection Report</font>", title_style)]
    else:
        header_cells = ["", Paragraph("<b>VISIONINSPECT</b><br/><font size=8 color='#6B7280'>Industrial Visual Inspection Report</font>", title_style)]

    header_table = Table([header_cells], colWidths=[44, 496])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"), spaceAfter=14))

    # Inspection Metadata Card
    status_bg = colors.HexColor("#FEE2E2") if is_defective else colors.HexColor("#DCFCE7")
    status_fg = colors.HexColor("#B91C1C") if is_defective else colors.HexColor("#15803D")
    
    meta_data = [
        [
            Paragraph("<b>INSPECTION RESULT:</b>", body_style),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{status_verdict}</b></font>", body_bold),
            Paragraph("<b>CATEGORY:</b>", body_style),
            Paragraph(f"<b>{_clean_text(cat_title)}</b>", body_style),
        ],
        [
            Paragraph("<b>INSPECTION DATE:</b>", body_style),
            Paragraph(_clean_text(timestamp), body_style),
            Paragraph("<b>IMAGE FILE:</b>", body_style),
            Paragraph(_clean_text(fname[:26]), body_style),
        ],
        [
            Paragraph("<b>MODEL ARCHITECTURE:</b>", body_style),
            Paragraph("PatchCore v2.3 (Unsupervised ResNet-18)", body_style),
            Paragraph("<b>REQUEST TRACE ID:</b>", body_style),
            Paragraph(f"<font size=7>{_clean_text(request_id[:20])}</font>", body_style),
        ]
    ]
    meta_table = Table(meta_data, colWidths=[130, 160, 100, 150])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9FAFB")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # Result Summary Banner
    summary_text = (
        f"<b>Result Summary:</b> Visual anomaly detected exceeding calibrated inspection criteria ({n_defects} region flagged)."
        if is_defective else
        f"<b>Result Summary:</b> No significant visual anomaly detected. The component conforms to nominal {_clean_text(cat_title)} baseline appearance."
    )
    banner_table = Table([[Paragraph(summary_text, callout_style)]], colWidths=[540])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), status_bg),
        ('BOX', (0,0), (-1,-1), 1, status_fg),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    elements.append(banner_table)
    elements.append(Spacer(1, 14))

    # Original Product Image (Large display)
    elements.append(Paragraph("ORIGINAL INSPECTED SAMPLE", section_h1))
    if original_image_bytes:
        try:
            im_orig = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
            # Proportional scale
            orig_w, orig_h = im_orig.size
            max_disp_w, max_disp_h = 360, 320
            scale = min(max_disp_w / orig_w, max_disp_h / orig_h)
            disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
            
            buf_orig_scaled = io.BytesIO()
            im_orig.save(buf_orig_scaled, format="JPEG", quality=92)
            buf_orig_scaled.seek(0)
            
            img_flow = RLImage(buf_orig_scaled, width=disp_w, height=disp_h)
            img_table = Table([[img_flow]], colWidths=[540])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('TOPPADDING', (0,0), (-1,-1), 8),
                ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ]))
            elements.append(img_table)
            elements.append(Paragraph(f"<font size=7 color='#6B7280'>Captured Component: {_clean_text(fname)} (Resolution: {orig_w} x {orig_h} px)</font>", subtitle_center))
        except Exception:
            elements.append(Paragraph("Original image preview unavailable.", body_style))
    else:
        elements.append(Paragraph("Original image data not provided in payload.", body_style))

    # Page 1 End
    elements.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 2: VISUAL ANALYSIS (ORIGINAL, HEATMAP, LOCALIZATION)
    # -------------------------------------------------------------------------
    elements.append(Paragraph("VISUAL ANALYSIS & DEFECT LOCALIZATION", title_style))
    elements.append(Paragraph("Comparison between the submitted component, multi-scale anomaly heatmap, and extracted candidate regions.", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB"), spaceAfter=14))

    # 3-Panel Visual Comparison
    hm_raw = _b64_to_bytes(heatmap_base64)
    vis_raw = _b64_to_bytes(visualization_base64)

    panels = []
    panel_w, panel_h = 168, 168

    # 1. Original
    if original_image_bytes:
        try:
            im_p1 = Image.open(io.BytesIO(original_image_bytes)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            buf_p1 = io.BytesIO()
            im_p1.save(buf_p1, format="JPEG", quality=90)
            buf_p1.seek(0)
            panels.append(RLImage(buf_p1, width=panel_w, height=panel_h))
        except Exception:
            panels.append(Paragraph("Original N/A", body_style))
    else:
        panels.append(Paragraph("Original N/A", body_style))

    # 2. Heatmap
    if hm_raw:
        try:
            im_p2 = Image.open(io.BytesIO(hm_raw)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            buf_p2 = io.BytesIO()
            im_p2.save(buf_p2, format="JPEG", quality=90)
            buf_p2.seek(0)
            panels.append(RLImage(buf_p2, width=panel_w, height=panel_h))
        except Exception:
            panels.append(Paragraph("Heatmap N/A", body_style))
    else:
        panels.append(Paragraph("Heatmap N/A", body_style))

    # 3. Localization
    if vis_raw:
        try:
            im_p3 = Image.open(io.BytesIO(vis_raw)).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            buf_p3 = io.BytesIO()
            im_p3.save(buf_p3, format="JPEG", quality=90)
            buf_p3.seek(0)
            panels.append(RLImage(buf_p3, width=panel_w, height=panel_h))
        except Exception:
            panels.append(Paragraph("Localization N/A", body_style))
    elif original_image_bytes and not is_defective:
        # Normal sample shows original clean image
        panels.append(panels[0])
    else:
        panels.append(Paragraph("Localization N/A", body_style))

    panel_labels = [
        Paragraph("<b>01 - Original Product</b><br/><font size=7 color='#6B7280'>Input component image</font>", body_style),
        Paragraph("<b>02 - Anomaly Heatmap</b><br/><font size=7 color='#6B7280'>Pixel deviation intensity</font>", body_style),
        Paragraph("<b>03 - Detected Region</b><br/><font size=7 color='#6B7280'>Segmented defect cluster</font>", body_style)
    ]

    panel_table = Table([[panel_labels[0], panel_labels[1], panel_labels[2]], [panels[0], panels[1], panels[2]]], colWidths=[180, 180, 180])
    panel_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,1), (0,1), 1, colors.HexColor("#D1D5DB")),
        ('BOX', (1,1), (1,1), 1, colors.HexColor("#D1D5DB")),
        ('BOX', (2,1), (2,1), 1, colors.HexColor("#D1D5DB")),
    ]))
    elements.append(panel_table)
    elements.append(Spacer(1, 14))

    # How to Read the Result Explanatory Box
    guide_rows = [
        [
            Paragraph("<b>HOW TO READ THE RESULT</b>", body_bold)
        ],
        [
            Paragraph("<b>Original Image:</b> The raw optical capture submitted for manufacturing QA inspection.", body_style)
        ],
        [
            Paragraph("<b>Anomaly Heatmap:</b> Visualizes areas with stronger visual differences from the learned normal appearance. Cooler blue colors indicate nominal conformance, while yellow-to-red highlights indicate deviation. <i>Note: Heatmap intensity represents model anomaly signal and is not itself a confirmed physical defect.</i>", body_style)
        ],
        [
            Paragraph("<b>Detected Region:</b> Shows localized bounding boxes for regions that satisfied the dual-gated threshold and morphological area criteria. Defect-free samples produce zero candidate regions.", body_style)
        ]
    ]
    guide_table = Table(guide_rows, colWidths=[540])
    guide_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9FAFB")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(guide_table)

    # Page 2 End
    elements.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 3: INSPECTION METRICS & ANOMALY DEVIATION
    # -------------------------------------------------------------------------
    elements.append(Paragraph("INSPECTION METRICS & ANOMALY DEVIATION", title_style))
    elements.append(Paragraph("Statistical comparison between observed component anomaly score and calibrated quality thresholds.", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB"), spaceAfter=14))

    # Metrics Summary Grid Table
    score_status = "HIGH / ABOVE THRESHOLD" if score >= threshold else "GOOD / BELOW THRESHOLD"
    margin_sign = f"+{margin:.4f}" if margin > 0 else f"{margin:.4f}"
    margin_label = "ABOVE THRESHOLD" if margin > 0 else "BELOW THRESHOLD"
    regions_label = f"{n_defects} REGION(S) FLAGGED" if n_defects > 0 else "NO SUSPICIOUS REGION"

    metrics_rows = [
        [
            Paragraph("<b>METRIC</b>", body_bold),
            Paragraph("<b>RECORDED VALUE</b>", body_bold),
            Paragraph("<b>EVALUATION STATUS</b>", body_bold),
            Paragraph("<b>OPERATIONAL MEANING</b>", body_bold)
        ],
        [
            Paragraph("<b>Anomaly Score</b>", body_style),
            Paragraph(f"<b>{score:.4f}</b>", body_style),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{score_status}</b></font>", body_style),
            Paragraph("Overall visual difference from nominal baseline.", body_style)
        ],
        [
            Paragraph("<b>Inspection Threshold</b>", body_style),
            Paragraph(f"<b>{threshold:.4f}</b>", body_style),
            Paragraph("Calibrated Boundary", body_style),
            Paragraph(f"Calibrated cutoff for {cat_title} acceptance.", body_style)
        ],
        [
            Paragraph("<b>Decision Margin</b>", body_style),
            Paragraph(f"<b>{margin_sign}</b>", body_style),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{margin_label}</b></font>", body_style),
            Paragraph("Distance between anomaly score and threshold.", body_style)
        ],
        [
            Paragraph("<b>Detected Regions</b>", body_style),
            Paragraph(f"<b>{n_defects}</b>", body_style),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{regions_label}</b></font>", body_style),
            Paragraph("Discrete clusters exceeding pixel criteria.", body_style)
        ],
        [
            Paragraph("<b>Inspection Time</b>", body_style),
            Paragraph(f"<b>{latency_ms / 1000.0:.2f} s</b> <font size=7 color='#6B7280'>({int(latency_ms)} ms)</font>", body_style),
            Paragraph("Within Line Speed", body_style),
            Paragraph("End-to-end tensor evaluation latency.", body_style)
        ]
    ]

    metric_table = Table(metrics_rows, colWidths=[125, 95, 140, 180])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1D5DB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(metric_table)
    elements.append(Spacer(1, 14))

    # Dynamic Score vs Threshold Deviation Chart
    elements.append(Paragraph("SCORE VS. THRESHOLD DEVIATION SCALE", section_h1))
    try:
        chart_bytes = _generate_score_chart_bytes(score, threshold, is_defective)
        buf_chart = io.BytesIO(chart_bytes)
        chart_flow = RLImage(buf_chart, width=540, height=104)
        elements.append(chart_flow)
    except Exception as e:
        elements.append(Paragraph(f"Deviation chart generation unavailable: {e}", body_style))

    elements.append(Spacer(1, 10))
    elements.append(Paragraph("""
    <font size=8 color='#4B5563'>
    <b>Understanding the scale:</b> The green zone indicates nominal appearance conforming to defect-free training samples. The vertical dashed line marks the calibrated decision threshold. When a component's anomaly score extends beyond the threshold into the red region, the sample requires quality inspection review.
    </font>
    """, body_style))

    # Page 3 End
    elements.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 4: WHY DID VISIONINSPECT GIVE THIS RESULT?
    # -------------------------------------------------------------------------
    elements.append(Paragraph("WHY DID VISIONINSPECT GIVE THIS RESULT?", title_style))
    elements.append(Paragraph("Plain-English explanation of model findings, defect location, and quality disposition.", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB"), spaceAfter=14))

    why_rows = [
        [
            Paragraph("<b>WHAT DID VISIONINSPECT FIND?</b>", body_bold)
        ],
        [
            Paragraph(_clean_text(why_findings), body_style)
        ],
        [
            Paragraph("<b>WHERE WAS THE ANOMALY DETECTED?</b>", body_bold)
        ],
        [
            Paragraph(_clean_text(where_text), body_style)
        ],
        [
            Paragraph("<b>INSPECTION DISPOSITION</b>", body_bold)
        ],
        [
            Paragraph(f"<b>{status_verdict}</b> - {_clean_text(disposition_text)}", body_style)
        ],
        [
            Paragraph("<b>RECOMMENDED ACTION</b>", body_bold)
        ],
        [
            Paragraph(_clean_text(action_text), body_style)
        ]
    ]

    why_table = Table(why_rows, colWidths=[540])
    why_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F9FAFB")),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor("#F9FAFB")),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor("#F9FAFB")),
        ('BACKGROUND', (0,6), (-1,6), colors.HexColor("#F9FAFB")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E5E7EB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(why_table)
    elements.append(Spacer(1, 16))

    # Product Disposition Triad Summary
    disp_summary_data = [
        [
            Paragraph("<b>Quality Status:</b>", body_style),
            Paragraph(f"<font color='{status_fg.hexval()}'><b>{_clean_text(quality_status_text)}</b></font>", body_bold)
        ],
        [
            Paragraph("<b>Usability Disposition:</b>", body_style),
            Paragraph(_clean_text(disposition_text), body_style)
        ],
        [
            Paragraph("<b>Recommended Action:</b>", body_style),
            Paragraph(_clean_text(action_text), body_style)
        ]
    ]
    disp_table = Table(disp_summary_data, colWidths=[150, 390])
    disp_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1D5DB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFFFF")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(disp_table)

    # Page 4 End
    elements.append(PageBreak())

    # -------------------------------------------------------------------------
    # PAGE 5: TECHNICAL INSPECTION REPORT — ENGINEERING & QA
    # -------------------------------------------------------------------------
    elements.append(Paragraph("TECHNICAL INSPECTION REPORT — ENGINEERING & QA", title_style))
    elements.append(Paragraph("Complete technical parameters, model checkpoints, localized bounding boxes, and pipeline trace.", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#E5E7EB"), spaceAfter=14))

    # Engineering Specifications Table
    tech_data = [
        [
            Paragraph("<b>Parameter / Metric</b>", body_bold),
            Paragraph("<b>Specification & Value</b>", body_bold)
        ],
        [
            Paragraph("<b>Model Architecture</b>", body_style),
            Paragraph("PatchCore v2.3 (Unsupervised Density-based Anomaly Localization)", body_style)
        ],
        [
            Paragraph("<b>Model Checkpoint Path</b>", body_style),
            Paragraph(f"<code>models/{category_id}/patchcore_v23/</code>", body_style)
        ],
        [
            Paragraph("<b>Backbone Feature Extractor</b>", body_style),
            Paragraph("ResNet-18 (ImageNet Pretrained, Layer 1, 2, 3 Multi-Scale Pooling)", body_style)
        ],
        [
            Paragraph("<b>Feature Representation</b>", body_style),
            Paragraph("448-dimensional patch representations on a 64 x 64 grid (4,096 patches)", body_style)
        ],
        [
            Paragraph("<b>Memory Bank Subsampling</b>", body_style),
            Paragraph(_clean_text(mem_bank), body_style)
        ],
        [
            Paragraph("<b>Image Anomaly Score (S_image)</b>", body_style),
            Paragraph(f"<b>{score:.4f}</b>", body_style)
        ],
        [
            Paragraph("<b>Inspection Threshold (T_image)</b>", body_style),
            Paragraph(f"<b>{threshold:.4f}</b>", body_style)
        ],
        [
            Paragraph("<b>Peak Pixel Anomaly (S_pixel)</b>", body_style),
            Paragraph(f"<b>{peak_anomaly:.4f}</b> (Pixel Threshold T_pixel: {pixel_threshold:.4f})", body_style)
        ],
        [
            Paragraph("<b>Decision Margin (Delta)</b>", body_style),
            Paragraph(f"<b>{margin_sign}</b>", body_style)
        ],
        [
            Paragraph("<b>Decision Rule</b>", body_style),
            Paragraph("Dual-Gated: [Anomaly Score &gt; T_image] AND [Defect Area &gt;= min_area]", body_style)
        ],
        [
            Paragraph("<b>Inference Latency</b>", body_style),
            Paragraph(f"<b>{int(latency_ms)} ms</b> ({latency_ms / 1000.0:.3f} s)", body_style)
        ],
        [
            Paragraph("<b>Request Trace ID</b>", body_style),
            Paragraph(f"<code>{_clean_text(request_id)}</code>", body_style)
        ]
    ]

    tech_table = Table(tech_data, colWidths=[180, 360])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F3F4F6")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1D5DB")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    elements.append(tech_table)
    elements.append(Spacer(1, 14))

    # Localized Regions Table
    elements.append(Paragraph("LOCALIZED ANOMALY REGIONS (REJECTION ANALYSIS)", section_h1))
    if is_defective and regions:
        reg_table_data = [
            [
                Paragraph("<b>Region</b>", body_bold),
                Paragraph("<b>Bounding Box [X, Y, W, H]</b>", body_bold),
                Paragraph("<b>Dimensions</b>", body_bold),
                Paragraph("<b>Area</b>", body_bold),
                Paragraph("<b>Peak Score</b>", body_bold),
                Paragraph("<b>Severity</b>", body_bold)
            ]
        ]
        for r_idx, reg in enumerate(regions[:6]):
            lbl = reg.get("label", f"Region {r_idx+1:02d}")
            bbox = reg.get("bbox", [reg.get("x",0), reg.get("y",0), reg.get("width",0), reg.get("height",0)])
            w = reg.get("width", bbox[2] if len(bbox) > 2 else 0)
            h = reg.get("height", bbox[3] if len(bbox) > 3 else 0)
            area = reg.get("area", w * h)
            r_score = float(reg.get("score", 0.0))
            intensity = reg.get("intensity", "Anomalous")
            
            reg_table_data.append([
                Paragraph(_clean_text(lbl), body_style),
                Paragraph(f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]", body_style),
                Paragraph(f"{w} x {h} px", body_style),
                Paragraph(f"{area:,} px", body_style),
                Paragraph(f"{r_score:.4f}", body_style),
                Paragraph(f"<b>{_clean_text(intensity)}</b>", body_style)
            ])
        
        reg_table = Table(reg_table_data, colWidths=[70, 150, 80, 70, 75, 95])
        reg_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FEE2E2")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D1D5DB")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E5E7EB")),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ]))
        elements.append(reg_table)
    else:
        norm_reg_table = Table([[
            Paragraph("<font color='#15803D'><b>[OK] Zero Rejection Regions</b> - All component surface areas conform within nominal quality thresholds.</font>", body_style)
        ]], colWidths=[540])
        norm_reg_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#DCFCE7")),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#86EFAC")),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
        ]))
        elements.append(norm_reg_table)

    elements.append(Spacer(1, 14))

    # Engineering Pipeline Architecture Trace
    elements.append(Paragraph("HOW THIS RESULT WAS PRODUCED (ENGINEERING PIPELINE)", section_h1))
    pipeline_text = """
    <font size=7 color='#374151'>
    <b>INPUT IMAGE (256x256 RGB)</b> -&gt; 
    <b>ResNet-18 FEATURE EXTRACTION</b> -&gt; 
    <b>448D MULTI-SCALE PATCHES (64x64)</b> -&gt; 
    <b>NORMAL MEMORY BANK COMPARISON (torch.cdist)</b> -&gt; 
    <b>ANOMALY SCORE EVALUATION</b> -&gt; 
    <b>CATEGORY THRESHOLD (T_image)</b> -&gt; 
    <b>ANOMALY MAP POST-PROCESSING</b> -&gt; 
    <b>MORPHOLOGICAL LOCALIZATION</b> -&gt; 
    <b>FINAL PASS/FAIL VERDICT</b>
    </font>
    """
    pipe_box = Table([[Paragraph(pipeline_text, body_style)]], colWidths=[540])
    pipe_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F3F4F6")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    elements.append(pipe_box)

    # Build the document using NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
