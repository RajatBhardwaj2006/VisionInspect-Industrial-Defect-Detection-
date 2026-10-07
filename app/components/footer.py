"""
app/components/footer.py
VisionInspect Minimalist Reference-Style Footer Component

Visual Rhythm:
- High-contrast, clean light aesthetic (#F5F5F3) against dark industrial upper sections
- Large editorial headline ("See defects. Understand them.")
- Two structured navigation columns: Product & Resources
- Oversized responsive brand wordmark ("VisionInspect") spanning the lower area
- Subtle horizontal divider with brand signature and essential navigation links
- Responsive layout scaling down cleanly without horizontal overflow
"""

import streamlit as st


def get_footer_html() -> str:
    """Returns the pure HTML/CSS representation of the VisionInspect footer."""
    return """
<style>
/* VisionInspect Theme-Matched Footer with Mirror Reflection Effect */
.vi-footer-container, #vi-site-footer {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    background-color: var(--surface) !important;
    color: var(--text-primary) !important;
    border-radius: 12px 12px 0 0;
    border: 1px solid var(--border) !important;
    border-bottom: none !important;
    padding: 44px 44px 28px 44px;
    margin-top: 48px;
    margin-bottom: 0 !important;
    width: 100%;
    box-sizing: border-box;
    position: relative;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
}

.vi-footer-container * {
    box-sizing: border-box;
}

.vi-footer-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 40px;
    flex-wrap: wrap;
}

.vi-footer-left {
    flex: 1 1 360px;
    max-width: 500px;
}

.vi-footer-headline {
    font-size: clamp(2.0rem, 3.6vw, 3.0rem);
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em;
    line-height: 1.14;
    margin: 0 0 12px 0;
}

.vi-footer-sub {
    font-size: 0.88rem;
    color: var(--text-secondary) !important;
    line-height: 1.55;
    margin: 0;
    max-width: 420px;
}

.vi-footer-nav-grid {
    display: flex;
    gap: 60px;
    flex-wrap: wrap;
}

.vi-footer-nav-col {
    min-width: 125px;
}

.vi-footer-nav-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-primary) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 14px;
}

.vi-footer-nav-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.vi-footer-nav-list li a {
    color: var(--text-secondary) !important;
    font-size: 0.86rem;
    font-weight: 500;
    text-decoration: none;
    transition: color 0.15s ease;
}

.vi-footer-nav-list li a:hover {
    color: var(--text-primary) !important;
    text-decoration: underline;
}

/* Oversized Brand Wordmark with Mirror Reflection Effect */
.vi-footer-wordmark-wrap {
    width: 100%;
    overflow: hidden;
    margin-top: 44px;
    margin-bottom: 34px;
    text-align: center;
    user-select: none;
    pointer-events: none;
    line-height: 0.88;
    position: relative;
    padding-bottom: 18px;
}

.vi-footer-wordmark {
    font-size: clamp(3.2rem, 11.2vw, 10.5rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 0.88;
    white-space: nowrap;
    text-align: center;
    display: inline-block;
    transform: scaleY(0.96);
    background: linear-gradient(180deg, #38BDF8 0%, #0284C7 52%, #0369A1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    -webkit-box-reflect: below -8px linear-gradient(to bottom, rgba(0, 0, 0, 0.55) 0%, rgba(0, 0, 0, 0.15) 35%, transparent 75%);
}

@supports not (-webkit-box-reflect: below 1px) {
    .vi-footer-wordmark-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .vi-footer-wordmark-reflection {
        display: inline-block;
        font-size: clamp(3.2rem, 11.2vw, 10.5rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 0.88;
        white-space: nowrap;
        text-align: center;
        background: linear-gradient(180deg, #38BDF8 0%, #0284C7 52%, #0369A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        transform: scaleY(-0.96);
        margin-top: -8px;
        opacity: 0.4;
        mask-image: linear-gradient(to top, transparent 20%, black 100%);
        -webkit-mask-image: linear-gradient(to top, transparent 20%, black 100%);
    }
}

/* Bottom Bar */
.vi-footer-divider {
    height: 1px;
    background-color: var(--border) !important;
    width: 100%;
    margin: 16px 0 14px 0;
}

.vi-footer-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
    font-size: 0.78rem;
    color: var(--text-secondary) !important;
}

.vi-footer-bottom-left {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 600;
    color: var(--text-primary) !important;
}

.vi-footer-bottom-left span:first-child {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

.vi-footer-copy {
    font-weight: 400;
    color: var(--text-muted) !important;
    font-size: 0.74rem;
}

.vi-footer-bottom-right {
    display: flex;
    align-items: center;
    gap: 18px;
}

.vi-footer-bottom-right a {
    color: var(--text-secondary) !important;
    text-decoration: none;
    font-weight: 500;
    font-size: 0.78rem;
    transition: color 0.15s ease;
}

.vi-footer-bottom-right a:hover {
    color: var(--text-primary) !important;
    text-decoration: underline;
}

@media (max-width: 768px) {
    .vi-footer-container, #vi-site-footer {
        padding: 30px 20px 16px 20px;
        margin-top: 28px;
    }
    .vi-footer-top {
        flex-direction: column;
        gap: 24px;
    }
    .vi-footer-nav-grid {
        gap: 32px;
        width: 100%;
    }
    .vi-footer-wordmark-wrap {
        margin-top: 26px;
        margin-bottom: 20px;
        padding-bottom: 10px;
    }
    .vi-footer-bottom {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }
}
</style>

<div class="vi-footer-container" id="vi-site-footer" role="contentinfo">
    <!-- Top Row: Headline & Navigation Columns -->
    <div class="vi-footer-top">
        <div class="vi-footer-left">
            <h2 class="vi-footer-headline">Experience VisionInspect</h2>
            <p class="vi-footer-sub">
                Autonomous unsupervised optical anomaly detection engineered for high-precision manufacturing inspection.
            </p>
        </div>

        <div class="vi-footer-nav-grid">
            <!-- Column 1: Product -->
            <div class="vi-footer-nav-col">
                <div class="vi-footer-nav-title">Product</div>
                <ul class="vi-footer-nav-list">
                    <li><a href="?page=Inspect" target="_self">Inspect</a></li>
                    <li><a href="?page=Models" target="_self">Models</a></li>
                    <li><a href="?page=Models" target="_self">How It Works</a></li>
                    <li><a href="?page=Models" target="_self">Documentation</a></li>
                </ul>
            </div>

            <!-- Column 2: Resources -->
            <div class="vi-footer-nav-col">
                <div class="vi-footer-nav-title">Resources</div>
                <ul class="vi-footer-nav-list">
                    <li><a href="?page=About" target="_self">About</a></li>
                    <li><a href="https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-" target="_blank" rel="noopener noreferrer">GitHub</a></li>
                    <li><a href="?page=About" target="_self">Project Report</a></li>
                    <li><a href="http://127.0.0.1:8000/docs" target="_blank" rel="noopener noreferrer">Technical Documentation</a></li>
                </ul>
            </div>
        </div>
    </div>

    <!-- Oversized Brand Wordmark with Mirror Reflection Effect -->
    <div class="vi-footer-wordmark-wrap">
        <div class="vi-footer-wordmark">VisionInspect</div>
    </div>

    <!-- Bottom Horizontal Divider -->
    <div class="vi-footer-divider"></div>

    <!-- Bottom Footer Bar -->
    <div class="vi-footer-bottom">
        <div class="vi-footer-bottom-left">
            <span>VisionInspect</span>
            <span class="vi-footer-copy">© 2026 • Edge Visual Quality Inspection</span>
        </div>
        <div class="vi-footer-bottom-right">
            <a href="?page=About" target="_self">About</a>
            <a href="?page=About" target="_self">Privacy & Terms</a>
            <a href="https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
    </div>
</div>
"""


def render_footer():
    """Renders the VisionInspect minimalist footer into the Streamlit app layout."""
    raw_html = get_footer_html()
    cleaned = "\n".join([line.strip() for line in raw_html.strip().split("\n") if line.strip()])
    st.markdown(cleaned, unsafe_allow_html=True)
