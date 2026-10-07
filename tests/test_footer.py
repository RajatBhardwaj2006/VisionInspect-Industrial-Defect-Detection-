"""
tests/test_footer.py
Unit tests verifying the VisionInspect minimalist reference footer component:
- HTML structure and essential tokens
- Clean typography and oversized brand wordmark
- Real navigation routes and resources
- Strict absence of external branding (no Google, no Antigravity)
"""

import pytest
from app.components.footer import get_footer_html, render_footer


def test_footer_html_content_and_structure():
    """Verify footer contains correct headline, columns, wordmark, and links."""
    html = get_footer_html()
    assert html, "Footer HTML must not be empty"

    # 1. Headline
    assert "Experience VisionInspect" in html or ("See defects." in html and "Understand them." in html)

    # 2. Navigation Columns: Product & Resources
    assert "Product" in html
    assert "Resources" in html
    assert "Inspect" in html
    assert "Models" in html
    assert "How It Works" in html
    assert "Documentation" in html
    assert "About" in html
    assert "GitHub" in html
    assert "Technical Documentation" in html

    # 3. Oversized Wordmark
    assert "vi-footer-wordmark" in html
    assert "VisionInspect" in html

    # 4. Bottom Footer Bar
    assert "vi-footer-bottom" in html
    assert "vi-footer-divider" in html


def test_footer_no_external_branding():
    """Verify strictly zero external branding is present (Google, Antigravity)."""
    html = get_footer_html().lower()
    assert "google" not in html, "Forbidden external brand 'Google' found in footer"
    assert "antigravity" not in html, "Forbidden external brand 'Antigravity' found in footer"


def test_footer_valid_links_only():
    """Verify all internal links point to valid parameters or real GitHub repo."""
    html = get_footer_html()
    assert "?page=Inspect" in html
    assert "?page=Models" in html
    assert "?page=About" in html
    assert "https://github.com/RajatBhardwaj2006/VisionInspect-Industrial-Defect-Detection-" in html
    assert "http://127.0.0.1:8000/docs" in html


def test_footer_theme_and_reflection_effect():
    """Verify footer uses CSS theme variables and features mirror reflection effect on wordmark."""
    html = get_footer_html()
    assert "var(--surface)" in html, "Footer must match page surface theme"
    assert "var(--text-primary)" in html, "Footer headline must match page primary text theme"
    assert "-webkit-box-reflect" in html, "Wordmark must have mirror reflection effect"
    assert "linear-gradient" in html, "Wordmark must have gradient text effect"
