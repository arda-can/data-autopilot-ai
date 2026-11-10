"""PDF yerleşimi: gerçek Olist ödeme verisiyle yapılan sayfa sayfa kontrolden çıkan regresyon testleri."""

from __future__ import annotations

import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from reportlab.platypus import Paragraph, SimpleDocTemplate

from core.autopilot import AutopilotEngine, AutopilotResult
from core.health import DataHealthAnalyzer
from core.report import ReportChart, build_report, render_html, render_pdf


def _story(monkeypatch: pytest.MonkeyPatch) -> list:
    """render_pdf'in sayfaya dizdiği akış öğelerini yakalar."""
    story: list = []
    original = SimpleDocTemplate.build

    def capture(self, flowables, **kwargs):
        story.extend(flowables)
        return original(self, flowables, **kwargs)

    monkeypatch.setattr(SimpleDocTemplate, "build", capture)
    return story


def _texts(story: list) -> list[str]:
    texts: list[str] = []
    for item in story:
        children = getattr(item, "_content", None)
        for child in (children if children is not None else [item]):
            if isinstance(child, Paragraph):
                texts.append(child.getPlainText())
    return texts


def _page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b", pdf))


def test_bullets_are_separate_blocks(monkeypatch: pytest.MonkeyPatch, dirty_result: AutopilotResult) -> None:
    story = _story(monkeypatch)
    pdf = render_pdf(build_report(dirty_result, "test.csv"))
    bullets = [item for item in story if isinstance(item, Paragraph) and item.getPlainText().startswith(("• ", "1. "))]
    assert bullets
    for item in bullets:
        assert item.style.spaceAfter >= 0.9 * item.style.leading
        assert item.getPlainText().count("•") <= 1


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext kurulu değil")
def test_extracted_text_keeps_one_bullet_per_line(tmp_path, dirty_result: AutopilotResult) -> None:
    path = tmp_path / "rapor.pdf"
    path.write_bytes(render_pdf(build_report(dirty_result, "test.csv")))
    text = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, check=True
    ).stdout.decode("utf-8")
    assert "•" in text
    assert not [line for line in text.splitlines() if line.count("•") > 1 or re.search(r"\S ?• ", line)]
