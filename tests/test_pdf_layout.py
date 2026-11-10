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


def _report_with_charts(result: AutopilotResult, figures: list) -> object:
    report = build_report(result, "test.csv")
    report.charts = [ReportChart(title=f"Grafik {index}", reason="", insight=f"bulgu {index}.") for index in range(len(figures))]
    return report


def test_findings_of_charts_missing_from_pdf_point_to_html(monkeypatch: pytest.MonkeyPatch, dirty_result: AutopilotResult) -> None:
    figures = [
        px.scatter(x=[1, 2], y=[3, 4]),
        px.scatter_3d(x=[1, 2], y=[2, 3], z=[3, 4]),
        px.parallel_coordinates(pd.DataFrame({"a": [1, 2], "b": [2, 3]})),
    ]
    story = _story(monkeypatch)
    render_pdf(_report_with_charts(dirty_result, figures), figures)
    texts = _texts(story)
    assert "• Grafik 1: bulgu 1. (grafiği HTML raporunda)" in texts
    assert "• Grafik 2: bulgu 2. (grafiği HTML raporunda)" in texts
    assert not any("Grafik 0" in text and "HTML raporunda" in text for text in texts)


def test_expected_range_never_goes_below_zero_for_nonnegative_columns() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "payment_installments": np.r_[rng.integers(1, 11, 290), [48] * 10],
        "ölçüm": np.r_[rng.exponential(50, 290), [5000] * 10],
        "kar_zarar": np.r_[rng.normal(0, 10, 290), [900] * 10],
        "Yaş": np.r_[rng.integers(20, 60, 298), [-5, 999]],
    })
    findings = {item.column: item for item in DataHealthAnalyzer().find_outliers(frame)}
    assert findings["payment_installments"].lower_bound >= 0
    assert findings["ölçüm"].lower_bound >= 0         # veride negatif yok
    assert findings["kar_zarar"].lower_bound < 0       # negatif olabilen ölçü: sınır kırpılmaz
    assert findings["Yaş"].lower_bound >= 0 and findings["Yaş"].min == -5
    assert findings["Yaş"].count == 2                  # -5 artık alt sınırın dışında sayılır


def _skewed_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "payment_value": np.r_[rng.lognormal(4, 1.2, 2990), [0.0] * 10],   # çarpık, negatif yok
        "sıcaklık": rng.normal(20, 5, 3000),                                # simetrik
        "kar_zarar": np.r_[rng.normal(0, 5, 2990), rng.normal(900, 50, 10)],  # çarpık ama negatif içerir
        "grup": ["a", "b", "c"] * 1000,
    })


def test_skewed_positive_columns_use_log_scale_iqr() -> None:
    frame = _skewed_frame()
    assert frame["payment_value"].skew() > 2 and frame["kar_zarar"].skew() > 2
    findings = {item.column: item for item in DataHealthAnalyzer().find_outliers(frame)}
    log = findings["payment_value"]
    assert log.method == "IQR (log ölçek)" and log.lower_bound >= 0
    linear_upper = frame["payment_value"].quantile(0.75) + 1.5 * (frame["payment_value"].quantile(0.75) - frame["payment_value"].quantile(0.25))
    assert log.upper_bound > linear_upper            # uzun kuyruk artık topluca aykırı sayılmıyor
    assert log.ratio < 0.02
    assert findings["kar_zarar"].method == "IQR"     # negatif içeren sütunda log uygulanmaz
    assert all(item.method == "IQR" for name, item in findings.items() if name == "sıcaklık")


def test_report_states_log_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    result = AutopilotEngine().run_full_autopilot(_skewed_frame())
    report = build_report(result, "test.csv")
    assert any("log ölçekte hesaplandı" in item for item in report.recommendations)
    page = render_html(report, embed_plotly=False)
    assert "IQR (log ölçek)" in page and "Yöntem" in page
    story = _story(monkeypatch)
    render_pdf(report)
    assert any(text.startswith("IQR (log ölçek): çarpıklığı 2'den büyük") for text in _texts(story))


def test_sampled_charts_carry_a_sample_note(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(9)
    width = rng.normal(30, 8, 4000)
    frame = pd.DataFrame({"genişlik": width, "fiyat": 3 * width + rng.normal(0, 5, 4000), "grup": rng.choice(["a", "b"], 4000)})
    result = AutopilotEngine().run_full_autopilot(frame)
    sizes = {chart.title: chart.sample_size for chart in result.charts}
    assert sizes["2B Yoğunluk Kontur Haritası"] == 2500 and sizes["Violin ve Çarpıklık Analizi"] == 2500
    report = build_report(result, "test.csv")
    story = _story(monkeypatch)
    render_pdf(report, [chart.figure for chart in result.charts])
    notes = [text for text in _texts(story) if "2.500 satırlık rastgele örneklemle" in text]
    assert len(notes) == sum(1 for size in sizes.values() if size)
    assert "2.500 satırlık rastgele örneklemle" in render_html(report, [chart.figure for chart in result.charts], embed_plotly=False)


def test_small_data_charts_have_no_sample_note(dirty_result: AutopilotResult) -> None:
    assert all(chart.sample_size is None for chart in dirty_result.charts)
