"""Zaman serisinin yönetici raporuna (HTML ve PDF) ve PDF grafik yedeğine yansıması."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from reportlab.platypus import Image

from core.autopilot import AutopilotResult
from core.chart_export import _mpl_color, chart_png
from core.report import ExecutiveReport, build_report, render_html, render_pdf
from core.timeseries import TimeSeriesEngine, TimeSeriesResult
from tests.test_pdf_layout import _story, _texts
from tests.test_timeseries import _events, _trend_weekly_counts

WHEN = datetime(2026, 11, 16, 10, 0)


@pytest.fixture(scope="module")
def series_result() -> TimeSeriesResult:
    return TimeSeriesEngine().analyze(_events(_trend_weekly_counts()))


@pytest.fixture
def result_with_series(orders_result: AutopilotResult, series_result: TimeSeriesResult) -> AutopilotResult:
    return replace(orders_result, timeseries=series_result, timeseries_message="")


def _figures(series: TimeSeriesResult) -> list:
    return [series.forecast_figure, series.seasonal_figure]


def test_report_carries_timeseries_summary(result_with_series: AutopilotResult, series_result: TimeSeriesResult) -> None:
    report = build_report(result_with_series, "siparisler.csv", WHEN)
    assert report.timeseries == series_result.summary
    assert series_result.summary.trend_text in report.headline
    assert ExecutiveReport.model_validate_json(report.model_dump_json()).timeseries == series_result.summary


def test_anomalies_become_a_recommendation(result_with_series: AutopilotResult, series_result: TimeSeriesResult) -> None:
    report = build_report(result_with_series, "siparisler.csv", WHEN)
    top = series_result.summary.anomalies[0]
    advice = next(item for item in report.recommendations if "olağan dışı dönemler" in item)
    assert top.label in advice and "doğrulanmalı" in advice


def test_failed_baseline_is_a_recommendation(result_with_series: AutopilotResult, series_result: TimeSeriesResult) -> None:
    weak = replace(series_result, summary=series_result.summary.model_copy(update={"beats_baseline": False}))
    report = build_report(replace(result_with_series, timeseries=weak), "x.csv", WHEN)
    assert any("naif yöntemi anlamlı biçimde geçemedi" in item for item in report.recommendations)


def test_html_has_timeseries_section(result_with_series: AutopilotResult, series_result: TimeSeriesResult) -> None:
    report = build_report(result_with_series, "siparisler.csv", WHEN)
    without = render_html(report, embed_plotly=False)
    page = render_html(report, embed_plotly=False, timeseries_figures=_figures(series_result))
    assert "<h2>Zaman Serisi</h2>" in page and series_result.summary.verdict in page
    assert "Olağan dışı dönemler" in page and series_result.summary.anomalies[0].label in page
    assert page.count('class="plotly-graph-div"') - without.count('class="plotly-graph-div"') == 2
    assert "Haftalık döngü" in page and "Frekans:</b> günlük" in page


def test_html_escapes_timeseries_text(orders_result: AutopilotResult) -> None:
    hostile = "<script>x</script>"
    frame = _events(_trend_weekly_counts(days=200)).rename(columns={"sipariş_tarihi": hostile})
    series = TimeSeriesEngine().analyze(frame)
    report = build_report(replace(orders_result, timeseries=series), "x.csv", WHEN)
    page = render_html(report, embed_plotly=False)
    assert hostile not in page and "&lt;script&gt;x&lt;/script&gt;" in page


def test_failure_message_is_reported(orders_result: AutopilotResult) -> None:
    message = "Zaman serisi analizi yapılamadı: en az 12 dönem gerekir."
    report = build_report(replace(orders_result, timeseries=None, timeseries_message=message), "x.csv", WHEN)
    assert report.timeseries is None and report.timeseries_message == message
    assert message in render_html(report, embed_plotly=False)


def test_pdf_has_timeseries_section_within_image_budget(
    monkeypatch: pytest.MonkeyPatch, result_with_series: AutopilotResult, series_result: TimeSeriesResult
) -> None:
    report = build_report(result_with_series, "siparisler.csv", WHEN)
    story = _story(monkeypatch)
    pdf = render_pdf(report, [chart.figure for chart in result_with_series.charts], timeseries_figures=_figures(series_result))
    assert pdf.startswith(b"%PDF")
    texts = _texts(story)
    assert "Zaman Serisi" in texts and "Olağan dışı dönemler" in texts
    assert any(text.startswith("Not: Analiz") or text.startswith("Not: Son") for text in texts) or not series_result.summary.notes
    images = [item for item in story if isinstance(item, Image)]
    images += [child for item in story for child in getattr(item, "_content", []) if isinstance(child, Image)]
    assert len(images) + int(bool(report.model and report.model.top_features)) <= 6
    assert sum(1 for item in story if isinstance(item, Image)) == 2  # tahmin + mevsimsellik


def test_forecast_chart_renders_with_matplotlib_fallback(monkeypatch: pytest.MonkeyPatch, series_result: TimeSeriesResult) -> None:
    def blocked(self, **kwargs):
        raise RuntimeError("Kaleido yok")

    monkeypatch.setattr(go.Figure, "to_image", blocked)
    for figure in _figures(series_result):
        png = chart_png(figure)
        assert png.fallback and png.startswith(b"\x89PNG")


def test_fallback_band_fills_between_bounds(monkeypatch: pytest.MonkeyPatch, series_result: TimeSeriesResult) -> None:
    from matplotlib.axes import Axes

    calls: list[tuple] = []
    original = Axes.fill_between

    def spy(self, x, y1, y2=0, **kwargs):
        calls.append((np.asarray(y1, dtype=float) if np.ndim(y1) else y1, np.asarray(y2, dtype=float)))
        return original(self, x, y1, y2, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", spy)
    from core.chart_export import _matplotlib_png

    _matplotlib_png(series_result.forecast_figure)
    lower = np.array([point.lower for point in series_result.summary.forecast])
    upper = np.array([point.upper for point in series_result.summary.forecast])
    assert len(calls) == 1
    assert np.allclose(calls[0][0], lower, atol=1e-3) and np.allclose(calls[0][1], upper, atol=1e-3)


def test_mpl_color_conversion() -> None:
    assert _mpl_color("rgba(217,119,6,0.22)", None) == pytest.approx((217 / 255, 119 / 255, 6 / 255, 0.22))
    assert _mpl_color("rgb(0, 0, 255)", None) == pytest.approx((0, 0, 1, 1))
    assert _mpl_color("#1d5fd1", None) == "#1d5fd1"
    assert _mpl_color(None, "red") == "red"
    assert _mpl_color(["#fff", "rgba(0,0,0,0.5)"], None)[1] == pytest.approx((0, 0, 0, 0.5))
