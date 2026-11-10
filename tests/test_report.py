from __future__ import annotations

from datetime import datetime

import pandas as pd
from streamlit.testing.v1 import AppTest

from core import report as report_module
from core.autopilot import AutopilotEngine, AutopilotResult
from core.report import ExecutiveReport, build_report, render_html, render_pdf
from tests.conftest import ROOT

WHEN = datetime(2026, 11, 10, 14, 30)


def test_report_summarises_the_run(dirty_raw: pd.DataFrame, dirty_result: AutopilotResult) -> None:
    report = build_report(dirty_result, "kirli_musteri_verisi.csv", WHEN)
    assert report.generated_at == "10.11.2026 14:30"
    assert (report.rows_before, report.rows_after) == (len(dirty_raw), len(dirty_raw) - 6)
    assert len(report.actions) == len(dirty_result.audit_logs)
    assert report.headline and report.recommendations
    assert any("mükerrer" in item for item in report.recommendations)
    assert [item.column for item in report.outliers] == ["Yaş"]
    assert report.model is not None and report.model.target == "Harcama"
    assert report.model.problem_type == "Regresyon"


def test_report_is_serialisable_and_free_of_raw_values(dirty_result: AutopilotResult) -> None:
    payload = build_report(dirty_result, "kirli.csv", WHEN).model_dump_json()
    assert ExecutiveReport.model_validate_json(payload).dataset_name == "kirli.csv"
    for name in ("Ayşe", "Mehmet", "Fatma", "Zeynep"):
        assert name not in payload


def test_report_states_when_model_does_not_beat_baseline(dirty_result: AutopilotResult) -> None:
    report = build_report(dirty_result, "kirli.csv", WHEN)
    assert report.model is not None and not report.model.beats_baseline
    assert any("geçemedi" in item for item in report.headline)
    assert any("kullanılmamalı" in item for item in report.recommendations)


def test_report_for_learnable_target(orders_result: AutopilotResult) -> None:
    report = build_report(orders_result, "siparisler.csv", WHEN)
    assert report.model is not None and report.model.problem_type == "Sınıflandırma"
    assert {item.column for item in report.missing_columns} == {"review_comment_message", "order_delivered_customer_date"}
    assert any("review_comment_message" in item for item in report.recommendations)
    assert report.model.leaderboard and report.model.baseline_score > 0


def test_html_contains_sections_and_charts(orders_result: AutopilotResult) -> None:
    report = build_report(orders_result, "siparisler.csv", WHEN)
    figures = [chart.figure for chart in orders_result.charts]
    page = render_html(report, figures, orders_result.automl_result.figure, embed_plotly=False)
    for heading in ("Yönetici Özeti", "Öneriler", "Öne Çıkan Grafikler", "Tahmin Modeli", "Veri Kalitesi", "Uygulanan İşlemler"):
        assert heading in page
    assert page.count('class="plotly-graph-div"') == len(figures) + 1
    assert page.count("cdn.plot.ly") == 1
    assert page.startswith("<!doctype html>") and 'lang="tr"' in page


def test_html_escapes_data_derived_text() -> None:
    hostile = "<script>alert(1)</script>"
    frame = pd.DataFrame({hostile: ["sabit"] * 30, "değer": range(30), "grup": ["a", "b", "c"] * 10})
    result = AutopilotEngine().run_full_autopilot(frame)
    report = build_report(result, f"{hostile}.csv", WHEN)
    assert any(action.column == hostile for action in report.actions)
    page = render_html(report, [chart.figure for chart in result.charts], embed_plotly=False)
    assert hostile not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_pdf_is_generated_with_and_without_model(orders_result: AutopilotResult) -> None:
    with_model = render_pdf(build_report(orders_result, "siparisler.csv", WHEN))
    assert with_model.startswith(b"%PDF") and len(with_model) > 5_000
    reference = AutopilotEngine().run_full_autopilot(
        pd.DataFrame({"şehir": ["İstanbul", "Ankara", "İzmir"] * 10, "ilçe": ["Kadıköy", "Çankaya", "Konak"] * 10})
    )
    report = build_report(reference, "şehirler.csv", WHEN)
    assert report.model is None
    assert render_pdf(report).startswith(b"%PDF")


def test_pdf_falls_back_to_ascii_when_no_turkish_font(monkeypatch, orders_result: AutopilotResult) -> None:
    assert report_module._ascii_safe("Şağlık İşi → ölçüm") == "Saglik Isi -> ölçüm"
    monkeypatch.setattr(report_module, "_font_candidates", lambda: [])
    report_module._pdf_fonts.cache_clear()
    try:
        assert report_module._pdf_fonts()[0] == "Helvetica"
        assert render_pdf(build_report(orders_result, "siparişler.csv", WHEN)).startswith(b"%PDF")
    finally:
        report_module._pdf_fonts.cache_clear()


def test_app_builds_report_downloads() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()
    app.sidebar.selectbox[0].select("kirli_musteri_verisi.csv").run()
    app.button(key="report_button").click().run()
    assert not app.exception and not app.error
    files = app.session_state["report_files"]
    assert files["pdf"].startswith(b"%PDF")
    assert "Yönetici Özeti" in files["html"].decode("utf-8")
