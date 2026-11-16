"""Rapor bütünlüğü: değerlendirme ayrıntıları, PDP, dönem karşılaştırması, birleştirme kaydı ve plan kaynağı."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest
from reportlab.platypus import Image

from core.automl import AutoMLEngine
from core.autopilot import AutopilotResult
from core.drift import DriftEngine
from core.executor import RejectedAction
from core.explain import ModelExplainer
from core.relations import JoinPlan, JoinStep, apply_plan
from core.report import ExecutiveReport, build_report, render_html, render_pdf
from tests.test_pdf_layout import _story, _texts

WHEN = datetime(2026, 11, 16, 12, 0)


@pytest.fixture(scope="module")
def extras(orders_result: AutopilotResult):
    cleaned = orders_result.cleaned_frame
    half = len(cleaned) // 2
    drift = DriftEngine().compare(cleaned.iloc[:half], cleaned.iloc[half:].assign(price=lambda f: f["price"] * 3),
                                  "İlk yarı", "İkinci yarı")
    explanation = ModelExplainer().explain(orders_result.automl_result)
    customers = pd.DataFrame({"customer_id": cleaned["customer_id"].unique(), "segment": "A"})
    joins = apply_plan({"orders": cleaned, "customers": customers}, JoinPlan(base="orders", steps=[JoinStep(
        table="customers", left_table="orders", left_column="customer_id", right_column="customer_id",
        mode="eşleştir", depth=1,
    )])).logs
    return drift, explanation, joins


@pytest.fixture(scope="module")
def full_report(orders_result: AutopilotResult, extras) -> ExecutiveReport:
    drift, explanation, joins = extras
    result = replace(orders_result, rejected_actions=[RejectedAction(
        column="order_status", action="drop_column", plan_reason="Gereksiz.", rejection="Sütun dolu ve değişken.",
    )])
    return build_report(result, "siparisler.csv", WHEN, drift=drift.report, explanation=explanation, join_logs=joins)


def test_report_carries_every_section_and_round_trips(full_report: ExecutiveReport) -> None:
    assert full_report.drift is not None and full_report.drift.counts["kırmızı"] >= 1
    assert full_report.pdp and full_report.join_logs and full_report.rejected_actions
    assert full_report.plan_origin is not None and full_report.plan_origin.source == "yerel kurallar"
    assert ExecutiveReport.model_validate_json(full_report.model_dump_json()) == full_report
    assert any(item.startswith("Dönem karşılaştırması (İlk yarı → İkinci yarı)") for item in full_report.headline)
    assert any("Dönem karşılaştırmasında" in item and "“price”" in item for item in full_report.recommendations)


def test_model_evaluation_details_are_reported(full_report: ExecutiveReport) -> None:
    model = full_report.model
    assert model.train_rows + model.test_rows == 300 and model.cv_folds == 5
    assert model.split_method.startswith("rastgele") and model.cv_score is not None
    page = render_html(full_report, embed_plotly=False)
    assert "Değerlendirme: rastgele" in page and "eğitim 240 / test 60 satır" in page
    assert "5 katlı çapraz doğrulamayla seçildi" in page


def test_sampled_model_is_disclosed(orders_result: AutopilotResult, orders: pd.DataFrame) -> None:
    sampled = AutoMLEngine().run(orders_result.cleaned_frame.sample(150, random_state=1), "review_score")
    report = build_report(replace(orders_result, automl_result=sampled), "x.csv", WHEN)
    assert "300 satırın 150 satırlık rastgele örneklemiyle eğitildi" in render_html(report, embed_plotly=False)


def test_html_sections(full_report: ExecutiveReport, extras) -> None:
    _, explanation, _ = extras
    page = render_html(full_report, embed_plotly=False, pdp_figures=explanation.figures)
    for heading in ("Dönem Karşılaştırması", "Model Neden Böyle Tahmin Ediyor?", "Veri Birleştirme",
                    "Uygulanmayan plan önerileri (veri koruma)"):
        assert heading in page
    assert "Temizlik planı: yerel kurallar — Gemini API anahtarı girilmedi." in page
    assert explanation.curves[0].interpretation.split("“")[0] in page
    assert page.count('class="plotly-graph-div"') == len(explanation.figures)


def test_pdf_sections_within_image_budget(monkeypatch: pytest.MonkeyPatch, full_report: ExecutiveReport,
                                          orders_result: AutopilotResult, extras) -> None:
    _, explanation, _ = extras
    story = _story(monkeypatch)
    pdf = render_pdf(full_report, [chart.figure for chart in orders_result.charts], pdp_figures=explanation.figures)
    assert pdf.startswith(b"%PDF")
    texts = _texts(story)
    for heading in ("Dönem Karşılaştırması", "Model Neden Böyle Tahmin Ediyor?", "Veri Birleştirme",
                    "Uygulanan İşlemler", "Uygulanmayan plan önerileri (veri koruma)"):
        assert heading in texts
    images = sum(isinstance(item, Image) for item in story)
    images += sum(isinstance(child, Image) for item in story for child in getattr(item, "_content", []))
    assert images + int(bool(full_report.model and full_report.model.top_features)) <= 6
