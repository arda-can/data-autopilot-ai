from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest

from core.autopilot import AutopilotEngine
from core.chart_export import can_export, chart_png
from core.executor import ActionExecutor
from core.formatting import decimal_tr, display_frame, localize_figure
from core.health import DataHealthAnalyzer
from core.planner import ColumnPlan
from core.report import build_report, render_pdf
from tests.conftest import ROOT


def test_automatic_drop_reason_does_not_claim_user_approval() -> None:
    frame = pd.DataFrame({"Notlar": [None] * 19 + ["not"], "değer": range(20)})
    executor = ActionExecutor()
    executor.execute_plan(frame, [ColumnPlan(column="Notlar", action="drop_column", reason="%90 üzerinde boş.")])
    reason = executor.audit_logs[0].reason
    assert "onay" not in reason and "%90'dan fazlası boş" in reason and "%95,0" in reason
    executor.drop_column(frame, "değer", "İlgisiz sütun.")
    assert "İlgisiz sütun" in executor.audit_logs[-1].reason


@pytest.mark.parametrize("column", ["Yaş", "yaş", "BirimFiyat", "sale_price", "Adet", "stock_quantity"])
def test_negative_domain_values_are_reported_for_small_discrete_columns(column: str) -> None:
    frame = pd.DataFrame({column: [-5, 0, 20, None], "sıcaklık": [-20, 1, 5, 6]})
    original = frame.copy(deep=True)
    report = DataHealthAnalyzer().analyze(frame)
    assert [(item.column, item.count) for item in report.domain_findings] == [(column, 1)]
    assert any("negatif" in issue for issue in report.sorunlar)
    pd.testing.assert_frame_equal(frame, original)


def test_domain_tokens_do_not_match_unrelated_substrings() -> None:
    frame = pd.DataFrame({"stage": [-2, 3], "price_id": [-1, 2], "longitude": [-50, 10]})
    assert not DataHealthAnalyzer().find_domain_violations(frame)


@pytest.mark.parametrize("categorical", [False, True])
def test_category_variants_reported_without_modifying_values(categorical: bool) -> None:
    frame = pd.DataFrame({"şehir": ["sao paulo", "são paulo", "SÃO PAULO", "İZMİR", "izmir", None]})
    if categorical:
        frame["şehir"] = frame["şehir"].astype("category")
    original = frame.copy(deep=True)
    finding = DataHealthAnalyzer().analyze(frame).category_variants[0]
    assert (finding.group_count, finding.affected_count) == (2, 5)
    pd.testing.assert_frame_equal(frame, original)


def test_domain_and_category_advice_reaches_report_without_raw_values() -> None:
    frame = pd.DataFrame({"Yaş": [-5, 20, 30] * 10, "şehir": ["sao paulo", "são paulo", "SÃO PAULO"] * 10})
    result = AutopilotEngine().run_full_autopilot(frame)
    assert result.cleaned_frame["Yaş"].min() == -5
    assert set(result.cleaned_frame["şehir"]) == set(frame["şehir"])
    report = build_report(result, "test.csv")
    assert any("negatif" in item for item in report.recommendations)
    assert any("kategori kopyası" in item for item in report.recommendations)
    assert "são paulo" not in report.model_dump_json()


def test_heatmap_with_three_metrics_and_sampled_parallel_coordinates() -> None:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(2200, 4)), columns=["a", "b", "c", "d"])
    for number in (2, 3, 4):
        charts, insights = AutopilotEngine()._select_charts(frame.iloc[:, :number])
        titles = [chart.title for chart in charts]
        assert ("Korelasyon Isı Haritası" in titles) == (number >= 3)
        assert len(charts) <= 6 and len(insights) == len(charts)
        if number == 4:
            parallel = next(chart.figure for chart in charts if chart.title == "Paralel Koordinatlar")
            assert len(parallel.data[0].dimensions[0].values) == 2000
            assert any("," in text for text in parallel.data[0].dimensions[0].ticktext)


def test_turkish_numbers_in_tables_axes_and_heatmap() -> None:
    frame = pd.DataFrame({"değer": [1234.5, np.nan]})
    assert decimal_tr(1234.5) == "1.234,5"
    assert display_frame(frame).iloc[0, 0] == "1.234,500"
    assert pd.api.types.is_float_dtype(frame["değer"])
    figure = localize_figure(px.imshow([[1.0, 0.25], [0.25, 1.0]], text_auto=True))
    assert figure.layout.separators == ",."
    assert figure.data[0].text[0][1] == "0,25"
    assert figure.data[0].texttemplate == "%{text}"


def test_kaleido_attempt_uses_light_theme_and_falls_back(monkeypatch) -> None:
    attempts = []
    def blocked(self, **kwargs):
        attempts.append(self)
        raise RuntimeError("Kaleido engellendi")
    monkeypatch.setattr(go.Figure, "to_image", blocked)
    figure = px.scatter(x=[1.5, 2.5], y=[3.5, 4.5], template="plotly_dark")
    png = chart_png(figure)
    assert png.startswith(b"\x89PNG")
    assert attempts[0].layout.paper_bgcolor == "white"
    assert attempts[0].layout.separators == ",."
    assert figure.layout.paper_bgcolor is None


@pytest.mark.parametrize("figure", [px.imshow([[1, 0.5], [0.5, 1]]),
    px.violin(y=[1, 2, 3, 4]), px.density_contour(x=[1, 2, 3], y=[3, 1, 2], marginal_x="histogram", marginal_y="violin"),
    px.pie(names=["a", "b"], values=[2, 3]), px.area(x=["a", "b"], y=[1, 2]),
    px.sunburst(names=["a", "b"], parents=["", ""], values=[2, 3]),
    go.Figure(go.Scatterpolar(theta=["a", "b", "a"], r=[0, 1, 0], fill="toself"))])
def test_matplotlib_supports_selected_chart_types(monkeypatch, figure) -> None:
    monkeypatch.setattr(go.Figure, "to_image", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    assert chart_png(figure).startswith(b"\x89PNG")


def test_pdf_graph_limit_and_skipped_types(monkeypatch, dirty_result) -> None:
    import core.chart_export as export
    from core.report import ReportChart
    from reportlab.platypus import SimpleDocTemplate, Table, Paragraph
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from core.report import _pdf_fonts

    report = build_report(dirty_result, "test.csv")
    report.issues = ["Sabit sütunlar: Notlar.", "%90 üzeri boş sütunlar: Notlar."]
    assert report.model is not None
    report.model.leaderboard = [{"Model": "HistGradientBoostingRegressor", "R2": 0.123, "MAE": 1.5, "RMSE": 2.5}]
    skipped = [px.scatter_3d(x=[1], y=[2], z=[3]), px.parallel_coordinates(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))]
    assert not any(can_export(figure) for figure in skipped)
    figures = [*skipped, *[px.scatter(x=[1, 2], y=[3, 4]) for _ in range(8)]]
    report.charts = [ReportChart(title=str(i), reason="", insight="bulgu") for i in range(len(figures))]
    calls, story, model_widths = [], [], []
    png = export._matplotlib_png(figures[-1])
    monkeypatch.setattr(export, "chart_png", lambda figure: calls.append(figure) or png)
    original = SimpleDocTemplate.build
    def capture(self, flowables, **kwargs):
        story.extend(flowables)
        for item in flowables:
            if (isinstance(item, Table) and isinstance(item._cellvalues[0][0], Paragraph)
                    and item._cellvalues[0][0].getPlainText() == "Model"):
                model_widths.append(item._colWidths[0])
        return original(self, flowables, **kwargs)
    monkeypatch.setattr(SimpleDocTemplate, "build", capture)
    pdf = render_pdf(report, figures)
    assert pdf.startswith(b"%PDF") and b"/Subtype /Image" in pdf
    assert len(calls) + int(bool(report.model.top_features)) == 6
    paragraphs = [item.getPlainText() for item in story if isinstance(item, Paragraph)]
    assert "• Sabit sütunlar: Notlar." in paragraphs
    assert "• %90 üzeri boş sütunlar: Notlar." in paragraphs
    regular, _, _ = _pdf_fonts()
    assert stringWidth("HistGradientBoostingRegressor", regular, 8.5) + 12 <= model_widths[0]


def test_app_uses_new_width_parameter() -> None:
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "use_container_width" not in source
    assert 'width="stretch"' in source
