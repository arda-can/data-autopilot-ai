"""Müşteri analizi: elle doğrulanabilir küçük örneklerle RFM, kohort ve tarihsel değer tanımları."""

from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
import pytest

from core.autopilot import AutopilotEngine, AutopilotResult
from core.customers import (
    SEGMENT_RULES, CustomerAnalyzer, CustomerColumns, _score, cohort_retention, detect_customer_columns,
    historical_value, rfm_table, transactions_from,
)
from core.demo_data import DEMO_FILENAME, write_demo_data
from core.ingestion import read_local_csv
from core.report import build_report, render_html, render_pdf

COLUMNS = CustomerColumns(customer="müşteri_no", date="tarih", amount="tutar")


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["müşteri_no", "tarih", "tutar"]).assign(tarih=lambda f: pd.to_datetime(f["tarih"]))


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> AutopilotResult:
    path = write_demo_data(tmp_path_factory.mktemp("musteri") / DEMO_FILENAME)
    return AutopilotEngine().run_full_autopilot(read_local_csv(path))


def test_rfm_values_follow_definitions() -> None:
    frame = _frame([
        ("A", "2025-01-01", 100), ("A", "2025-03-01", 50),
        ("B", "2025-02-15", 500),
        ("C", "2025-03-10", 20), ("C", "2025-03-10", 30), ("C", "2025-01-05", 10),
    ])
    transactions, excluded = transactions_from(frame, COLUMNS)
    table, analysis_date, _ = rfm_table(transactions)
    assert excluded == 0 and analysis_date == pd.Timestamp("2025-03-11")
    rows = table.set_index("müşteri")
    assert rows.loc["A", ["yakınlık_gün", "sıklık", "parasal"]].tolist() == [10, 2, 150]
    assert rows.loc["B", ["yakınlık_gün", "sıklık", "parasal"]].tolist() == [24, 1, 500]
    assert rows.loc["C", ["yakınlık_gün", "sıklık", "parasal"]].tolist() == [1, 3, 60]


def test_tied_values_get_the_same_score() -> None:
    scores = _score(pd.Series([1, 1, 1, 1, 1, 1, 2, 2, 3, 5]))
    assert scores.iloc[:6].nunique() == 1 and scores.iloc[0] == 1
    assert scores.iloc[6] == scores.iloc[7] and scores.iloc[-1] == 5
    recency = _score(pd.Series([3, 10, 40]), higher_is_better=False)  # az gün = yakın = yüksek skor
    assert recency.iloc[0] > recency.iloc[1] > recency.iloc[2]


def test_every_recency_frequency_pair_has_exactly_one_segment() -> None:
    for recency, frequency in itertools.product(range(1, 6), repeat=2):
        matches = [name for pattern, name, _ in SEGMENT_RULES if re.fullmatch(pattern, f"{recency}{frequency}")]
        assert len(matches) == 1, (recency, frequency, matches)


def test_invalid_amounts_are_excluded_and_orders_merged() -> None:
    frame = _frame([("A", "2025-01-01", 100), ("A", "2025-01-01", 50), ("B", "2025-01-02", -10),
                    ("C", "2025-01-03", np.nan), (None, "2025-01-04", 10)])
    frame["sipariş_no"] = ["S1", "S1", "S2", "S3", "S4"]
    transactions, excluded = transactions_from(frame, COLUMNS.model_copy(update={"order": "sipariş_no"}))
    assert excluded == 3
    assert transactions[["customer", "amount"]].values.tolist() == [["A", 150.0]]


def test_cohort_retention_leaves_unobserved_cells_empty() -> None:
    frame = _frame([
        ("A", "2025-01-05", 10), ("A", "2025-02-07", 10),
        ("B", "2025-01-20", 10),
        ("C", "2025-02-01", 10), ("C", "2025-03-03", 10),
        ("D", "2025-03-15", 10),
    ])
    matrix, sizes, curve = cohort_retention(transactions_from(frame, COLUMNS)[0])
    assert sizes.to_dict() == {"2025-01": 2, "2025-02": 1, "2025-03": 1}
    assert matrix.loc["2025-01", 1] == 0.5 and matrix.loc["2025-01", 2] == 0.0
    assert matrix.loc["2025-02", 1] == 1.0 and np.isnan(matrix.loc["2025-02", 2])
    assert np.isnan(matrix.loc["2025-03", 1])
    assert curve[1] == pytest.approx((0.5 * 2 + 1.0 * 1) / 3)  # yalnızca gözlenen kohortlar, büyüklükle ağırlıklı


def test_historical_value_uses_only_fully_observed_customers() -> None:
    frame = _frame([
        ("A", "2025-01-01", 100), ("A", "2025-02-15", 50), ("A", "2025-05-01", 999),
        ("B", "2025-01-10", 40),
        ("C", "2025-04-20", 70),  # son tarih 1 Mayıs: C için 3 ay gözlenmedi
    ])
    rows, _, _ = historical_value(transactions_from(frame, COLUMNS)[0], horizons=(3,))
    assert rows[0].customers == 2
    assert rows[0].revenue_per_customer == pytest.approx((150 + 40) / 2)  # 1 Nisan sonrası 999 girmez
    assert rows[0].orders_per_customer == pytest.approx(1.5)


def test_window_limits_rfm_but_not_cohorts() -> None:
    frame = _frame([("A", "2024-01-01", 100), ("A", "2025-06-01", 10), ("B", "2024-03-01", 50),
                    *[(f"K{i}", "2025-06-15", 5) for i in range(3)]])
    analysis = CustomerAnalyzer().analyze(frame, COLUMNS, window_label="Son 6 ay")
    assert set(analysis.rfm_table["müşteri"]) == {"A", "K0", "K1", "K2"}
    assert analysis.rfm_table.set_index("müşteri").loc["A", "parasal"] == 10
    assert "2024-03" in analysis.cohort_matrix.index


def test_detection_on_demo_and_refusal_without_customer(demo_result: AutopilotResult) -> None:
    columns = detect_customer_columns(demo_result.cleaned_frame)
    assert columns == CustomerColumns(customer="müşteri_no", date="sipariş_tarihi", amount="tutar", order="sipariş_no")
    assert detect_customer_columns(demo_result.cleaned_frame.drop(columns=["müşteri_no"])) is None


def test_demo_story_is_visible_in_customer_analysis(demo_result: AutopilotResult) -> None:
    analysis = CustomerAnalyzer().analyze(demo_result.cleaned_frame, detect_customer_columns(demo_result.cleaned_frame))
    summary = analysis.summary
    assert summary.repeat_customer_share > 0.3 and summary.excluded_rows == 4  # 4 negatif tutar
    assert summary.retention[1] > summary.retention[3] > summary.retention[6]
    assert [row.horizon_months for row in summary.clv] == [3, 6, 12]
    assert summary.clv[0].revenue_per_customer < summary.clv[2].revenue_per_customer
    assert sum(item.customers for item in summary.segments) == summary.customers
    assert sum(item.revenue_share for item in summary.segments) == pytest.approx(1.0)


def test_report_customer_section_has_no_customer_ids(demo_result: AutopilotResult) -> None:
    analysis = CustomerAnalyzer().analyze(demo_result.cleaned_frame, detect_customer_columns(demo_result.cleaned_frame))
    report = build_report(demo_result, "demo.csv", customers=analysis.summary)
    page = render_html(report, embed_plotly=False)
    assert "<h2>Müşteri Analizi</h2>" in page and "Kış uykusunda" in page
    assert "MUS-0" not in page and "Tarihsel müşteri değeri" in page
    assert any("tekrar eden" not in item and "birden fazla işlem" in item for item in report.headline)
    assert render_pdf(report).startswith(b"%PDF")
