from __future__ import annotations

import pandas as pd

from core.autopilot import AutopilotResult
from tests.conftest import ISO_FORMAT


def test_sample_dataset_hygiene(dirty_raw: pd.DataFrame, dirty_result: AutopilotResult) -> None:
    cleaned = dirty_result.cleaned_frame
    assert list(cleaned.columns) == ["Müşteri Adı", "Şehir", "Yaş", "Harcama"]
    assert len(cleaned) == len(dirty_raw) - 6
    assert pd.api.types.is_numeric_dtype(cleaned["Harcama"])
    assert not cleaned["Müşteri Adı"].str.startswith(" ").any()
    assert dirty_result.after_report.kalite_skoru > dirty_result.before_report.kalite_skoru
    assert len(dirty_result.audit_logs) >= 6


def test_identifier_columns_survive_cleaning(orders_raw: pd.DataFrame, orders_result: AutopilotResult) -> None:
    cleaned = orders_result.cleaned_frame
    assert cleaned["order_id"].tolist() == orders_raw["order_id"].tolist()
    assert cleaned["customer_id"].tolist() == orders_raw["customer_id"].tolist()


def test_target_values_are_not_altered(orders_raw: pd.DataFrame, orders_result: AutopilotResult) -> None:
    cleaned = orders_result.cleaned_frame
    assert cleaned["review_score"].tolist() == orders_raw["review_score"].tolist()
    assert cleaned["price"].tolist() == orders_raw["price"].tolist()


def test_missing_values_are_not_fabricated(orders_raw: pd.DataFrame, orders_result: AutopilotResult) -> None:
    cleaned = orders_result.cleaned_frame
    for column in ("review_comment_message", "order_delivered_customer_date"):
        assert int(cleaned[column].isna().sum()) == int(orders_raw[column].isna().sum())


def test_dates_are_parsed_correctly(orders_raw: pd.DataFrame, orders_result: AutopilotResult) -> None:
    cleaned = orders_result.cleaned_frame
    for column in ("order_purchase_timestamp", "order_delivered_customer_date"):
        assert pd.api.types.is_datetime64_any_dtype(cleaned[column])
        expected = pd.to_datetime(orders_raw[column], format=ISO_FORMAT)
        assert cleaned[column].equals(expected.astype(cleaned[column].dtype))


def test_integer_score_target_is_classification(orders_result: AutopilotResult) -> None:
    assert orders_result.suggested_target == "review_score"
    assert orders_result.automl_result is not None
    assert orders_result.automl_result.problem_type == "Classification"


def test_charts_are_produced(orders_result: AutopilotResult, dirty_result: AutopilotResult) -> None:
    assert orders_result.charts and dirty_result.charts
