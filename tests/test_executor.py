from __future__ import annotations

import pandas as pd
import pytest

from core.executor import ActionExecutor
from tests.conftest import ISO_FORMAT


def test_methods_do_not_mutate_input(dirty: pd.DataFrame) -> None:
    snapshot = dirty.copy(deep=True)
    executor = ActionExecutor()
    executor.drop_constant_columns(dirty)
    executor.drop_high_null_columns(dirty)
    executor.remove_duplicates(dirty)
    executor.strip_whitespace(dirty)
    executor.clean_currency_and_numeric_strings(dirty)
    executor.parse_datetime_columns(dirty)
    executor.impute_median(dirty, "Yaş")
    executor.impute_mode(dirty, "Neredeyse Boş")
    executor.cap_outliers_iqr(dirty)
    pd.testing.assert_frame_equal(dirty, snapshot)


def test_every_change_is_logged(dirty: pd.DataFrame) -> None:
    executor = ActionExecutor()
    executor.strip_whitespace(dirty)
    assert [log.column for log in executor.audit_logs] == ["Müşteri Adı"]
    log = executor.audit_logs[0]
    assert log.affected_count > 0 and log.action and log.reason


def test_unknown_action_raises(dirty: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        ActionExecutor().execute_plan(dirty, [{"column": "Yaş", "action": "run_python"}])


def test_turkish_currency_is_parsed(dirty: pd.DataFrame) -> None:
    cleaned = ActionExecutor().clean_currency_and_numeric_strings(dirty)
    assert pd.api.types.is_numeric_dtype(cleaned["Harcama"])
    row = dirty.index[dirty["Harcama"] == "1.250,50 TL"][0]
    assert cleaned.loc[row, "Harcama"] == pytest.approx(1250.50)


def test_hash_ids_stay_text(orders: pd.DataFrame) -> None:
    executor = ActionExecutor()
    cleaned = executor.clean_currency_and_numeric_strings(orders)
    assert cleaned["order_id"].tolist() == orders["order_id"].tolist()
    assert cleaned["customer_id"].tolist() == orders["customer_id"].tolist()
    assert executor.audit_logs == []


def test_text_containing_digits_is_not_converted() -> None:
    frame = pd.DataFrame({
        "adres": ["Cadde 12 No 5", "Sokak 7 Daire 3", "Bulvar 44"],
        "tarih": ["02/10/2017", "24/07/2018", "08/08/2018"],
        "posta_kutusu": ["01037", "04530", "09790"],
    })
    cleaned = ActionExecutor().clean_currency_and_numeric_strings(frame)
    pd.testing.assert_frame_equal(cleaned, frame)


def test_us_and_plain_decimal_formats() -> None:
    frame = pd.DataFrame({
        "us": ["1,250.50", "99.90", "2,000.00"],
        "ondalik": ["58.90", "13.29", "7.5"],
    })
    cleaned = ActionExecutor().clean_currency_and_numeric_strings(frame)
    assert cleaned["us"].tolist() == pytest.approx([1250.50, 99.90, 2000.00])
    assert cleaned["ondalik"].tolist() == pytest.approx([58.90, 13.29, 7.5])


def test_iso_dates_are_not_day_month_swapped(orders: pd.DataFrame) -> None:
    cleaned = ActionExecutor().parse_datetime_columns(orders)
    column = "order_purchase_timestamp"
    assert pd.api.types.is_datetime64_any_dtype(cleaned[column])
    expected = pd.to_datetime(orders[column], format=ISO_FORMAT)
    assert (cleaned[column] == expected).all()


def test_day_first_dates_are_parsed() -> None:
    frame = pd.DataFrame({"tarih": ["02.10.2017", "24.07.2018", "13.01.2019"]})
    cleaned = ActionExecutor().parse_datetime_columns(frame)
    assert cleaned["tarih"].tolist() == [
        pd.Timestamp(2017, 10, 2), pd.Timestamp(2018, 7, 24), pd.Timestamp(2019, 1, 13),
    ]


def test_plain_text_is_not_parsed_as_date(orders: pd.DataFrame) -> None:
    cleaned = ActionExecutor().parse_datetime_columns(orders)
    for column in ("order_id", "order_status", "product_category_name"):
        assert not pd.api.types.is_datetime64_any_dtype(cleaned[column])
