from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.columns import (
    is_identifier_name,
    is_text_series,
    name_tokens,
    parse_datetime_text,
    parse_numeric_text,
)


def test_text_detection_covers_pandas_string_dtypes() -> None:
    assert is_text_series(pd.Series(["a", None]))
    assert is_text_series(pd.Series(["a", None], dtype=object))
    assert not is_text_series(pd.Series([1.0, None]))
    assert not is_text_series(pd.Series(pd.to_datetime(["2020-01-01"])))


@pytest.mark.parametrize(("values", "expected"), [
    (["1.250,50 TL", "250,00 TL", "75 TL"], [1250.50, 250.0, 75.0]),
    (["₺1.250,50", "₺99,90"], [1250.50, 99.90]),
    (["1,250.50", "99.90", "$2,000.00"], [1250.50, 99.90, 2000.0]),
    (["58.90", "13.29", "-7.5"], [58.90, 13.29, -7.5]),
    (["%45", "12,5%"], [45.0, 12.5]),
    (["1.250 TL", "2.000 TL"], [1250.0, 2000.0]),
])
def test_numeric_text_formats(values: list[str], expected: list[float]) -> None:
    numbers, valid = parse_numeric_text(pd.Series(values))
    assert valid.all()
    assert numbers.tolist() == pytest.approx(expected)


@pytest.mark.parametrize("value", [
    "e481f51cbdc54678b7cc49136f2d6af7", "Cadde 12 No 5", "02/10/2017", "2017-10-02 10:56:33",
    "01037", "05321234567", "4543123456789012", "merhaba", "12 adet", "",
])
def test_non_numbers_are_rejected(value: str) -> None:
    numbers, valid = parse_numeric_text(pd.Series([value, "10"]))
    assert valid.tolist() == [False, True]
    assert np.isnan(numbers.iloc[0]) and numbers.iloc[1] == 10


def test_missing_values_stay_missing() -> None:
    numbers, valid = parse_numeric_text(pd.Series(["1,5", None, "2,5"]))
    assert valid.tolist() == [True, False, True]
    assert np.isnan(numbers.iloc[1])


@pytest.mark.parametrize(("values", "expected"), [
    (["2017-10-02 10:56:33", "2018-07-24 20:41:37"], ["2017-10-02 10:56:33", "2018-07-24 20:41:37"]),
    (["2017-10-02", "2018-07-24"], ["2017-10-02", "2018-07-24"]),
    (["2017/10/02", "2018/07/24"], ["2017-10-02", "2018-07-24"]),
    (["02.10.2017", "24.07.2018"], ["2017-10-02", "2018-07-24"]),
    (["02/10/2017 08:30", "24/07/2018 17:45"], ["2017-10-02 08:30", "2018-07-24 17:45"]),
])
def test_datetime_text_formats(values: list[str], expected: list[str]) -> None:
    parsed = parse_datetime_text(pd.Series(values))
    assert parsed is not None
    assert parsed.tolist() == [pd.Timestamp(item) for item in expected]


@pytest.mark.parametrize("values", [
    ["İstanbul", "Ankara"], ["2017", "2018"], ["1.250,50", "99,90"], ["e481f51c", "53cdb2fc"], [None, None],
])
def test_non_dates_are_not_parsed(values: list) -> None:
    assert parse_datetime_text(pd.Series(values, dtype=object)) is None


def test_mostly_missing_date_column_is_still_recognised() -> None:
    parsed = parse_datetime_text(pd.Series(["2017-10-02", None, None, None, "2018-07-24"]))
    assert parsed is not None and int(parsed.notna().sum()) == 2


@pytest.mark.parametrize("name", [
    "order_id", "customerId", "ID", "uuid", "TC Kimlik No", "customer_zip_code_prefix", "ürün_kodu",
    "Müşteri Adı", "ad_soyad", "Ad", "customer_name", "name", "sipariş_no",
])
def test_identifier_names(name: str) -> None:
    assert is_identifier_name(name)


@pytest.mark.parametrize("name", [
    "product_width_cm", "humidity", "paid_amount", "valid", "video_count", "product_category_name",
    "kodlama_suresi", "Şehir", "review_score", "ad_spend", "product_name_lenght",
])
def test_ordinary_names_are_not_identifiers(name: str) -> None:
    assert not is_identifier_name(name)


def test_name_tokens() -> None:
    assert name_tokens("customerZipCode_Prefix2") == ["customer", "zip", "code", "prefix"]
    assert name_tokens("Müşteri Adı") == ["müşteri", "adı"]
