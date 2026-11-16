"""Sohbette filtre, tarih kırılımı ve sorgu şeffaflığı."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.autopilot import AutopilotResult
from core.chat import DataChatEngine, QueryFilter, StructuredDataQuery


@pytest.fixture(scope="module")
def customers(dirty_result: AutopilotResult) -> pd.DataFrame:
    return dirty_result.cleaned_frame


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    days = pd.date_range("2017-01-01", "2018-12-31", freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "tarih": days.repeat(2),
        "durum": rng.choice(["teslim", "iptal"], len(days) * 2, p=[0.9, 0.1]),
        "tutar": rng.uniform(10, 100, len(days) * 2),
    })


def _ask(frame: pd.DataFrame, question: str):
    engine = DataChatEngine()
    table, summary, figure, query = engine.answer(frame, question)
    return table, summary, query, engine.explain(frame, query)


def test_category_filter_without_group_gives_single_value(customers: pd.DataFrame) -> None:
    table, summary, query, code = _ask(customers, "İstanbul için ortalama harcama")
    assert [(item.column, item.operator, item.value) for item in query.filters] == [("Şehir", "==", "İstanbul")]
    assert query.group_col is None and query.target_col == "Harcama"
    expected = customers.loc[customers["Şehir"] == "İstanbul", "Harcama"].mean()
    assert table["Değer"].iloc[0] == pytest.approx(expected)
    assert "Filtre: Şehir == İstanbul" in summary
    assert code.sql == 'SELECT AVG("Harcama") AS deger\nFROM veri\nWHERE "Şehir" = \'İstanbul\';'


def test_numeric_filter_with_grouping(customers: pd.DataFrame) -> None:
    table, _, query, code = _ask(customers, "Yaşı 40'tan büyük müşterilerin şehirlere göre ortalama harcaması")
    assert [(item.column, item.operator) for item in query.filters] == [("Yaş", ">")]
    assert (query.group_col, query.target_col) == ("Şehir", "Harcama")
    expected = customers[customers["Yaş"] > 40].groupby("Şehir")["Harcama"].mean()
    assert dict(zip(table["Şehir"], table["Değer"])) == pytest.approx(expected.to_dict())
    assert code.rows_used == int((customers["Yaş"] > 40).sum())
    assert "df = df[(df['Yaş'] > 40.0)]" in code.pandas


def test_several_category_values_and_total_is_not_top_n(customers: pd.DataFrame) -> None:
    table, _, query, code = _ask(customers, "Ankara ve İzmir için toplam harcama")
    assert query.operation == "groupby_agg" and query.agg_func == "sum"  # "toplam" içindeki "top" yanıltmaz
    assert query.filters[0].operator == "in" and query.filters[0].values == ["Ankara", "İzmir"]
    assert set(table["Şehir"]) == {"Ankara", "İzmir"}
    assert "IN ('Ankara', 'İzmir')" in code.sql


def test_year_filter_with_monthly_breakdown(events: pd.DataFrame) -> None:
    table, summary, query, code = _ask(events, "2018'de aylık kayıt sayısı")
    assert query.operation == "time_trend" and query.time_grain == "month" and query.agg_func == "count"
    assert list(table["Dönem"])[:2] == ["Ocak 2018", "Şubat 2018"] and len(table) == 12
    assert table["Kayıt Sayısı"].sum() == int((events["tarih"].dt.year == 2018).sum())
    assert "DATE_TRUNC('month', \"tarih\")" in code.sql and "TIMESTAMP '2018-01-01 00:00:00'" in code.sql


def test_month_name_filter_with_daily_breakdown(events: pd.DataFrame) -> None:
    table, _, query, _ = _ask(events, "Kasım 2017'de günlük kayıt sayısı")
    assert [item.value for item in query.filters] == ["2017-11-01", "2017-12-01"]
    assert len(table) == 30 and table["Dönem"].iloc[0] == "01.11.2017"


def test_time_trend_of_a_measure_with_category_filter(events: pd.DataFrame) -> None:
    table, _, query, _ = _ask(events, "iptal olanların aylık ortalama tutar")
    assert query.filters[0].column == "durum" and query.filters[0].value == "iptal"
    assert (query.target_col, query.agg_func) == ("tutar", "mean")
    expected = events[events["durum"] == "iptal"].groupby(events["tarih"].dt.to_period("M"))["tutar"].mean()
    assert table["Değer"].to_numpy() == pytest.approx(expected.to_numpy())


def test_no_matching_rows_raises(customers: pd.DataFrame) -> None:
    query = StructuredDataQuery(operation="groupby_agg", target_col="Harcama",
                                filters=[QueryFilter(column="Yaş", operator=">", value="5000")], explanation="x")
    with pytest.raises(ValueError, match="Filtrelere uyan kayıt yok"):
        DataChatEngine().apply_filters(customers, query)


def test_filter_validation(customers: pd.DataFrame) -> None:
    engine = DataChatEngine()
    unknown = StructuredDataQuery(operation="value_counts", group_col="Şehir",
                                  filters=[QueryFilter(column="yok", value="1")], explanation="x")
    with pytest.raises(ValueError, match="geçersiz sütun"):
        engine.apply_filters(customers, unknown)
    text_compare = StructuredDataQuery(operation="value_counts", group_col="Şehir",
                                       filters=[QueryFilter(column="Şehir", operator=">", value="A")], explanation="x")
    with pytest.raises(ValueError, match="metin sütunu"):
        engine.apply_filters(customers, text_compare)
    not_a_number = StructuredDataQuery(operation="value_counts", group_col="Şehir",
                                       filters=[QueryFilter(column="Yaş", operator=">", value="çok")], explanation="x")
    with pytest.raises(ValueError, match="sayı değil"):
        engine.apply_filters(customers, not_a_number)


def test_sql_escapes_quotes_and_input_is_not_mutated(customers: pd.DataFrame) -> None:
    frame = customers.assign(**{'Ad "takma"': "x"})
    snapshot = frame.copy(deep=True)
    query = StructuredDataQuery(operation="value_counts", group_col='Ad "takma"',
                                filters=[QueryFilter(column="Şehir", value="O'Brien")], explanation="x")
    code = DataChatEngine().explain(frame.assign(Şehir="O'Brien"), query)
    assert "'O''Brien'" in code.sql and '"Ad ""takma"""' in code.sql
    pd.testing.assert_frame_equal(frame, snapshot)


def test_plain_questions_keep_previous_behaviour(customers: pd.DataFrame) -> None:
    _, _, query, _ = _ask(customers, "Şehirlere göre ortalama harcama nedir?")
    assert query.filters == [] and (query.operation, query.group_col) == ("groupby_agg", "Şehir")
    _, _, query, code = _ask(customers, "en yüksek 5 harcama")
    assert query.operation == "top_n" and query.limit == 5 and code.sql.endswith("LIMIT 5;")
