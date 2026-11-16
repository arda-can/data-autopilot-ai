"""Gösterilen SQL, pandas hesabıyla aynı sonucu vermeli.

SQL burada yalnızca test için SQLite'ta çalıştırılır; uygulama bu metinleri hiçbir zaman çalıştırmaz.
Tarih (DATE_TRUNC, TIMESTAMP) ve medyan (PERCENTILE_CONT) PostgreSQL'e özgü olduğundan kapsam dışıdır.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from core.chat import DataChatEngine, QueryFilter, StructuredDataQuery


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({
        "şehir": rng.choice(["Ankara", "İzmir", "Bursa"], 200),
        "durum": rng.choice(["teslim", "iptal", None], 200, p=[0.6, 0.2, 0.2]),
        "tutar": np.round(rng.uniform(10, 100, 200), 2),
        "adet": rng.integers(1, 6, 200),
    })
    frame.loc[frame.index[::9], "tutar"] = np.nan
    return frame


def _sql_rows(frame: pd.DataFrame, sql: str) -> list[tuple]:
    connection = sqlite3.connect(":memory:")
    frame.to_sql("veri", connection, index=False)
    rows = connection.execute(sql).fetchall()
    connection.close()
    return rows


QUERIES = [
    StructuredDataQuery(operation="groupby_agg", group_col="şehir", target_col="tutar", agg_func="mean",
                        filters=[QueryFilter(column="durum", operator="!=", value="iptal")], explanation="x"),
    StructuredDataQuery(operation="groupby_agg", group_col="şehir", target_col="tutar", agg_func="sum",
                        filters=[QueryFilter(column="adet", operator=">=", value="3")], explanation="x"),
    StructuredDataQuery(operation="groupby_agg", target_col="tutar", agg_func="mean",
                        filters=[QueryFilter(column="şehir", operator="in", values=["Ankara", "Bursa"])], explanation="x"),
    StructuredDataQuery(operation="value_counts", group_col="durum",
                        filters=[QueryFilter(column="tutar", operator="<", value="50")], explanation="x"),
    StructuredDataQuery(operation="top_n", group_col="şehir", target_col="tutar", limit=5,
                        filters=[QueryFilter(column="durum", operator="==", value="teslim")], explanation="x"),
]


@pytest.mark.parametrize("query", QUERIES, ids=lambda query: f"{query.operation}-{query.filters[0].operator}")
def test_sql_matches_pandas(table: pd.DataFrame, query: StructuredDataQuery) -> None:
    engine = DataChatEngine()
    filtered, _ = engine.apply_filters(table, query)
    result, _, _ = engine._execute(filtered, query)
    sql_rows = _sql_rows(table, engine.explain(table, query).sql)
    pandas_rows = [tuple(row) for row in result.itertuples(index=False)]
    def normalize(rows: list[tuple]) -> list[tuple]:
        cleaned = [
            tuple(None if isinstance(value, float) and np.isnan(value) else
                  (round(value, 6) if isinstance(value, float) else value) for value in row)
            for row in rows
        ]
        return sorted(cleaned, key=lambda row: tuple((value is None, str(value)) for value in row))

    if query.group_col is None:
        pandas_rows = [(row[-1],) for row in pandas_rows]  # pandas tek değerin yanına okunur bir etiket koyar
    if query.operation == "value_counts":
        pandas_rows = [(None if pd.isna(key) else key, count) for key, count in pandas_rows]
    assert normalize(sql_rows) == normalize(pandas_rows)


def test_not_equal_excludes_missing_like_sql(table: pd.DataFrame) -> None:
    query = QUERIES[0]
    filtered, _ = DataChatEngine().apply_filters(table, query)
    assert filtered["durum"].notna().all() and (filtered["durum"] != "iptal").all()
    assert "df['durum'].notna()" in DataChatEngine().explain(table, query).pandas
