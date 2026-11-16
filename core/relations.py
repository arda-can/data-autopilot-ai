"""Çoklu tablo keşfi: ortak anahtarları bulur, güvenli birleştirme planı önerir ve onaylanan planı uygular.

Satır patlamasını önlemek için yalnızca iki birleştirme türü vardır:
- **Eşleştir:** karşı tablonun anahtarı benzersizse sol birleştirme (N-1 / 1-1). Satır sayısı değişmez.
- **Topla:** karşı tablo anahtar başına birden çok satır içeriyorsa (1-N) önce anahtar başına kayıt sayısı,
  sayısal ortalama/toplam ve en sık kategori hesaplanır, sonra sol birleştirme yapılır.
Çoktan-çoğa (N-N) ilişkiler listelenir ama önerilmez; anlamsız eşleşmeler (ör. müşteri ve satıcı posta kodu)
otomatik birleştirilmez.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from core.columns import is_identifier_name, is_text_series, name_tokens
from core.formatting import decimal_tr, integer_tr
from core.health import is_nonnegative_name
from core.ingestion import read_local_csv

RelationKind = Literal["1-1", "N-1", "1-N", "N-N"]
JoinMode = Literal["eşleştir", "topla"]

MIN_CONTAINMENT = 0.50
MIN_SHARED_VALUES = 10
MIN_DISTINCT = 10
MAX_DEPTH = 2
_MAX_MODE_CATEGORIES = 20
_MODE_ROW_LIMIT = 300_000


class TableInfo(BaseModel):
    name: str
    rows: int
    columns: int
    unique_keys: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    left_table: str
    left_column: str
    right_table: str
    right_column: str
    kind: RelationKind
    containment: float  # sol tablonun farklı değerlerinin sağda bulunma oranı
    shared_values: int
    suggested: bool
    note: str


class JoinStep(BaseModel):
    table: str
    left_table: str          # anahtarın geldiği (zaten birleştirilmiş) tablo
    left_column: str
    right_column: str
    mode: JoinMode
    depth: int


class JoinPlan(BaseModel):
    base: str
    steps: list[JoinStep] = Field(default_factory=list)


class JoinLog(BaseModel):
    table: str
    mode: JoinMode
    key: str
    rows_before: int
    rows_after: int
    match_rate: float
    added_columns: list[str]
    note: str


@dataclass
class JoinResult:
    frame: pd.DataFrame
    plan: JoinPlan
    logs: list[JoinLog] = field(default_factory=list)


# --- Yükleme ve yardımcılar ------------------------------------------------------------------


def short_name(path_or_name: str | Path) -> str:
    """`olist_order_items_dataset.csv` → `order_items`."""
    stem = Path(str(path_or_name)).stem
    for prefix in ("olist_",):
        stem = stem.removeprefix(prefix)
    for suffix in ("_dataset",):
        stem = stem.removesuffix(suffix)
    return stem


def load_tables(paths: list[Path]) -> dict[str, pd.DataFrame]:
    return {short_name(path): read_local_csv(path) for path in paths}


def _key_like(series: pd.Series) -> bool:
    """Anahtar olabilecek tip: metin veya tam sayı; tarih, ondalıklı sayı ve mantıksal değer değil."""
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series) or is_text_series(series):
        return True
    if pd.api.types.is_float_dtype(series):
        values = series.dropna()
        return bool(len(values)) and bool((values % 1 == 0).all())
    return False


def _distinct(series: pd.Series) -> set[str]:
    values = series.dropna()
    if pd.api.types.is_float_dtype(values):
        values = values.astype("int64")
    return set(values.astype(str).unique())


def _is_unique(series: pd.Series) -> bool:
    values = series.dropna()
    return len(values) > 0 and values.is_unique


def _name_affinity(left: str, right: str) -> bool:
    """Aynı ad ya da en az iki kelimelik ortak son ek (`customer_zip_code_prefix` ~ `seller_zip_code_prefix`)."""
    a, b = name_tokens(left), name_tokens(right)
    if not a or not b:
        return False
    if a == b:
        return True
    shared = 0
    while shared < min(len(a), len(b)) and a[-1 - shared] == b[-1 - shared]:
        shared += 1
    return shared >= 2 or (shared >= 1 and min(len(a), len(b)) == 1)


# --- Keşif -----------------------------------------------------------------------------------


class RelationFinder:
    def describe(self, tables: Mapping[str, pd.DataFrame]) -> list[TableInfo]:
        return [
            TableInfo(
                name=name, rows=len(frame), columns=frame.shape[1],
                unique_keys=[str(column) for column in frame.columns
                             if _key_like(frame[column]) and frame[column].nunique() >= MIN_DISTINCT
                             and _is_unique(frame[column])],
            )
            for name, frame in tables.items()
        ]

    def discover(self, tables: Mapping[str, pd.DataFrame]) -> list[Relation]:
        names = list(tables)
        distinct_cache: dict[tuple[str, str], set[str]] = {}

        def distinct(table: str, column: str) -> set[str]:
            key = (table, column)
            if key not in distinct_cache:
                distinct_cache[key] = _distinct(tables[table][column])
            return distinct_cache[key]

        relations: list[Relation] = []
        for i, left_table in enumerate(names):
            for right_table in names[i + 1:]:
                left_frame, right_frame = tables[left_table], tables[right_table]
                for left_column in map(str, left_frame.columns):
                    if not _key_like(left_frame[left_column]):
                        continue
                    for right_column in map(str, right_frame.columns):
                        if not _key_like(right_frame[right_column]) or not _name_affinity(left_column, right_column):
                            continue
                        left_values = distinct(left_table, left_column)
                        right_values = distinct(right_table, right_column)
                        if min(len(left_values), len(right_values)) < MIN_DISTINCT:
                            continue
                        shared = len(left_values & right_values)
                        containment = max(shared / len(left_values), shared / len(right_values))
                        if shared < MIN_SHARED_VALUES or containment < MIN_CONTAINMENT:
                            continue
                        relations.append(self._relation(
                            left_table, left_column, _is_unique(left_frame[left_column]),
                            right_table, right_column, _is_unique(right_frame[right_column]),
                            shared, len(left_values), len(right_values),
                        ))
        return relations

    @staticmethod
    def _relation(
        left_table: str, left_column: str, left_unique: bool,
        right_table: str, right_column: str, right_unique: bool,
        shared: int, left_distinct: int, right_distinct: int,
    ) -> Relation:
        # "Çok" tarafı solda tutulur: N-1 okunması kolaydır (sipariş → müşteri).
        if left_unique and not right_unique:
            left_table, right_table = right_table, left_table
            left_column, right_column = right_column, left_column
            left_unique, right_unique = right_unique, left_unique
            left_distinct, right_distinct = right_distinct, left_distinct
        kind: RelationKind = (
            "1-1" if left_unique and right_unique else "N-1" if right_unique else "N-N"
        )
        containment = shared / left_distinct
        if kind == "N-N":
            note = "Çoktan-çoğa: doğrudan birleştirme satırları çoğaltır; otomatik önerilmez."
        else:
            note = (
                f"“{left_table}” tablosundaki farklı değerlerin %{decimal_tr(containment * 100)} kadarı "
                f"“{right_table}” içinde bulunuyor."
            )
        return Relation(
            left_table=left_table, left_column=left_column, right_table=right_table, right_column=right_column,
            kind=kind, containment=round(containment, 4), shared_values=shared,
            suggested=kind != "N-N", note=note,
        )


# --- Plan ------------------------------------------------------------------------------------


def suggest_plan(base: str, relations: list[Relation], tables: Mapping[str, Any]) -> JoinPlan:
    """Taban tablodan en fazla iki adım uzaklıktaki tablolar için güvenli plan.

    Eşleştirilen (benzersiz anahtarlı) tablolardan bir sonraki adıma geçilebilir; toplanan tablolar
    satır düzeyi bilgi taşımadığı için zincir orada durur.
    """
    if base not in tables:
        raise ValueError(f"“{base}” adlı tablo bulunamadı.")
    joined = {base}
    frontier = [base]
    steps: list[JoinStep] = []
    for depth in range(1, MAX_DEPTH + 1):
        next_frontier: list[str] = []
        for source in frontier:
            for relation in relations:
                if not relation.suggested:
                    continue
                if relation.left_table == source and relation.right_table not in joined:
                    target, source_column, target_column = relation.right_table, relation.left_column, relation.right_column
                    mode: JoinMode = "eşleştir"  # sağ taraf benzersiz (N-1 veya 1-1)
                elif relation.right_table == source and relation.left_table not in joined:
                    target, source_column, target_column = relation.left_table, relation.right_column, relation.left_column
                    mode = "eşleştir" if relation.kind == "1-1" else "topla"
                else:
                    continue
                if any(step.table == target for step in steps):
                    continue
                steps.append(JoinStep(table=target, left_table=source, left_column=source_column,
                                      right_column=target_column, mode=mode, depth=depth))
                joined.add(target)
                if mode == "eşleştir":
                    next_frontier.append(target)
        frontier = next_frontier
    return JoinPlan(base=base, steps=steps)


def prune_plan(plan: JoinPlan, approved_tables: set[str]) -> tuple[JoinPlan, list[str]]:
    """Onaylanan adımları tutar; onaylanmamış (ya da toplanmış) bir tabloya dayanan adımları düşürür.

    Dönüş: (geçerli plan, düşürülen adımların gerekçeleri).
    """
    reachable = {plan.base}
    kept: list[JoinStep] = []
    dropped: list[str] = []
    for step in plan.steps:
        if step.table not in approved_tables:
            continue
        if step.left_table not in reachable:
            dropped.append(f"“{step.table}” atlandı: dayandığı “{step.left_table}” birleştirilmedi.")
            continue
        kept.append(step)
        if step.mode == "eşleştir":
            reachable.add(step.table)
    return JoinPlan(base=plan.base, steps=kept), dropped


def default_base(relations: list[Relation], names: list[str]) -> str:
    """En çok önerilen ilişkiye sahip tablo (Olist'te `orders`); eşitlikte ilk sıradaki."""
    degree = {name: 0 for name in names}
    for relation in relations:
        if relation.suggested:
            degree[relation.left_table] = degree.get(relation.left_table, 0) + 1
            degree[relation.right_table] = degree.get(relation.right_table, 0) + 1
    return max(names, key=lambda name: (degree.get(name, 0), -names.index(name)))


# --- Uygulama --------------------------------------------------------------------------------


def _aggregate(frame: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Anahtar başına: kayıt sayısı, sayısal ortalama (+ negatif olamayan ölçülerde toplam), en sık kategori."""
    grouped = frame.dropna(subset=[key]).groupby(key)
    parts = [grouped.size().rename(f"{prefix}_kayıt_sayısı")]
    for column in frame.columns:
        if column == key:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if is_identifier_name(column):
                continue  # kimlik/kod: ortalaması anlamsız. Tam sayı olmak tek başına kimlik göstergesi değildir.
            parts.append(grouped[column].mean().rename(f"{prefix}_{column}_ort"))
            if is_nonnegative_name(column):
                parts.append(grouped[column].sum(min_count=1).rename(f"{prefix}_{column}_toplam"))
        elif (
            is_text_series(series) and len(frame) <= _MODE_ROW_LIMIT
            and 1 < series.nunique() <= _MAX_MODE_CATEGORIES
        ):
            # Vektörel "en sık değer": (anahtar, değer) sayımlarından anahtar başına en büyüğü; eşitlikte
            # alfabetik ilk değer seçilir, sonuç deterministiktir.
            counts = (
                frame[[key, column]].dropna().groupby([key, column]).size().rename("adet").reset_index()
                .sort_values([key, "adet", column], ascending=[True, False, True])
                .drop_duplicates(key).set_index(key)[column]
            )
            parts.append(counts.rename(f"{prefix}_{column}_en_sık"))
    return pd.concat(parts, axis=1).reset_index()


def _comparable(series: pd.Series) -> pd.Series:
    """Anahtarları metin olarak karşılaştırır (12 ile '12' eşleşsin; ondalık .0 kalıntısı olmasın)."""
    values = series
    if pd.api.types.is_float_dtype(values):
        values = values.astype("Int64")
    return values.astype("string")


def apply_plan(tables: Mapping[str, pd.DataFrame], plan: JoinPlan) -> JoinResult:
    """Onaylanan planı sırayla uygular; girdileri değiştirmez, her adımı kayda geçirir."""
    if plan.base not in tables:
        raise ValueError(f"“{plan.base}” adlı tablo bulunamadı.")
    result = tables[plan.base].copy()
    # Her tablonun özgün sütununun, birleşik tablodaki adı (çakışmada sonek alır).
    names: dict[str, dict[str, str]] = {plan.base: {str(column): str(column) for column in result.columns}}
    logs: list[JoinLog] = []
    for step in plan.steps:
        if step.left_table not in names:
            raise ValueError(f"“{step.table}” adımı, henüz birleştirilmemiş “{step.left_table}” tablosuna dayanıyor.")
        left_key = names[step.left_table].get(step.left_column)
        if left_key is None:
            raise ValueError(
                f"“{step.left_table}.{step.left_column}” birleşik tabloda yok; “{step.table}” birleştirilemez."
            )
        right = tables[step.table]
        rows_before = len(result)
        if step.mode == "eşleştir":
            if not _is_unique(right[step.right_column]):
                raise ValueError(
                    f"“{step.table}.{step.right_column}” benzersiz değil; eşleştirme satırları çoğaltırdı."
                )
            addition = right.copy()
        else:
            addition = _aggregate(right, step.right_column, step.table)
        renamed: dict[str, str] = {}
        for column in map(str, addition.columns):
            if column == step.right_column:
                continue
            target = column if column not in result.columns else f"{column}_{step.table}"
            renamed[column] = target
        addition = addition.rename(columns=renamed)
        join_key = "__birlesim_anahtari__"
        left_values = _comparable(result[left_key])
        addition[join_key] = _comparable(addition[step.right_column])
        # Boş anahtar hiçbir şeyle eşleşmez: pandas birleştirmede NA'yı NA ile eşleştirirdi ve boş kimlikli
        # siparişlere rastgele bir müşterinin alanları yazılırdı (iki boş sağ anahtar ise MergeError verirdi).
        addition = addition[addition[join_key].notna()]
        matched = left_values.isin(set(addition[join_key])).fillna(False).astype(bool)
        merged = result.assign(**{join_key: left_values}).merge(
            addition.drop(columns=[step.right_column]), on=join_key, how="left", validate="many_to_one",
        ).drop(columns=[join_key])
        merged.index = result.index
        result = merged
        names[step.table] = {**renamed, step.right_column: left_key}
        match_rate = float(matched.mean()) if len(matched) else 0.0
        logs.append(JoinLog(
            table=step.table, mode=step.mode, key=f"{left_key} = {step.table}.{step.right_column}",
            rows_before=rows_before, rows_after=len(result), match_rate=round(match_rate, 4),
            added_columns=list(renamed.values()),
            note=(
                f"{step.mode.capitalize()}: {integer_tr(len(renamed))} sütun eklendi; taban satırlarının "
                f"%{decimal_tr(match_rate * 100)} kadarı eşleşti."
                + (" Eşleşmeyen satırlarda yeni sütunlar boş kaldı." if match_rate < 1 else "")
            ),
        ))
    return JoinResult(frame=result, plan=plan, logs=logs)


# --- Görsel ----------------------------------------------------------------------------------


def relation_figure(tables: list[TableInfo], relations: list[Relation]) -> go.Figure:
    """Tabloları çember üzerine dizer; önerilen ilişkiler düz, çoktan-çoğa ilişkiler kesik çizgi."""
    count = max(len(tables), 1)
    positions = {
        info.name: (math.cos(2 * math.pi * index / count), math.sin(2 * math.pi * index / count))
        for index, info in enumerate(tables)
    }
    figure = go.Figure()
    for relation in relations:
        (x0, y0), (x1, y1) = positions[relation.left_table], positions[relation.right_table]
        figure.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines", showlegend=False, hoverinfo="skip",
            line={"color": "#1d5fd1" if relation.suggested else "#94a3b8", "width": 2,
                  "dash": "solid" if relation.suggested else "dash"},
        ))
        figure.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2], mode="text", showlegend=False,
            text=[f"{relation.kind}<br>{relation.left_column}"], textfont={"size": 10, "color": "#5b6778"},
            hovertext=[f"{relation.left_table}.{relation.left_column} → {relation.right_table}.{relation.right_column}"],
            hoverinfo="text",
        ))
    figure.add_trace(go.Scatter(
        x=[positions[info.name][0] for info in tables], y=[positions[info.name][1] for info in tables],
        mode="markers+text", text=[f"<b>{info.name}</b><br>{integer_tr(info.rows)} satır" for info in tables],
        textposition="top center", marker={"size": 22, "color": "#16202e"}, showlegend=False, hoverinfo="text",
    ))
    figure.update_layout(
        title="Tablolar arası ilişkiler (düz: önerilen, kesik: çoktan-çoğa)",
        xaxis={"visible": False, "range": [-1.5, 1.5]}, yaxis={"visible": False, "range": [-1.4, 1.5]},
        height=560, plot_bgcolor="white",
    )
    return figure
