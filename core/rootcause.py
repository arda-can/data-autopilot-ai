"""“Neden değişti?”: bir metriğin iki dönem arasındaki değişimini boyut değerlerinin katkılarına ayırır.

Ayrıştırma tamdır, yani katkıların toplamı değişime birebir eşittir:

- Toplanabilir metrikler (kayıt sayısı, toplam): grup katkısı = güncel değer − referans değer.
- Ortalama ve oran: simetrik shift-share. Grup payı w, grup ortalaması m, genel ortalamalar M ise
  katkı = Δw·(m̄ − M̄) [karma etkisi] + w̄·Δm [grup içi değişim]; çubuklu değerler iki dönemin ortalaması.
  Bir dönemde hiç kaydı olmayan grubun grup içi değişimi 0 sayılır; etkisi tamamen karma etkisidir.

Boyutlar "beklenenden sapma" ile sıralanır: değişim tüm gruplara eşit yayılsaydı (toplanabilir metrikte
grubun referans payı kadar, ortalamada grubun ağırlığı kadar) her grubun beklenen katkısı ne olurdu?
Sapma = Σ|katkı − beklenen| / (2·|Δ|). Büyük bir grup yalnızca büyük olduğu için öne çıkmaz.

Sınırlamalar açıkça yazılır: bu nedensellik değil, muhasebe ayrıştırmasıdır; boyutlar birbirinden bağımsız
incelenir (etkileşim yok) ve küçük grupların katkısı gürültü olabilir.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from core.columns import is_identifier_name, is_text_series
from core.metadata import MetadataGenerator

MetricKind = Literal["sayım", "toplam", "ortalama", "oran"]
MAX_GROUPS = 50
MIN_GROUP_ROWS = 30
MAX_FILL_CHANGE = 0.3  # doluluğu dönemler arasında bundan fazla değişen boyut atlanır
EMPTY_LABEL = "(boş)"
METHOD_NOTE = (
    "Katkılar toplamı değişime birebir eşittir. Toplam ve sayımda katkı grubun farkıdır; ortalama ve oranda "
    "karma etkisi (grubun payının değişmesi) ile grup içi değişim ayrı gösterilir. Boyutlar, değişimin tüm "
    "gruplara eşit yayıldığı varsayımından sapmaya göre sıralanır. Bu bir muhasebe ayrıştırmasıdır, nedensellik "
    "kanıtı değildir; boyutlar ayrı ayrı incelenir ve 30 kayıttan küçük grupların katkısı gürültü olabilir."
)


class MetricSpec(BaseModel):
    kind: MetricKind
    column: str | None = None
    value: str | None = None  # oran: column == value

    def label(self) -> str:
        if self.kind == "sayım":
            return "Kayıt sayısı"
        if self.kind == "toplam":
            return f"{self.column} toplamı"
        if self.kind == "ortalama":
            return f"{self.column} ortalaması"
        return f"“{self.column} = {self.value}” oranı"


class GroupContribution(BaseModel):
    group: str
    reference_rows: int
    current_rows: int
    reference_value: float | None
    current_value: float | None
    contribution: float
    share_of_change: float
    expected: float
    mix_effect: float | None = None
    rate_effect: float | None = None
    small: bool = False
    only_in: Literal["referans", "güncel"] | None = None


class DimensionResult(BaseModel):
    dimension: str
    score: float
    groups: list[GroupContribution]  # |katkı| büyükten küçüğe
    summary: str


class RootCauseReport(BaseModel):
    metric: MetricSpec
    metric_label: str
    reference_label: str
    current_label: str
    reference_value: float
    current_value: float
    change: float
    relative_change: float | None
    dimensions: list[DimensionResult]
    skipped: list[str]
    headline: str
    method_note: str = METHOD_NOTE


class RootCauseResult:
    def __init__(self, report: RootCauseReport, figure: go.Figure | None) -> None:
        self.report, self.figure = report, figure


def candidate_dimensions(frame: pd.DataFrame, exclude: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """(uygun boyutlar, atlananlar ve nedenleri). Kimlik, kişisel veri ve çok değerli sütunlar boyut olmaz."""
    dimensions, skipped = [], []
    for column in frame.columns:
        name = str(column)
        if name in exclude or not is_text_series(frame[column]):
            continue
        if is_identifier_name(column) or MetadataGenerator.is_sensitive_name(column):
            skipped.append(f"“{name}” kimlik ya da kişisel veri olduğu için boyut olarak kullanılmadı.")
        elif frame[column].nunique() > MAX_GROUPS:
            skipped.append(f"“{name}” {MAX_GROUPS}'den fazla farklı değer içerdiği için boyut olarak kullanılmadı.")
        elif frame[column].nunique() >= 2:
            dimensions.append(name)
    return dimensions, skipped


def _validate(frame: pd.DataFrame, metric: MetricSpec) -> None:
    if metric.kind == "sayım":
        return
    if metric.column not in frame.columns:
        raise ValueError(f"“{metric.column}” sütunu veride yok.")
    if metric.kind in ("toplam", "ortalama") and not pd.api.types.is_numeric_dtype(frame[metric.column]):
        raise ValueError(f"“{metric.column}” sayısal değil; {metric.kind} hesaplanamaz.")
    if metric.kind == "oran" and metric.value is None:
        raise ValueError("Oran metriği için bir değer seçilmeli.")


def _values(frame: pd.DataFrame, metric: MetricSpec) -> pd.Series:
    """Satır başına metrik girdisi: sayımda 1, oranda 0/1, diğerlerinde sütun değeri (boşlar ortalamaya girmez)."""
    if metric.kind == "sayım":
        return pd.Series(1.0, index=frame.index)
    if metric.kind == "oran":
        present = frame[metric.column].notna()
        return (frame[metric.column].astype(str) == metric.value).astype(float).where(present)
    return pd.to_numeric(frame[metric.column], errors="coerce").astype(float)


def _aggregate(values: pd.Series, metric: MetricSpec) -> float:
    if metric.kind in ("sayım", "toplam"):
        return float(values.sum())
    present = values.dropna()
    return float(present.mean()) if len(present) else float("nan")


def _groups(frame: pd.DataFrame, dimension: str) -> pd.Series:
    return frame[dimension].astype(str).where(frame[dimension].notna(), EMPTY_LABEL)


def _decompose(
    reference: pd.DataFrame, current: pd.DataFrame, metric: MetricSpec, dimension: str,
    reference_values: pd.Series, current_values: pd.Series, change: float,
) -> DimensionResult:
    additive = metric.kind in ("sayım", "toplam")
    ref_groups, cur_groups = _groups(reference, dimension), _groups(current, dimension)
    labels = sorted(set(ref_groups) | set(cur_groups))
    ref_total = _aggregate(reference_values, metric)
    stats = []
    for label in labels:
        ref_part, cur_part = reference_values[ref_groups == label], current_values[cur_groups == label]
        stats.append((label, ref_part, cur_part))

    contributions: list[GroupContribution] = []
    if additive:
        for label, ref_part, cur_part in stats:
            ref_value, cur_value = float(ref_part.sum()), float(cur_part.sum())
            expected = change * (ref_value / ref_total) if ref_total else change / len(stats)
            contributions.append(GroupContribution(
                group=label, reference_rows=len(ref_part), current_rows=len(cur_part),
                reference_value=ref_value, current_value=cur_value, contribution=cur_value - ref_value,
                share_of_change=0.0, expected=expected,
            ))
    else:
        ref_n, cur_n = reference_values.notna().sum(), current_values.notna().sum()
        ref_mean, cur_mean = _aggregate(reference_values, metric), _aggregate(current_values, metric)
        overall = (ref_mean + cur_mean) / 2
        for label, ref_part, cur_part in stats:
            ref_present, cur_present = ref_part.dropna(), cur_part.dropna()
            w0 = len(ref_present) / ref_n if ref_n else 0.0
            w1 = len(cur_present) / cur_n if cur_n else 0.0
            m0 = float(ref_present.mean()) if len(ref_present) else None
            m1 = float(cur_present.mean()) if len(cur_present) else None
            if m0 is None and m1 is None:
                continue
            m0_used, m1_used = (m1 if m0 is None else m0), (m0 if m1 is None else m1)
            mix = (w1 - w0) * ((m0_used + m1_used) / 2 - overall)
            rate = (w0 + w1) / 2 * (m1_used - m0_used)
            contributions.append(GroupContribution(
                group=label, reference_rows=len(ref_part), current_rows=len(cur_part),
                reference_value=m0, current_value=m1, contribution=mix + rate, share_of_change=0.0,
                expected=(w0 + w1) / 2 * change, mix_effect=mix, rate_effect=rate,
            ))
    for item in contributions:
        item.share_of_change = item.contribution / change if change else 0.0
        item.only_in = "güncel" if item.reference_rows == 0 else "referans" if item.current_rows == 0 else None
        item.small = (max if item.only_in else min)(item.reference_rows, item.current_rows) < MIN_GROUP_ROWS
    contributions.sort(key=lambda item: abs(item.contribution), reverse=True)
    score = sum(abs(item.contribution - item.expected) for item in contributions) / (2 * abs(change)) if change else 0.0
    return DimensionResult(dimension=dimension, score=score, groups=contributions,
                           summary=_dimension_summary(dimension, contributions, change, additive))


def _pct(value: float) -> str:
    return f"%{value * 100:.1f}".replace(".", ",")


def _dimension_summary(dimension: str, groups: list[GroupContribution], change: float, additive: bool) -> str:
    if not groups or not change:
        return f"“{dimension}” boyutunda belirgin bir katkı yok."
    top = groups[0]
    expected_share = top.expected / change
    text = (f"“{dimension}” boyutunda en büyük katkı “{top.group}”: değişimin {_pct(top.share_of_change)}'i "
            f"(değişim eşit yayılsaydı beklenen {_pct(expected_share)}).")
    if not additive and top.mix_effect is not None and top.rate_effect is not None:
        driver = "grup içi değişimden" if abs(top.rate_effect) >= abs(top.mix_effect) else "grubun payının değişmesinden"
        text += f" Katkının çoğu {driver} geliyor."
    if top.only_in:
        text += f" Bu grup yalnızca {top.only_in} dönemde var."
    if top.small:
        text += " Bu grup küçük (30 kayıttan az); sonuç gürültü olabilir."
    return text


def _number(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", ".")
    return f"{value:,.3f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")


def _waterfall(report: RootCauseReport) -> go.Figure | None:
    if not report.dimensions or not report.change:
        return None
    best = report.dimensions[0]
    shown = best.groups[:8]
    rest = sum(item.contribution for item in best.groups[8:])
    labels = [report.reference_label, *[item.group for item in shown]] + (["Diğer"] if len(best.groups) > 8 else [])
    values = [report.reference_value, *[item.contribution for item in shown]] + ([rest] if len(best.groups) > 8 else [])
    figure = go.Figure(go.Waterfall(
        x=[*labels, report.current_label], y=[*values, report.current_value],
        measure=["absolute", *["relative"] * (len(values) - 1), "total"],
        increasing={"marker": {"color": "#059669"}}, decreasing={"marker": {"color": "#dc2626"}},
        totals={"marker": {"color": "#1d5fd1"}}, connector={"line": {"color": "#94a3b8"}},
    ))
    low, high = sorted((report.reference_value, report.current_value))
    padding = max(abs(report.change) * 1.5, abs(high) * 0.02)
    figure.update_layout(
        title=f"{report.metric_label}: “{best.dimension}” boyutuna göre değişim", separators=",.",
        yaxis={"range": [low - padding, high + padding]} if report.metric.kind in ("ortalama", "oran") else {},
        showlegend=False,
    )
    return figure


class RootCauseAnalyzer:
    def analyze(
        self, reference: pd.DataFrame, current: pd.DataFrame, reference_label: str, current_label: str,
        metric: MetricSpec, dimensions: list[str] | None = None,
    ) -> RootCauseResult:
        _validate(reference, metric)
        if reference.empty or current.empty:
            raise ValueError("Dönemlerden birinde hiç kayıt yok.")
        exclude = tuple(column for column in (metric.column,) if column)
        available, skipped = candidate_dimensions(pd.concat([reference, current]), exclude)
        if dimensions is not None:
            unknown = [name for name in dimensions if name not in available]
            if unknown:
                raise ValueError(f"Boyut olarak kullanılamayan sütun: {', '.join(unknown)}")
            available = list(dimensions)
        for dimension in list(available):
            fill_change = abs(reference[dimension].notna().mean() - current[dimension].notna().mean())
            if fill_change > MAX_FILL_CHANGE:
                available.remove(dimension)
                skipped.append(
                    f"“{dimension}” dönemler arasında doluluğu {fill_change * 100:.0f} puan değiştiği için (ör. yeni "
                    "toplanmaya başlanan alan) boyut olarak kullanılmadı; şema değişikliği katkı gibi görünürdü."
                )
        reference_values, current_values = _values(reference, metric), _values(current, metric)
        reference_value, current_value = _aggregate(reference_values, metric), _aggregate(current_values, metric)
        if np.isnan(reference_value) or np.isnan(current_value):
            raise ValueError("Dönemlerden birinde metrik hesaplanamadı (tüm değerler boş).")
        change = current_value - reference_value
        results = [
            _decompose(reference, current, metric, dimension, reference_values, current_values, change)
            for dimension in available
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        relative = change / abs(reference_value) if reference_value else None
        direction = "arttı" if change > 0 else "azaldı" if change < 0 else "değişmedi"
        headline = (f"{metric.label()} “{reference_label}” döneminde {_number(reference_value)}, “{current_label}” "
                    f"döneminde {_number(current_value)}: {direction}"
                    + (f" ({'+' if change > 0 else ''}{_pct(relative)})." if relative is not None and change else "."))
        if results and change:
            headline += " " + results[0].summary
        elif not results:
            headline += " Ayrıştırma için uygun boyut (az değerli kategorik sütun) yok."
        report = RootCauseReport(
            metric=metric, metric_label=metric.label(), reference_label=reference_label, current_label=current_label,
            reference_value=reference_value, current_value=current_value, change=change, relative_change=relative,
            dimensions=results, skipped=skipped, headline=headline,
        )
        return RootCauseResult(report, _waterfall(report))
