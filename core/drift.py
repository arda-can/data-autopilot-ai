"""Dönem karşılaştırması (veri kayması): iki dönem ya da iki dosya arasında ne değişti?

Her ortak sütun için:
- Sayısal sütunlarda referansın ondalık dilimlerine göre PSI ve Kolmogorov-Smirnov testi,
- Kategorik sütunlarda kategori paylarına göre PSI ve ki-kare testi,
- Eksik değer oranındaki değişim.
Durum (yeşil/sarı/kırmızı) etki büyüklüğüne göre verilir; büyük örneklemlerde küçük farklar bile
istatistiksel olarak anlamlı çıktığından p-değeri yalnızca bilgi amaçlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel
from scipy import stats

from core.columns import is_identifier_column, is_text_series
from core.executor import ActionExecutor
from core.formatting import decimal_tr, integer_tr, localize_figure

Status = Literal["yeşil", "sarı", "kırmızı"]

PSI_WARN = 0.10
PSI_ALERT = 0.25
MISSING_WARN = 0.05
MISSING_ALERT = 0.10
MIN_ROWS = 30
_BINS = 10
_MAX_CATEGORIES = 20
_MAX_DISTINCT = 200
_LOW_CARDINALITY = 12
_TEST_SAMPLE = 5_000
_EPS = 1e-4
_SEVERITY = {"kırmızı": 2, "sarı": 1, "yeşil": 0}
STATUS_COLORS = {"yeşil": "#059669", "sarı": "#d97706", "kırmızı": "#dc2626"}
METHOD_NOTE = (
    "Durum etki büyüklüğüne göre verilir: PSI 0,10'un altında yeşil, 0,10–0,25 arası sarı, 0,25'in üstünde "
    "kırmızı; eksik oranı 5 puan değişirse sarı, 10 puan değişirse kırmızı. Büyük örneklemlerde küçük farklar "
    "bile istatistiksel olarak anlamlı çıktığından p-değeri yalnızca bilgi amaçlıdır."
)


class ColumnDrift(BaseModel):
    column: str
    kind: Literal["sayısal", "kategorik", "metin"]
    status: Status
    psi: float
    test: str
    p_value: float | None = None
    missing_reference: float
    missing_current: float
    reference_summary: str
    current_summary: str
    explanation: str


class SchemaChange(BaseModel):
    column: str
    change: Literal["eklendi", "kaldırıldı", "tip değişti"]
    detail: str


class DriftReport(BaseModel):
    """JSON'a çevrilebilir karşılaştırma özeti; ham satır içermez."""

    reference_label: str
    current_label: str
    reference_rows: int
    current_rows: int
    schema_changes: list[SchemaChange]
    columns: list[ColumnDrift]
    skipped: list[str]
    counts: dict[str, int]
    summary: str
    method_note: str = METHOD_NOTE


@dataclass
class DriftResult:
    report: DriftReport
    figure: Any | None
    detail_figures: dict[str, Any] = field(default_factory=dict)


# --- Yardımcılar -----------------------------------------------------------------------------


def _kind(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "tarih"
    if pd.api.types.is_bool_dtype(series):
        return "kategorik"
    if pd.api.types.is_numeric_dtype(series):
        return "sayısal"
    return "metin" if is_text_series(series) else "kategorik"


def _psi(reference: np.ndarray, current: np.ndarray) -> float:
    p = np.clip(reference, _EPS, None)
    q = np.clip(current, _EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _status_for(psi: float, missing_change: float) -> Status:
    if psi > PSI_ALERT or missing_change >= MISSING_ALERT:
        return "kırmızı"
    if psi >= PSI_WARN or missing_change >= MISSING_WARN:
        return "sarı"
    return "yeşil"


def _sample(values: pd.Series, seed: int = 0) -> pd.Series:
    return values.sample(_TEST_SAMPLE, random_state=seed) if len(values) > _TEST_SAMPLE else values


def _num(value: float) -> str:
    return integer_tr(int(value)) if float(value).is_integer() else decimal_tr(float(value), 2)


def prepare_for_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """İkinci dosyayı ilkiyle aynı yapısal hijyenden geçirir (boşluk, para/sayı metni, tarih); istatistik öğrenmez."""
    executor = ActionExecutor()
    prepared = executor.strip_whitespace(frame)
    prepared = executor.clean_currency_and_numeric_strings(prepared)
    return executor.parse_datetime_columns(prepared)


def split_by_date(
    frame: pd.DataFrame, date_column: str, cut: pd.Timestamp | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Tarih sütununa göre ikiye böler: kesimden önceki kayıtlar referans, sonrakiler güncel dönem."""
    dates = _dates(frame, date_column)
    if cut is None:
        ordered = dates.dropna().sort_values()
        cut = ordered.iloc[len(ordered) // 2].normalize()
    cut = pd.Timestamp(cut)
    reference, current = frame[dates < cut], frame[dates >= cut]
    last_reference_day = cut - pd.Timedelta(days=1)
    return (
        reference, current,
        f"{dates.min():%d.%m.%Y} – {last_reference_day:%d.%m.%Y}",
        f"{cut:%d.%m.%Y} – {dates.max():%d.%m.%Y}",
    )


def split_by_years(
    frame: pd.DataFrame, date_column: str, reference_year: int, current_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """İki takvim yılını karşılaştırır ("2017 ile 2018 arasında ne değişti?")."""
    years = _dates(frame, date_column).dt.year
    return frame[years == reference_year], frame[years == current_year], str(reference_year), str(current_year)


def _dates(frame: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column not in frame.columns or not pd.api.types.is_datetime64_any_dtype(frame[date_column]):
        raise ValueError(f"'{date_column}' bir tarih sütunu değil.")
    return frame[date_column]


# --- Motor -----------------------------------------------------------------------------------


class DriftEngine:
    """Deterministik dönem karşılaştırması; LLM kullanmaz, girdiyi değiştirmez."""

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        reference_label: str = "Referans",
        current_label: str = "Güncel",
    ) -> DriftResult:
        for name, frame in ((reference_label, reference), (current_label, current)):
            if not isinstance(frame, pd.DataFrame) or len(frame) < MIN_ROWS:
                raise ValueError(
                    f"Karşılaştırma için her dönemde en az {MIN_ROWS} satır gerekir; "
                    f"“{name}” döneminde {0 if frame is None else len(frame)} satır var."
                )
        schema_changes: list[SchemaChange] = []
        reference_columns = [str(column) for column in reference.columns]
        current_columns = [str(column) for column in current.columns]
        for column in current_columns:
            if column not in reference_columns:
                schema_changes.append(SchemaChange(column=column, change="eklendi", detail="Yalnızca güncel dönemde var."))
        for column in reference_columns:
            if column not in current_columns:
                schema_changes.append(SchemaChange(column=column, change="kaldırıldı", detail="Güncel dönemde yok."))

        drifts: list[ColumnDrift] = []
        skipped: list[str] = []
        for column in (column for column in reference_columns if column in current_columns):
            ref_series, cur_series = reference[column], current[column]
            ref_kind, cur_kind = _kind(ref_series), _kind(cur_series)
            if ref_kind != cur_kind:
                schema_changes.append(SchemaChange(
                    column=column, change="tip değişti", detail=f"{ref_kind} → {cur_kind}",
                ))
                continue
            if ref_kind == "tarih":
                skipped.append(f"{column}: tarih sütunu (dönemler zaten tarihe göre ayrışır).")
                continue
            if is_identifier_column(ref_series, column):
                drift = self._missing_only(
                    column, ref_series, cur_series, ref_kind, "kimlik veya neredeyse her satırı benzersiz"
                )
            else:
                drift = self._column(column, ref_series, cur_series, ref_kind)
            if isinstance(drift, str):
                skipped.append(drift)
            else:
                drifts.append(drift)

        drifts.sort(key=lambda item: (-_SEVERITY[item.status], -item.psi))
        counts = {status: sum(item.status == status for item in drifts) for status in ("kırmızı", "sarı", "yeşil")}
        report = DriftReport(
            reference_label=reference_label,
            current_label=current_label,
            reference_rows=len(reference),
            current_rows=len(current),
            schema_changes=schema_changes,
            columns=drifts,
            skipped=skipped,
            counts=counts,
            summary=self._summary(drifts, schema_changes, counts, reference_label, current_label),
        )
        return DriftResult(
            report=report,
            figure=self._psi_figure(drifts) if drifts else None,
            detail_figures={
                item.column: self._detail_figure(item, reference[item.column], current[item.column],
                                                 reference_label, current_label)
                for item in drifts if item.status != "yeşil" and item.test != "Yalnızca eksik oranı"
            },
        )

    # -- Sütun karşılaştırması --

    def _column(self, column: str, reference: pd.Series, current: pd.Series, kind: str) -> ColumnDrift | str:
        missing_ref, missing_cur = float(reference.isna().mean()), float(current.isna().mean())
        missing_change = abs(missing_cur - missing_ref)
        ref_values, cur_values = reference.dropna(), current.dropna()
        free_text = (
            kind != "sayısal" and pd.concat([ref_values, cur_values]).astype(str).nunique() > _MAX_DISTINCT
        )
        if free_text or len(ref_values) < MIN_ROWS or len(cur_values) < MIN_ROWS:
            reason = (
                f"{_MAX_DISTINCT}'den fazla farklı değer (serbest metin)" if free_text
                else "dolu değer sayısı dağılım karşılaştırması için yetersiz"
            )
            return self._missing_only(column, reference, current, "metin" if free_text else kind, reason)
        numeric = kind == "sayısal" and pd.concat([ref_values, cur_values]).nunique() > _LOW_CARDINALITY
        if numeric:
            psi, p_value, ref_summary, cur_summary, detail = self._numeric(ref_values.astype(float), cur_values.astype(float))
            test, kind_label = "Kolmogorov-Smirnov", "sayısal"
        else:
            psi, p_value, ref_summary, cur_summary, detail = self._categorical(ref_values.astype(str), cur_values.astype(str))
            test, kind_label = "Ki-kare", "kategorik"
        explanation = detail
        if missing_change >= MISSING_WARN:
            explanation += (
                f" Eksik oranı %{decimal_tr(missing_ref * 100)} → %{decimal_tr(missing_cur * 100)}."
            )
        return ColumnDrift(
            column=column, kind=kind_label, status=_status_for(psi, missing_change), psi=round(psi, 4),
            test=test, p_value=None if p_value is None else float(p_value),
            missing_reference=round(missing_ref, 4), missing_current=round(missing_cur, 4),
            reference_summary=ref_summary, current_summary=cur_summary, explanation=explanation,
        )

    @staticmethod
    def _missing_only(
        column: str, reference: pd.Series, current: pd.Series, kind: str, reason: str
    ) -> ColumnDrift | str:
        """Dağılım karşılaştırılamayan sütunda doluluk değişimi tek başına önemli bir bulgu olabilir
        (ör. yorum başlığı alanı yalnızca güncel dönemde kullanılmaya başlanmış, kimlik alanı boşalmış)."""
        missing_ref, missing_cur = float(reference.isna().mean()), float(current.isna().mean())
        missing_change = abs(missing_cur - missing_ref)
        if missing_change < MISSING_WARN:
            return f"{column}: {reason}, karşılaştırılmadı."
        return ColumnDrift(
            column=column, kind=kind if kind in ("sayısal", "kategorik", "metin") else "metin",
            status=_status_for(0.0, missing_change), psi=0.0, test="Yalnızca eksik oranı",
            missing_reference=round(missing_ref, 4), missing_current=round(missing_cur, 4),
            reference_summary=f"eksik %{decimal_tr(missing_ref * 100)}",
            current_summary=f"eksik %{decimal_tr(missing_cur * 100)}",
            explanation=(
                f"Eksik oranı %{decimal_tr(missing_ref * 100)} → %{decimal_tr(missing_cur * 100)} "
                f"({reason}; yalnızca doluluk karşılaştırıldı)."
            ),
        )

    @staticmethod
    def _numeric(reference: pd.Series, current: pd.Series) -> tuple[float, float | None, str, str, str]:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, _BINS + 1)))
        inner = edges[1:-1]
        bins = len(inner) + 1
        # Uçtaki dilimler açık uçlu: referans aralığının dışına düşen güncel değerler de sayılır.
        ref_share = np.bincount(np.searchsorted(inner, reference, side="right"), minlength=bins) / len(reference)
        cur_share = np.bincount(np.searchsorted(inner, current, side="right"), minlength=bins) / len(current)
        psi = _psi(ref_share, cur_share)
        p_value = float(stats.ks_2samp(_sample(reference), _sample(current, 1)).pvalue)
        ref_q1, ref_median, ref_q3 = (float(value) for value in reference.quantile([0.25, 0.5, 0.75]))
        cur_q1, cur_median, cur_q3 = (float(value) for value in current.quantile([0.25, 0.5, 0.75]))
        change = (
            f" (%{decimal_tr((cur_median - ref_median) / abs(ref_median) * 100)})" if ref_median else ""
        )
        # Medyan sabitken de dağılımın şekli değişebilir; çeyrekler bunu görünür kılar.
        return (
            psi, p_value, f"medyan {_num(ref_median)}", f"medyan {_num(cur_median)}",
            f"Medyan {_num(ref_median)} → {_num(cur_median)}{change}; çeyrekler arası "
            f"{_num(ref_q1)}–{_num(ref_q3)} → {_num(cur_q1)}–{_num(cur_q3)}.",
        )

    @staticmethod
    def _categorical(reference: pd.Series, current: pd.Series) -> tuple[float, float | None, str, str, str]:
        ref_counts, cur_counts = reference.value_counts(), current.value_counts()
        totals = ref_counts.add(cur_counts, fill_value=0).sort_values(ascending=False)
        top = list(totals.index[:_MAX_CATEGORIES])

        def bucket(counts: pd.Series) -> pd.Series:
            kept = counts.reindex(top, fill_value=0)
            other = counts.drop(labels=[label for label in top if label in counts.index]).sum()
            return pd.concat([kept, pd.Series({"Diğer": other})]) if len(totals) > len(top) else kept

        ref_bucket, cur_bucket = bucket(ref_counts), bucket(cur_counts)
        ref_share, cur_share = ref_bucket / ref_bucket.sum(), cur_bucket / cur_bucket.sum()
        psi = _psi(ref_share.to_numpy(), cur_share.to_numpy())
        table = np.vstack([ref_bucket.to_numpy(), cur_bucket.to_numpy()])
        table = table[:, table.sum(axis=0) > 0]
        p_value = float(stats.chi2_contingency(table).pvalue) if table.shape[1] > 1 else None
        shift = (cur_share - ref_share).abs().idxmax()
        detail = (
            f"“{shift}” payı %{decimal_tr(ref_share[shift] * 100)} → %{decimal_tr(cur_share[shift] * 100)}."
        )
        new_categories = [label for label in cur_counts.index if label not in ref_counts.index]
        if new_categories:
            detail += f" Güncel dönemde {len(new_categories)} yeni kategori var."
        return (
            psi, p_value,
            f"en sık “{ref_counts.index[0]}” (%{decimal_tr(ref_counts.iloc[0] / len(reference) * 100)})",
            f"en sık “{cur_counts.index[0]}” (%{decimal_tr(cur_counts.iloc[0] / len(current) * 100)})",
            detail,
        )

    # -- Metin ve grafikler --

    @staticmethod
    def _summary(
        drifts: list[ColumnDrift], schema: list[SchemaChange], counts: dict[str, int],
        reference_label: str, current_label: str,
    ) -> str:
        text = (
            f"“{reference_label}” ile “{current_label}” arasında {len(drifts)} sütun karşılaştırıldı: "
            f"{counts['kırmızı']} kırmızı, {counts['sarı']} sarı, {counts['yeşil']} yeşil."
        )
        if drifts and drifts[0].status != "yeşil":
            text += f" En büyük değişim “{drifts[0].column}”: {drifts[0].explanation}"
        elif drifts:
            text += " Dağılımlarda belirgin bir değişim yok."
        if schema:
            text += f" Şema farkı: {len(schema)} sütun eklendi, kaldırıldı veya tipi değişti."
        return text

    @staticmethod
    def _psi_figure(drifts: list[ColumnDrift]) -> go.Figure:
        ordered = drifts[:25][::-1]
        figure = go.Figure(go.Bar(
            x=[item.psi for item in ordered], y=[item.column for item in ordered], orientation="h",
            marker={"color": [STATUS_COLORS[item.status] for item in ordered]}, name="PSI",
        ))
        for threshold in (PSI_WARN, PSI_ALERT):
            figure.add_vline(x=threshold, line_dash="dot", line_color="#5b6778")
        figure.update_layout(title="Sütun bazında dağılım değişimi (PSI)", xaxis_title="PSI", yaxis_title="")
        return localize_figure(figure)

    @staticmethod
    def _detail_figure(
        item: ColumnDrift, reference: pd.Series, current: pd.Series, reference_label: str, current_label: str
    ) -> go.Figure:
        figure = go.Figure()
        if item.kind == "sayısal":
            for values, label, color in ((reference, reference_label, "#1d5fd1"), (current, current_label, "#d97706")):
                figure.add_trace(go.Histogram(
                    x=_sample(values.dropna().astype(float)), histnorm="percent", name=label,
                    opacity=0.6, marker={"color": color}, nbinsx=30,
                ))
            figure.update_layout(barmode="overlay", yaxis_title="Kayıtların yüzdesi")
        else:
            ref_share = reference.dropna().astype(str).value_counts(normalize=True)
            cur_share = current.dropna().astype(str).value_counts(normalize=True)
            labels = list(ref_share.add(cur_share, fill_value=0).sort_values(ascending=False).index[:15])
            for share, label, color in ((ref_share, reference_label, "#1d5fd1"), (cur_share, current_label, "#d97706")):
                figure.add_trace(go.Bar(x=labels, y=[float(share.get(name, 0)) * 100 for name in labels],
                                        name=label, marker={"color": color}))
            figure.update_layout(barmode="group", yaxis_title="Kayıtların yüzdesi")
        figure.update_layout(title=f"{item.column}: {reference_label} ve {current_label}", xaxis_title=item.column)
        return localize_figure(figure)
