"""Zaman Serisi: trend, mevsimsellik, tahmin, iki dönem/iki dosya karşılaştırması ve “Neden değişti?”."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift import DriftEngine, prepare_for_comparison, split_by_date, split_by_ranges, split_by_years
from core.formatting import decimal_tr, display_frame, integer_tr
from core.ingestion import content_signature, read_tabular_file
from core.rootcause import MetricSpec, RootCauseAnalyzer, candidate_dimensions
from core.timeseries import date_columns, value_columns
from views.common import (
    default_timeseries_request, index_of, remember, remembered, require_result, stored_drift, stored_root_cause,
    timeseries_for,
)

FREQUENCY_OPTIONS = {"Otomatik": "auto", "Günlük": "D", "Haftalık": "W", "Aylık": "M"}

result = require_result()
cleaned = result.cleaned_frame
ts_dates = date_columns(cleaned)


def _timeseries_section() -> None:
    st.header("📈 Zaman Serisi Otopilotu")
    default_date = default_timeseries_request(result)[0]
    metric_options = {"Kayıt sayısı": None, **{f"{column} toplamı": column for column in value_columns(cleaned)}}
    metric_labels, frequency_labels = list(metric_options), list(FREQUENCY_OPTIONS)
    date_column_ui, metric_ui, frequency_ui = st.columns(3)
    ts_date = date_column_ui.selectbox(
        "Tarih sütunu", ts_dates, index=index_of(ts_dates, remembered("ts_date"), ts_dates.index(default_date)),
        key="ts_date",
    )
    ts_metric = metric_ui.selectbox(
        "Ölçü", metric_labels, index=index_of(metric_labels, remembered("ts_metric")), key="ts_metric"
    )
    ts_frequency = frequency_ui.selectbox(
        "Frekans", frequency_labels, index=index_of(frequency_labels, remembered("ts_frequency")), key="ts_frequency"
    )
    request = (ts_date, metric_options[ts_metric], FREQUENCY_OPTIONS[ts_frequency])
    for key, value in (("ts_date", ts_date), ("ts_metric", ts_metric), ("ts_frequency", ts_frequency),
                       ("ts_request", request)):
        remember(key, value)

    timeseries, timeseries_error = timeseries_for(result, request)
    if timeseries is None:
        st.warning(timeseries_error or "Zaman serisi analizi yapılamadı.")
        return
    summary = timeseries.summary
    metrics = st.columns(4)
    metrics[0].metric("Frekans", summary.frequency_label.capitalize())
    metrics[1].metric("Dönem", integer_tr(summary.periods))
    metrics[2].metric(
        "Seviye Değişimi", "—" if summary.level_change is None else f"%{decimal_tr(summary.level_change * 100)}",
    )
    metrics[3].metric(
        "Mevsimsellik Gücü",
        "—" if summary.seasonality_strength is None else decimal_tr(summary.seasonality_strength, 2),
    )
    st.write(summary.trend_text)
    st.write(summary.seasonality_text)
    (st.success if summary.beats_baseline else st.warning)(summary.verdict)
    st.plotly_chart(timeseries.forecast_figure, width="stretch")
    st.caption(f"{summary.forecast_text} Model: {summary.model_name}.")
    if timeseries.seasonal_figure is not None:
        st.plotly_chart(timeseries.seasonal_figure, width="stretch")
    if summary.anomalies:
        st.subheader("Olağan dışı dönemler")
        st.dataframe(
            display_frame(pd.DataFrame([
                {"Dönem": item.label, "Gözlenen": item.observed, "Beklenen": item.expected,
                 "Sapma (z)": item.z, "Yön": item.direction}
                for item in summary.anomalies
            ])),
            width="stretch",
            hide_index=True,
        )
    for note in summary.notes:
        st.caption(f"Not: {note}")


def _drift_request() -> tuple[tuple | None, object]:
    """Karşılaştırma isteği ve (varsa) yüklenen ikinci dosya."""
    drift_modes = (["İki yıl", "Kesim tarihi"] if ts_dates else []) + ["İkinci dosya"]
    drift_mode = st.radio(
        "Karşılaştırma türü", drift_modes, index=index_of(drift_modes, remembered("drift_mode")),
        horizontal=True, key="drift_mode",
    )
    remember("drift_mode", drift_mode)
    if drift_mode == "İkinci dosya":
        # Yüklenen dosya sayfa değişince Streamlit tarafından bırakılır; dönüşte yeniden yüklenmelidir.
        upload = st.file_uploader("Karşılaştırılacak dosya (CSV veya Excel)", type=["csv", "xlsx"], key="drift_file")
        return (None if upload is None else ("file", upload.name, content_signature(upload))), upload

    drift_date = st.selectbox(
        "Tarih sütunu", ts_dates, index=index_of(ts_dates, remembered("drift_date")), key="drift_date"
    )
    remember("drift_date", drift_date)
    dates = cleaned[drift_date].dropna()
    if drift_mode == "Kesim tarihi":
        ordered = dates.sort_values()
        first, last = ordered.iloc[0].date(), ordered.iloc[-1].date()
        cut_default = remembered(f"drift_cut_{drift_date}", ordered.iloc[len(ordered) // 2].date())
        cut = st.date_input(
            "Kesim tarihi (bu tarih ve sonrası güncel dönem)",
            value=min(max(cut_default, first), last), min_value=first, max_value=last, key="drift_cut",
        )
        remember(f"drift_cut_{drift_date}", cut)
        return ("cut", drift_date, str(cut)), None

    years = sorted(int(year) for year in dates.dt.year.unique())
    if len(years) < 2:
        st.info("Bu tarih sütununda iki farklı yıl yok; kesim tarihiyle karşılaştırın.")
        return None, None
    reference_ui, current_ui = st.columns(2)
    reference_year = reference_ui.selectbox(
        "Referans yıl", years, index=index_of(years, remembered("drift_reference_year"), len(years) - 2),
        key="drift_reference_year",
    )
    current_year = current_ui.selectbox(
        "Güncel yıl", years, index=index_of(years, remembered("drift_current_year"), len(years) - 1),
        key="drift_current_year",
    )
    remember("drift_reference_year", reference_year)
    remember("drift_current_year", current_year)
    return ("years", drift_date, reference_year, current_year), None


def _drift_section() -> None:
    st.header("🔀 Dönem Karşılaştırması")
    st.caption(
        "İki dönem ya da iki dosya arasında şema, eksik oranı ve dağılım farkı. "
        "Trafik ışığı etki büyüklüğüne (PSI) göre verilir; ham satırlar karşılaştırılmaz."
    )
    request, upload = _drift_request()
    remember("drift_request", request)
    if request is not None and st.button("Karşılaştır", key="drift_button"):
        with st.spinner("Dönemler karşılaştırılıyor..."):
            try:
                if request[0] == "years":
                    parts = split_by_years(cleaned, request[1], request[2], request[3])
                elif request[0] == "cut":
                    parts = split_by_date(cleaned, request[1], request[2])
                else:
                    # İki dosya da aynı yapısal hijyenden geçer; istatistiksel dönüşüm yapılmaz.
                    parts = (
                        prepare_for_comparison(st.session_state.dataframe),
                        prepare_for_comparison(read_tabular_file(upload, upload.name)),
                        st.session_state.data_name,
                        upload.name,
                    )
                outcome = DriftEngine().compare(*parts)
            except ValueError as exc:
                outcome = f"Karşılaştırma yapılamadı: {exc}"
            st.session_state.drift_result = (st.session_state.data_signature, request, outcome)

    outcome = stored_drift(request)
    if outcome is None:
        return
    if isinstance(outcome, str):
        st.warning(outcome)
        return
    report = outcome.report
    st.write(report.summary)
    light_columns = st.columns(3)
    light_columns[0].metric("🔴 Kırmızı", report.counts["kırmızı"])
    light_columns[1].metric("🟡 Sarı", report.counts["sarı"])
    light_columns[2].metric("🟢 Yeşil", report.counts["yeşil"])
    for change in report.schema_changes:
        st.warning(f"Şema: “{change.column}” {change.change} ({change.detail})")
    if report.columns:
        lights = {"kırmızı": "🔴", "sarı": "🟡", "yeşil": "🟢"}
        st.dataframe(
            display_frame(pd.DataFrame([
                {
                    "Durum": f"{lights[item.status]} {item.status}", "Sütun": item.column, "Tür": item.kind,
                    "PSI": item.psi, report.reference_label: item.reference_summary,
                    report.current_label: item.current_summary, "Açıklama": item.explanation,
                    "Test": item.test, "p-değeri": item.p_value,
                }
                for item in report.columns
            ])),
            width="stretch",
            hide_index=True,
        )
    if outcome.figure is not None:
        st.plotly_chart(outcome.figure, width="stretch")
    if outcome.detail_figures:
        with st.expander("Değişen sütunların dağılımları"):
            for figure in outcome.detail_figures.values():
                st.plotly_chart(figure, width="stretch")
    if report.skipped:
        with st.expander(f"Karşılaştırılmayan sütunlar ({len(report.skipped)})"):
            for note in report.skipped:
                st.caption(note)
    st.caption(report.method_note)


METRIC_KINDS = {"Kayıt sayısı": "sayım", "Toplam": "toplam", "Ortalama": "ortalama", "Oran (bir değerin payı)": "oran"}


def _remembered_date(key: str, default, first, last):
    return min(max(remembered(key, default), first), last)


def _root_cause_periods(dates: pd.Series) -> tuple | None:
    period_modes = ["İki tarih aralığı", "İki yıl", "Kesim tarihi"]
    mode = st.selectbox("Dönemler", period_modes, index=index_of(period_modes, remembered("rc_mode")), key="rc_mode")
    remember("rc_mode", mode)
    first, last = dates.min().date(), dates.max().date()
    if mode == "İki tarih aralığı":
        # Varsayılan: son üç ay ile ondan önceki üç ay.
        current_start = (pd.Timestamp(last) - pd.DateOffset(months=3) + pd.Timedelta(days=1)).date()
        defaults = (
            (pd.Timestamp(current_start) - pd.DateOffset(months=3)).date(),
            (pd.Timestamp(current_start) - pd.Timedelta(days=1)).date(),
            current_start, last,
        )
        values = []
        for column, key, label, default in zip(
            st.columns(4), ("rc_ref_start", "rc_ref_end", "rc_cur_start", "rc_cur_end"),
            ("Referans başlangıç", "Referans bitiş", "Güncel başlangıç", "Güncel bitiş"), defaults,
        ):
            value = column.date_input(label, value=_remembered_date(key, max(default, first), first, last),
                                      min_value=first, max_value=last, key=key)
            remember(key, value)
            values.append(str(value))
        return ("ranges", (values[0], values[1]), (values[2], values[3]))
    if mode == "İki yıl":
        years = sorted(int(year) for year in dates.dt.year.unique())
        if len(years) < 2:
            st.info("Bu tarih sütununda iki farklı yıl yok; tarih aralığıyla karşılaştırın.")
            return None
        left, right = st.columns(2)
        reference_year = left.selectbox(
            "Referans yıl", years, index=index_of(years, remembered("rc_ref_year"), len(years) - 2), key="rc_ref_year"
        )
        current_year = right.selectbox(
            "Güncel yıl", years, index=index_of(years, remembered("rc_cur_year"), len(years) - 1), key="rc_cur_year"
        )
        remember("rc_ref_year", reference_year)
        remember("rc_cur_year", current_year)
        return ("years", reference_year, current_year)
    middle = dates.sort_values().iloc[len(dates) // 2].date()
    cut = st.date_input("Kesim tarihi (bu tarih ve sonrası güncel dönem)",
                        value=_remembered_date("rc_cut", middle, first, last), min_value=first, max_value=last,
                        key="rc_cut")
    remember("rc_cut", cut)
    return ("cut", str(cut))


def _root_cause_metric() -> tuple | None:
    kinds = list(METRIC_KINDS)
    kind_ui, column_ui, value_ui = st.columns(3)
    kind_label = kind_ui.selectbox("Metrik", kinds, index=index_of(kinds, remembered("rc_kind")), key="rc_kind")
    remember("rc_kind", kind_label)
    kind = METRIC_KINDS[kind_label]
    column, value = None, None
    if kind in ("toplam", "ortalama"):
        options = value_columns(cleaned)
        if not options:
            st.info("Toplam ve ortalama için sayısal sütun yok.")
            return None
        column = column_ui.selectbox("Sütun", options, index=index_of(options, remembered("rc_column")),
                                     key="rc_column")
    elif kind == "oran":
        options = [
            str(name) for name in cleaned.columns
            if not pd.api.types.is_datetime64_any_dtype(cleaned[name]) and 2 <= cleaned[name].nunique() <= 20
        ]
        if not options:
            st.info("Oran için en fazla 20 farklı değerli bir sütun gerekir.")
            return None
        column = column_ui.selectbox("Sütun", options, index=index_of(options, remembered("rc_column")),
                                     key="rc_rate_column")
        choices = sorted(cleaned[column].dropna().astype(str).unique())
        value = value_ui.selectbox("Değer", choices, index=index_of(choices, remembered("rc_value")), key="rc_value")
        remember("rc_value", value)
    remember("rc_column", column)
    return (kind, column, value)


def _root_cause_request() -> tuple | None:
    """(tarih sütunu, dönemler, metrik, boyutlar) ya da eksik seçimde None."""
    date_column = st.selectbox("Tarih sütunu", ts_dates, index=index_of(ts_dates, remembered("rc_date")),
                               key="rc_date")
    remember("rc_date", date_column)
    periods = _root_cause_periods(cleaned[date_column].dropna())
    metric = _root_cause_metric()
    if periods is None or metric is None:
        return None
    available, _ = candidate_dimensions(cleaned, tuple(item for item in (metric[1],) if item))
    chosen = [item for item in remembered("rc_dimensions", available) if item in available]
    dimensions = st.multiselect("Boyutlar", available, default=chosen or available, key="rc_dimensions")
    remember("rc_dimensions", dimensions)
    if not dimensions:
        st.info("En az bir boyut seçin.")
        return None
    return (date_column, periods, metric, tuple(dimensions))


def _group_rows(dimension, change: float) -> list[dict]:
    rows = []
    for group in dimension.groups:
        row = {
            "Grup": group.group, "Referans kayıt": group.reference_rows, "Güncel kayıt": group.current_rows,
            "Referans": group.reference_value, "Güncel": group.current_value, "Katkı": group.contribution,
            "Değişimdeki pay": group.share_of_change, "Beklenen pay": group.expected / change if change else 0.0,
        }
        if group.mix_effect is not None:
            row |= {"Karma etkisi": group.mix_effect, "Grup içi değişim": group.rate_effect}
        row["Not"] = "; ".join(filter(None, [
            f"yalnızca {group.only_in} dönemde" if group.only_in else "", "küçük grup" if group.small else "",
        ]))
        rows.append(row)
    return rows


def _root_cause_section() -> None:
    st.header("🔎 Neden Değişti?")
    st.caption(
        "Bir metriğin iki dönem arasındaki değişimini boyut değerlerinin katkılarına ayırır (ör. düşüşün ne kadarı "
        "hangi şehirden). Katkıların toplamı değişime birebir eşittir; nedensellik iddiası taşımaz."
    )
    request = _root_cause_request()
    remember("rc_request", request)
    if request is not None and st.button("Değişimi ayrıştır", key="rc_button"):
        date_column, periods, (kind, column, value), dimensions = request
        try:
            if periods[0] == "ranges":
                parts = split_by_ranges(cleaned, date_column, periods[1], periods[2])
            elif periods[0] == "years":
                parts = split_by_years(cleaned, date_column, periods[1], periods[2])
            else:
                parts = split_by_date(cleaned, date_column, periods[1])
            outcome = RootCauseAnalyzer().analyze(
                *parts, MetricSpec(kind=kind, column=column, value=value), dimensions=list(dimensions),
            )
        except ValueError as exc:
            outcome = f"Ayrıştırma yapılamadı: {exc}"
        st.session_state.root_cause_result = (st.session_state.data_signature, request, outcome)

    outcome = stored_root_cause(request)
    if outcome is None:
        return
    if isinstance(outcome, str):
        st.warning(outcome)
        return
    report = outcome.report
    (st.success if report.change else st.info)(report.headline)
    if outcome.figure is not None:
        st.plotly_chart(outcome.figure, width="stretch")
    if report.dimensions:
        st.dataframe(
            display_frame(pd.DataFrame([
                {"Boyut": item.dimension, "Beklenenden sapma": item.score, "Özet": item.summary}
                for item in report.dimensions
            ])),
            width="stretch", hide_index=True,
        )
        for position, dimension in enumerate(report.dimensions):
            with st.expander(f"“{dimension.dimension}” grupları", expanded=position == 0):
                st.dataframe(display_frame(pd.DataFrame(_group_rows(dimension, report.change))),
                             width="stretch", hide_index=True)
    for note in report.skipped:
        st.caption(note)
    st.caption(report.method_note)


if ts_dates:
    _timeseries_section()
else:
    st.info("Bu veride tarih sütunu bulunmadı; zaman serisi analizi yapılamaz. İki dosyayı yine de karşılaştırabilirsiniz.")
_drift_section()
if ts_dates:
    _root_cause_section()
