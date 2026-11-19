"""Zaman Serisi: trend, mevsimsellik, tahmin ve iki dönem/iki dosya karşılaştırması."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.drift import DriftEngine, prepare_for_comparison, split_by_date, split_by_years
from core.formatting import decimal_tr, display_frame, integer_tr
from core.ingestion import content_signature, read_tabular_file
from core.timeseries import date_columns, value_columns
from views.common import (
    default_timeseries_request, index_of, remember, remembered, require_result, stored_drift, timeseries_for,
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


if ts_dates:
    _timeseries_section()
else:
    st.info("Bu veride tarih sütunu bulunmadı; zaman serisi analizi yapılamaz. İki dosyayı yine de karşılaştırabilirsiniz.")
_drift_section()
