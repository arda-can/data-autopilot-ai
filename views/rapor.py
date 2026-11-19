"""Rapor: diğer sayfalardaki güncel seçimlerle HTML ve PDF yönetici raporu."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import streamlit as st

from core.report import build_report, render_html, render_pdf
from views.common import (
    automl_for, automl_selection, explanation_for, joined_active, remembered, require_result, stored_drift,
    timeseries_for, timeseries_request,
)

result = require_result()

st.header("📄 Yönetici Raporu")
st.caption(
    "Özet, öneriler, grafikler ve model sonucu tek dosyada. "
    "Ham satırlar ve audit log örnekleri rapora girmez."
)

# Rapor, diğer sayfalarda en son yapılan seçimleri kullanır; açılmamış sayfalar için otopilotun varsayılanları geçer.
ts_request = timeseries_request(result)
timeseries, timeseries_error = timeseries_for(result, ts_request)
target, settings = automl_selection(result)
automl, automl_error = automl_for(result, target, settings)
drift_request = remembered("drift_request")
drift_outcome = stored_drift(drift_request)
report_drift = None if drift_outcome is None or isinstance(drift_outcome, str) else drift_outcome.report

included = [
    f"zaman serisi: {timeseries.summary.date_column} ({timeseries.summary.frequency_label})"
    if timeseries is not None else "zaman serisi: yok",
    f"model hedefi: {target}" if automl is not None else "model: yok",
    "dönem karşılaştırması: var" if report_drift is not None else
    "dönem karşılaştırması: yok (Zaman Serisi sayfasında “Karşılaştır” ile eklenir)",
]
st.caption("Rapora girecekler — " + " · ".join(included))

report_key = (st.session_state.data_signature, target, ts_request, settings, str(drift_request))
if st.button("Raporu Hazırla", key="report_button"):
    with st.spinner("Rapor hazırlanıyor..."):
        explanation = explanation_for(target, settings, automl)
        report_explanation = None if explanation is None or isinstance(explanation, str) else explanation
        ts_figures = [] if timeseries is None else [
            figure for figure in (timeseries.forecast_figure, timeseries.seasonal_figure) if figure is not None
        ]
        pdp_figures = list(report_explanation.figures) if report_explanation is not None else []
        report = build_report(
            replace(
                result,
                suggested_target=target if automl is not None else result.suggested_target,
                automl_result=automl,
                automl_message=automl_error or result.automl_message,
                timeseries=timeseries,
                timeseries_message=timeseries_error,
            ),
            st.session_state.data_name,
            drift=report_drift,
            explanation=report_explanation,
            join_logs=st.session_state.get("join_logs") if joined_active() else None,
        )
        chart_figures = [chart.figure for chart in result.charts]
        st.session_state.report_files = {
            "key": report_key,
            "html": render_html(
                report, chart_figures, automl.figure if automl is not None else None,
                timeseries_figures=ts_figures, pdp_figures=pdp_figures,
            ).encode("utf-8"),
            "pdf": render_pdf(report, chart_figures, timeseries_figures=ts_figures, pdp_figures=pdp_figures),
        }

report_files = st.session_state.get("report_files")
if report_files and report_files["key"] == report_key:
    report_stem = Path(st.session_state.data_name).stem
    html_column, pdf_column = st.columns(2)
    html_column.download_button(
        "HTML raporu indir (etkileşimli grafikler)", report_files["html"],
        f"yonetici_raporu_{report_stem}.html", "text/html", width="stretch",
    )
    pdf_column.download_button(
        "PDF raporu indir", report_files["pdf"],
        f"yonetici_raporu_{report_stem}.pdf", "application/pdf", width="stretch",
    )
