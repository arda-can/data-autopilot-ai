"""Deney: A/B test analizi (oran/ortalama testi, güven aralığı, güç ve örneklem, SRM kontrolü)."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.columns import is_text_series
from core.experiment import ExperimentAnalyzer, ExperimentSpec, detect_experiment
from core.formatting import decimal_tr, display_frame, integer_tr
from views.common import index_of, remember, remembered, require_result, stored_experiment

ROW_LEVEL = "— satır düzeyi (her kayıt bağımsız) —"

result = require_result()
cleaned = result.cleaned_frame

st.header("🧪 A/B Test Analizi")
st.caption(
    "İki grubu bir oran ya da ortalama metriğinde karşılaştırır: anlamlılık testi, güven aralığı, örneklem oranı "
    "uyumu (SRM) ve güç/örneklem hesabı. Randomizasyon birimi (ör. müşteri) seçilirse analiz o düzeyde yapılır."
)

group_options = [str(column) for column in cleaned.columns
                 if is_text_series(cleaned[column]) and 2 <= cleaned[column].nunique() <= 10]
if not group_options:
    st.info("Deney analizi için 2–10 farklı değerli bir grup sütunu gerekir (ör. A/B).")
    st.stop()
detected = detect_experiment(cleaned)
default = remembered("experiment_spec", detected.model_dump() if detected else {})

left, middle, right = st.columns(3)
group_column = left.selectbox("Grup sütunu", group_options, index=index_of(group_options, default.get("group_column")),
                              key="ab_group")
labels = sorted(cleaned[group_column].dropna().astype(str).unique())
control = middle.selectbox("Kontrol grubu", labels, index=index_of(labels, default.get("control")), key="ab_control")
treatment_labels = [label for label in labels if label != control]
treatment = right.selectbox("Deney grubu", treatment_labels,
                            index=index_of(treatment_labels, default.get("treatment")), key="ab_treatment")

kinds = {"Oran (bir değerin payı)": "oran", "Ortalama": "ortalama"}
left, middle, right = st.columns(3)
kind_labels = list(kinds)
kind_label = left.selectbox(
    "Metrik türü", kind_labels,
    index=index_of(list(kinds.values()), default.get("metric_kind", "oran")), key="ab_kind",
)
kind = kinds[kind_label]
success_value = None
if kind == "oran":
    metric_options = [str(column) for column in cleaned.columns
                      if column != group_column and 2 <= cleaned[column].nunique() <= 20]
else:
    metric_options = [str(column) for column in cleaned.columns
                      if pd.api.types.is_numeric_dtype(cleaned[column]) and not pd.api.types.is_bool_dtype(cleaned[column])]
if not metric_options:
    st.info("Bu metrik türü için uygun sütun yok.")
    st.stop()
metric_column = middle.selectbox("Metrik sütunu", metric_options,
                                 index=index_of(metric_options, default.get("metric_column")), key="ab_metric")
if kind == "oran":
    values = sorted(cleaned[metric_column].dropna().astype(str).unique())
    success_value = right.selectbox("Başarı değeri", values, index=index_of(values, default.get("success_value")),
                                    key="ab_success")

unit_options = [ROW_LEVEL, *[str(column) for column in cleaned.columns
                             if column not in (group_column, metric_column) and is_text_series(cleaned[column])
                             and cleaned[column].nunique() > 20]]
with st.expander("Birim, beklenen dağılım ve güç ayarları", expanded=False):
    left, right = st.columns(2)
    unit = left.selectbox("Randomizasyon birimi", unit_options,
                          index=index_of(unit_options, default.get("unit_column")), key="ab_unit",
                          help="Atama müşteri bazındaysa müşteri sütununu seçin; aynı müşterinin kayıtları bağımsız değildir.")
    expected = right.number_input("Kontrol grubunun beklenen payı (%)", min_value=1.0, max_value=99.0,
                                  value=float(default.get("expected_control_share", 0.5)) * 100, step=1.0,
                                  key="ab_expected")
    left, middle, right = st.columns(3)
    alpha_options = [0.01, 0.05, 0.10]
    alpha = left.selectbox("Anlamlılık düzeyi (α)", alpha_options,
                           index=index_of(alpha_options, default.get("alpha"), 1), key="ab_alpha")
    power_options = [0.8, 0.9]
    power = middle.selectbox("Hedef güç", power_options, index=index_of(power_options, default.get("power")),
                             key="ab_power")
    mde = right.number_input("Yakalanmak istenen göreli fark (MDE, %)", min_value=0.5, max_value=500.0,
                             value=float(default.get("mde_relative", 0.10)) * 100, step=0.5, key="ab_mde")

try:
    spec = ExperimentSpec(
        group_column=group_column, control=control, treatment=treatment, metric_column=metric_column,
        metric_kind=kind, success_value=success_value, unit_column=None if unit == ROW_LEVEL else unit,
        expected_control_share=expected / 100, alpha=alpha, power=power, mde_relative=mde / 100,
    )
except ValidationError as exc:
    st.error("; ".join(str(item["msg"]).removeprefix("Value error, ") for item in exc.errors()))
    st.stop()
remember("experiment_spec", spec.model_dump())

if st.button("Deneyi analiz et", key="ab_button"):
    try:
        outcome = ExperimentAnalyzer().analyze(cleaned, spec)
    except ValueError as exc:
        outcome = f"Deney analizi yapılamadı: {exc}"
    st.session_state.experiment_result = (st.session_state.data_signature, spec.model_dump_json(), outcome)

outcome = stored_experiment(spec)
if outcome is None:
    st.stop()
if isinstance(outcome, str):
    st.warning(outcome)
    st.stop()
report = outcome.report
if report.srm.alarm:
    st.error(report.verdict)
elif report.significant:
    st.success(report.verdict)
else:
    st.warning(report.verdict)

metric_scale = 100 if spec.metric_kind == "oran" else 1
prefix = "%" if spec.metric_kind == "oran" else ""
metrics = st.columns(4)
metrics[0].metric(f"Kontrol ({report.control.label})", f"{prefix}{decimal_tr(report.control.mean * metric_scale, 2)}")
metrics[1].metric(f"Deney ({report.treatment.label})", f"{prefix}{decimal_tr(report.treatment.mean * metric_scale, 2)}")
metrics[2].metric(
    "Göreli fark", "—" if report.relative_lift is None else f"%{decimal_tr(report.relative_lift * 100)}",
    help="Güven aralığı yaklaşık; kontrol ortalamasının belirsizliğini yok sayar.",
)
metrics[3].metric("p-değeri", "< 0,001" if report.p_value < 0.001 else decimal_tr(report.p_value, 3))
st.plotly_chart(outcome.figure, width="stretch")

st.subheader("Örneklem oranı uyumu (SRM)")
srm = report.srm
(st.error if srm.alarm else st.caption)(
    f"Kontrol {integer_tr(srm.control_units)}, deney {integer_tr(srm.treatment_units)} birim; gözlenen kontrol payı "
    f"%{decimal_tr(srm.observed_control_share * 100)}, beklenen %{decimal_tr(srm.expected_control_share * 100)} "
    f"(ki-kare p = {decimal_tr(srm.p_value, 4)}; alarm eşiği 0,001)."
)
st.subheader("Güç ve örneklem")
power_report = report.power
lines = []
if power_report.detectable_relative is not None:
    lines.append(f"Mevcut örneklemle %{decimal_tr(power_report.power_target * 100, 0)} güçle yakalanabilecek en küçük "
                 f"göreli fark: %{decimal_tr(power_report.detectable_relative * 100)}.")
if power_report.power_at_mde is not None:
    lines.append(f"Hedeflenen %{decimal_tr(power_report.mde_relative * 100)} göreli farkta güç: "
                 f"%{decimal_tr(power_report.power_at_mde * 100)}.")
if power_report.required_per_group is not None:
    lines.append(f"Bu fark için grup başına gereken birim: {integer_tr(power_report.required_per_group)} "
                 f"(şu an kontrol {integer_tr(report.control.units)}, deney {integer_tr(report.treatment.units)}).")
for line in lines:
    st.write(line)

st.dataframe(
    display_frame(pd.DataFrame([
        {"Grup": f"{group.label} ({role})", "Birim": group.units, "Kayıt": group.rows, "Ortalama": group.mean,
         "Std. sapma": group.sd, "GA alt": group.ci_low, "GA üst": group.ci_high}
        for group, role in ((report.control, "kontrol"), (report.treatment, "deney"))
    ])),
    width="stretch", hide_index=True,
)
st.caption(f"Test: {report.test_name} · analiz birimi: {report.analysis_unit} · fark güven aralığı "
           f"{decimal_tr(report.ci_low * metric_scale, 3)} – {decimal_tr(report.ci_high * metric_scale, 3)}"
           + (" puan" if spec.metric_kind == "oran" else ""))
for note in report.notes:
    st.caption(f"Not: {note}")
st.caption(report.method_note)
