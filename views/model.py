"""Model: hedef seçimi, tahmin anı ayarları, AutoML sonucu, kısmi bağımlılık ve what-if."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.explain import PDP_NOTE, ModelExplainer
from core.formatting import display_frame, integer_tr
from core.timeseries import date_columns
from views.common import (
    DEFAULT_AUTOML_SETTINGS, NO_TARGET, automl_for, automl_selection, explanation_for, remember, remembered,
    require_result,
)

result = require_result()
cleaned = result.cleaned_frame

st.header("🤖 Otonom Makine Öğrenmesi (AutoML) Sonucu")
target_options = [str(column) for column in cleaned.columns]
remembered_target, _ = automl_selection(result)
target_options_for_ui = target_options if result.suggested_target in target_options else [NO_TARGET, *target_options]
selected_target = st.selectbox(
    "Hedef Sütun (Otopilot seçti, isterseniz değiştirebilirsiniz)",
    target_options_for_ui,
    index=target_options_for_ui.index(remembered_target) if remembered_target in target_options_for_ui else 0,
    key="autopilot_target",
)
target = None if selected_target == NO_TARGET else selected_target
remember("automl_target", target)

settings = DEFAULT_AUTOML_SETTINGS
if target is not None:
    with st.expander("Değerlendirme ve tahmin anı ayarları"):
        st.caption(
            "Model, eğitim bölmesinde çapraz doğrulamayla seçilir; test bölmesi seçime katılmaz. Tahmin anında "
            "hangi bilgilerin bilindiğini belirterek sızıntıyı önleyin ve gerekirse zamana göre değerlendirin."
        )
        time_options = ["Yok (rastgele bölme)", *date_columns(cleaned)]
        feature_options = [column for column in target_options if column != target]
        time_key, exclude_key, known_key = (
            f"automl_time_{target}", f"automl_exclude_{target}", f"automl_known_{target}",
        )
        remembered_time = remembered(time_key)
        evaluation_time = st.selectbox(
            "Zaman sırası sütunu (test = en son %20 dönem)", time_options,
            index=time_options.index(remembered_time) if remembered_time in time_options else 0, key=time_key,
        )
        excluded_columns = st.multiselect(
            "Tahmin anında bilinmeyen sütunlar (modelden çıkarılır)", feature_options,
            default=[column for column in remembered(exclude_key, []) if column in feature_options], key=exclude_key,
        )
        known_columns = st.multiselect(
            "Tahmin anında bilindiğini onayladığınız sütunlar (sızıntı uyarısı verilmez)", feature_options,
            default=[column for column in remembered(known_key, []) if column in feature_options], key=known_key,
        )
        for key, value in ((time_key, evaluation_time), (exclude_key, excluded_columns), (known_key, known_columns)):
            remember(key, value)
    settings = (
        None if evaluation_time == time_options[0] else evaluation_time,
        tuple(sorted(excluded_columns)), tuple(sorted(known_columns)),
    )
remember("automl_settings", settings)

automl, automl_error = automl_for(result, target, settings)
if target is not None:
    st.write(f"**Tahmin edilen hedef sütun:** `{target}`")
if automl is None:
    st.info(automl_error or result.automl_message or "Uygun hedef sütun bulunamadığı için AutoML atlandı.")
    st.stop()

st.write(f"**Problem tipi:** {automl.problem_type}")
st.write(f"**En iyi model:** {automl.best_model}")
(st.success if automl.beats_baseline and not automl.leakage_warnings else st.warning)(automl.verdict)
st.caption(
    f"Değerlendirme: {automl.split_method} · eğitim {integer_tr(automl.train_rows)} / test "
    f"{integer_tr(automl.test_rows)} satır · {automl.cv_folds} katlı çapraz doğrulama"
    + (f" · çıkarılan: {', '.join(automl.excluded_columns)}" if automl.excluded_columns else "")
    + (f" · tahmin anında bilindiği onaylanan: {', '.join(automl.acknowledged_columns)}"
       if automl.acknowledged_columns else "")
)
for leakage_warning in automl.leakage_warnings:
    st.warning(leakage_warning)
if automl.leakage_warnings:
    st.caption(
        "Bu bilgi tahmin anında bilinmiyorsa “Değerlendirme ve tahmin anı ayarları” bölümünde modelden "
        "çıkarın; biliniyorsa onaylayın, uyarı kalkar."
    )
st.subheader("Leaderboard")
st.dataframe(display_frame(automl.leaderboard), width="stretch", hide_index=True)
st.subheader("Özellik Önemi")
st.plotly_chart(automl.figure, width="stretch")

explanation = explanation_for(target, settings, automl)
if explanation is None:
    st.stop()
st.subheader("Model neden böyle tahmin ediyor? (Kısmi bağımlılık)")
if isinstance(explanation, str):
    st.info(explanation)
    st.stop()
st.caption(PDP_NOTE)
for curve, curve_figure in zip(explanation.curves, explanation.figures):
    st.plotly_chart(curve_figure, width="stretch")
    (st.warning if curve.interpretation.startswith("Uyarı") else st.write)(curve.interpretation)
for note in explanation.skipped:
    st.caption(note)
if explanation.what_if_features:
    st.subheader("What-if: değerleri değiştirip tahmini görün")
    st.caption("Diğer özellikler tipik değerlerinde (sayısalda medyan, kategoride en sık değer) sabit tutulur.")
    with st.form(key=f"what_if_{target}"):
        what_if_values = {}
        for feature in explanation.what_if_features:
            if feature.kind == "sayısal":
                span = feature.maximum - feature.minimum
                what_if_values[feature.name] = st.slider(
                    feature.name, min_value=float(feature.minimum), max_value=float(feature.maximum),
                    value=float(min(max(feature.default, feature.minimum), feature.maximum)),
                    step=float(span / 100) if span > 0 else 1.0,
                )
            else:
                what_if_values[feature.name] = st.selectbox(
                    feature.name, feature.options, index=feature.options.index(str(feature.default)),
                )
        submitted = st.form_submit_button("Tahmin et")
    if submitted:
        try:
            prediction = ModelExplainer().what_if(automl, what_if_values)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.success(prediction.text)
            if prediction.probabilities:
                st.dataframe(
                    display_frame(pd.DataFrame(
                        [{"Sınıf": label, "Olasılık": value} for label, value in prediction.probabilities.items()]
                    )),
                    width="stretch", hide_index=True,
                )
