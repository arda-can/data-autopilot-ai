"""Zero-Click Autopilot Streamlit arayüzü."""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from core.autopilot import AutopilotEngine, AutopilotResult
from core.automl import AutoMLEngine
from core.chat import DataChatEngine
from core.ingestion import (
    generate_sample_dirty_data,
    list_local_csv_files,
    read_local_csv,
    read_tabular_file,
)
from core.formatting import decimal_tr, display_frame
from core.report import build_report, render_html, render_pdf


def _model_dict(value: BaseModel) -> dict:
    return value.model_dump()


st.set_page_config(page_title="Data Autopilot AI", page_icon="🚀", layout="wide")
st.title("🚀 Data Autopilot AI")
st.caption("Sıfır tıklamalı, güvenli ve açıklanabilir otonom veri bilimi")

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None
if "autopilot_result" not in st.session_state:
    st.session_state.autopilot_result = None
if "data_signature" not in st.session_state:
    st.session_state.data_signature = None
if "data_name" not in st.session_state:
    st.session_state.data_name = "veri"

with st.sidebar:
    st.header("Veri Kaynağı")
    api_key = st.text_input(
        "Gemini API Key (isteğe bağlı)",
        type="password",
        help="Boş bırakılırsa KVKK maskeli metadata üzerinden güvenli yerel fallback kullanılır.",
    )
    files = list_local_csv_files()
    selected = st.selectbox("Yerel CSV seçin", ["-- Seçin --", *[p.name for p in files]])
    uploaded = st.file_uploader("CSV veya Excel yükleyin", type=["csv", "xlsx", "xls"])
    if st.button("Örnek Kirli Türkçe Veri Üret ve Yükle", width="stretch"):
        path = generate_sample_dirty_data()
        st.session_state.dataframe = read_local_csv(path)
        st.session_state.data_signature = f"sample:{path.stat().st_mtime_ns}"
        st.session_state.data_name = path.name
        st.session_state.autopilot_result = None
        st.rerun()

if uploaded is not None:
    source_signature = f"upload:{uploaded.name}:{uploaded.size}"
    if st.session_state.data_signature != source_signature:
        st.session_state.dataframe = read_tabular_file(uploaded, uploaded.name)
        st.session_state.data_signature = source_signature
        st.session_state.data_name = uploaded.name
        st.session_state.autopilot_result = None
elif selected != "-- Seçin --":
    path = next(path for path in files if path.name == selected)
    source_signature = f"local:{path}:{path.stat().st_mtime_ns}"
    if st.session_state.data_signature != source_signature:
        st.session_state.dataframe = read_local_csv(path)
        st.session_state.data_signature = source_signature
        st.session_state.data_name = path.name
        st.session_state.autopilot_result = None

frame = st.session_state.dataframe
if frame is None:
    st.info("Başlamak için sol menüden bir CSV seçin, Excel/CSV yükleyin veya örnek veri üretin.")
    st.stop()

if st.session_state.autopilot_result is None:
    with st.spinner("Otopilot veriyi temizliyor ve en mantıklı analizleri hazırlıyor..."):
        try:
            st.session_state.autopilot_result = AutopilotEngine().run_full_autopilot(
                frame, api_key=api_key
            )
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Otopilot çalıştırılamadı: {exc}")
            st.stop()

result: AutopilotResult = st.session_state.autopilot_result
before = result.before_report
after = result.after_report
cleaned = result.cleaned_frame

st.header("🚀 Otopilot Yönetici Özeti")
metrics = st.columns(4)
metrics[0].metric(
    "Kalite Skoru",
    f"{decimal_tr(after.kalite_skoru)}/100",
    delta=decimal_tr(after.kalite_skoru - before.kalite_skoru),
)
metrics[1].metric("Satır Değişimi", f"{before.satır_sayısı} → {after.satır_sayısı}")
metrics[2].metric("Sütun Değişimi", f"{before.sütun_sayısı} → {after.sütun_sayısı}")
metrics[3].metric("Otomatik İşlem", len(result.audit_logs))

st.write("**Başlangıçta tespit edilen veri sorunları**")
issues = before.sorunlar if before.sorunlar != ["Belirgin bir kalite sorunu tespit edilmedi."] else [
    "Belirgin bir başlangıç kalite sorunu tespit edilmedi; standart güvenlik kontrolleri uygulandı."
]
for issue in issues:
    st.write(f"• {issue}")
st.caption(
    "Eksik ve aykırı değerler bilinçli olarak değiştirilmez: doldurma ve sınırlama yalnızca "
    "modelleme sırasında, eğitim verisinden öğrenilerek uygulanır (veri sızıntısı önlemi)."
)
for finding in after.domain_findings:
    st.warning(
        f"{finding.column}: {finding.count} negatif değer şüpheli. "
        "Alan kuralı gereği negatif olmamalı; kaynak kayıtları inceleyin. Değerler değiştirilmedi."
    )
for finding in after.category_variants:
    st.info(
        f"{finding.column}: aksan veya büyük/küçük harf farkıyla oluşmuş "
        f"{finding.group_count} olası kategori kopyası grubu ({finding.affected_count} kayıt). "
        "Birleştirme öncesinde incelenmesi önerilir; değerler değiştirilmedi."
    )
if result.outliers:
    with st.expander(f"Şüpheli aykırı değerler: {len(result.outliers)} sütun (değiştirilmedi)"):
        st.dataframe(
            display_frame(pd.DataFrame([_model_dict(item) for item in result.outliers]).rename(columns={
                "column": "Sütun", "count": "Aykırı Sayısı", "ratio": "Oran",
                "lower_bound": "Alt Sınır", "upper_bound": "Üst Sınır",
                "min": "En Küçük", "max": "En Büyük",
            })),
            width="stretch",
            hide_index=True,
        )
buffer = io.BytesIO()
cleaned.to_csv(buffer, index=False, encoding="utf-8-sig")
st.download_button(
    "Temizlenmiş CSV'yi indir",
    buffer.getvalue(),
    "otopilot_temizlenmis_veri.csv",
    "text/csv",
)

st.header("📊 Veriye Özel Seçilen En Mantıklı Görseller ve Bulgular")
if result.charts:
    for start in range(0, len(result.charts), 2):
        columns = st.columns(2)
        for column, chart in zip(columns, result.charts[start:start + 2]):
            with column:
                st.subheader(chart.title)
                st.plotly_chart(chart.figure, width="stretch")
                st.caption(f"**Neden seçildi:** {chart.reason}")
                st.write(f"**Bulgu:** {chart.insight}")
else:
    st.info("Bu veri yapısı için güvenilir ve anlamlı otomatik grafik üretilemedi.")

st.header("🤖 Otonom Makine Öğrenmesi (AutoML) Sonucu")
target_options = [str(column) for column in cleaned.columns]
if result.suggested_target in target_options:
    target_options_for_ui = target_options
    default_index = target_options.index(result.suggested_target)
else:
    target_options_for_ui = ["-- Hedef seçin --", *target_options]
    default_index = 0
selected_target = st.selectbox(
    "Hedef Sütun (Otopilot seçti, isterseniz değiştirebilirsiniz)",
    target_options_for_ui,
    index=default_index,
    key="autopilot_target",
)
automl = result.automl_result
automl_error = ""
if selected_target != "-- Hedef seçin --" and (
    result.suggested_target is not None and selected_target != result.suggested_target
    or result.suggested_target is None and selected_target != target_options_for_ui[0]
):
    # Aynı veri ve hedef için model yalnızca bir kez eğitilir; her yeniden çizimde tekrarlanmaz.
    cache = st.session_state.setdefault("automl_cache", {})
    cache_key = (st.session_state.data_signature, selected_target)
    if cache_key not in cache:
        sample = cleaned.sample(10_000, random_state=42) if len(cleaned) > 10_000 else cleaned
        try:
            cache[cache_key] = AutoMLEngine().run(sample, selected_target)
        except (ValueError, TypeError, KeyError) as exc:
            cache[cache_key] = f"'{selected_target}' hedefi için AutoML çalıştırılamadı: {exc}"
    cached = cache[cache_key]
    automl, automl_error = (None, cached) if isinstance(cached, str) else (cached, "")
if selected_target != "-- Hedef seçin --":
    st.write(f"**Tahmin edilen hedef sütun:** `{selected_target}`")
if automl is not None:
    st.write(f"**Problem tipi:** {automl.problem_type}")
    st.write(f"**En iyi model:** {automl.best_model}")
    (st.success if automl.beats_baseline else st.warning)(automl.verdict)
    st.subheader("Leaderboard")
    st.dataframe(display_frame(automl.leaderboard), width="stretch", hide_index=True)
    st.subheader("Özellik Önemi")
    st.plotly_chart(automl.figure, width="stretch")
else:
    st.info(automl_error or result.automl_message or "Uygun hedef sütun bulunamadığı için AutoML atlandı.")

st.header("📄 Yönetici Raporu")
st.caption(
    "Özet, öneriler, grafikler ve model sonucu tek dosyada. "
    "Ham satırlar ve audit log örnekleri rapora girmez."
)
report_key = (st.session_state.data_signature, selected_target)
if st.button("Raporu Hazırla", key="report_button"):
    with st.spinner("Rapor hazırlanıyor..."):
        report = build_report(
            replace(
                result,
                suggested_target=selected_target if automl is not None else result.suggested_target,
                automl_result=automl,
                automl_message=automl_error or result.automl_message,
            ),
            st.session_state.data_name,
        )
        st.session_state.report_files = {
            "key": report_key,
            "html": render_html(
                report, [chart.figure for chart in result.charts], automl.figure if automl is not None else None
            ).encode("utf-8"),
            "pdf": render_pdf(report, [chart.figure for chart in result.charts]),
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

st.header("🔍 Detaylar & Manuel Müdahale")
with st.expander("Audit Log'u göster"):
    logs = [_model_dict(item) for item in result.audit_logs]
    st.dataframe(display_frame(pd.DataFrame(logs)), width="stretch", hide_index=True)
with st.expander("KVKK maskeli Metadata Kartını göster"):
    st.json(_model_dict(result.metadata))
with st.expander("Temizlenmiş veri önizlemesini göster"):
    st.dataframe(display_frame(cleaned.head(100)), width="stretch")
with st.expander("Veriyle sohbet et"):
    question = st.text_input(
        "Türkçe özel sorunuz",
        placeholder="Örn: Şehirlere göre ortalama harcama nedir?",
        key="autopilot_chat_question",
    )
    if st.button("Soruyu Analiz Et", key="autopilot_chat_button"):
        try:
            chat_result, summary, figure, query = DataChatEngine(api_key).answer(
                cleaned, question, result.metadata
            )
            st.write(f"**{summary}**")
            st.caption(f"Yapılandırılmış işlem: `{query.operation}`")
            st.plotly_chart(figure, width="stretch")
            st.dataframe(display_frame(chat_result), width="stretch")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
