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
    content_signature,
    frame_signature,
)
from core.formatting import decimal_tr, display_frame, integer_tr
from core.report import build_report, render_html, render_pdf
from core.timeseries import TimeSeriesEngine, date_columns, value_columns
from core.drift import DriftEngine, prepare_for_comparison, split_by_date, split_by_years
from core.explain import PDP_NOTE, ModelExplainer
from core.relations import (
    JoinPlan, RelationFinder, apply_plan, default_base, load_tables, prune_plan, relation_figure, suggest_plan,
)


def _model_dict(value: BaseModel) -> dict:
    return value.model_dump()


def _load_dataset(load, signature: str, name: str) -> None:
    """Veriyi yükler; okuma hatasında önceki veri korunur ve Türkçe hata gösterilir (aynı dosya tekrar denenmez)."""
    if signature in (st.session_state.data_signature, st.session_state.get("failed_signature")):
        return
    try:
        frame = load()
    except ValueError as exc:
        st.session_state.failed_signature = signature
        st.session_state.load_error = str(exc)
        return
    st.session_state.failed_signature = None
    st.session_state.load_error = None
    st.session_state.dataframe = frame
    st.session_state.data_signature = signature
    st.session_state.data_name = name
    st.session_state.autopilot_result = None


@st.cache_resource(show_spinner="Tablolar okunuyor ve ilişkiler aranıyor...")
def _discover_tables(signature: tuple[tuple[str, int, int], ...]) -> tuple:
    """`data/` klasöründeki tabloları bir kez okur; dosyalar değişince imza değişir ve yeniden okunur."""
    tables = load_tables([Path(item[0]) for item in signature])
    finder = RelationFinder()
    return tables, finder.describe(tables), finder.discover(tables)


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
    work_mode = st.radio("Çalışma modu", ["Tek tablo", "Çoklu tablo (data/ klasörü)"], key="work_mode")
    files = list_local_csv_files()
    selected, uploaded = "-- Seçin --", None
    if work_mode == "Tek tablo":
        selected = st.selectbox("Yerel CSV seçin", ["-- Seçin --", *[p.name for p in files]])
        uploaded = st.file_uploader("CSV veya Excel (.xlsx) yükleyin", type=["csv", "xlsx"])
    else:
        st.caption("Tablolar arası ilişkiler ana ekranda keşfedilir; onayladığınız plan tek tabloya birleştirilir.")
    if work_mode == "Tek tablo" and st.button("Örnek Kirli Türkçe Veri Üret ve Yükle", width="stretch"):
        path = generate_sample_dirty_data()
        _load_dataset(lambda: read_local_csv(path), f"sample:{path.stat().st_mtime_ns}", path.name)
        st.rerun()

if uploaded is not None:
    _load_dataset(
        # Önbellek anahtarı içerikten türetilir: aynı ad ve boyutta farklı dosya eski sonuçları göstermez.
        lambda: read_tabular_file(uploaded, uploaded.name), f"upload:{content_signature(uploaded)}", uploaded.name
    )
elif selected != "-- Seçin --":
    path = next(path for path in files if path.name == selected)
    file_stat = path.stat()
    _load_dataset(
        lambda: read_local_csv(path), f"local:{path}:{file_stat.st_mtime_ns}:{file_stat.st_size}", path.name
    )
if st.session_state.get("load_error"):
    st.error(f"Dosya yüklenemedi: {st.session_state.load_error}")

if work_mode != "Tek tablo":
    joined_active = str(st.session_state.data_signature or "").startswith("join:")
    with st.expander("🔗 Çoklu Tablo Keşfi ve Birleştirme", expanded=not joined_active):
        if len(files) < 2:
            st.info("Çoklu tablo için data/ klasöründe en az iki CSV gerekir.")
        else:
            tables, table_infos, relations = _discover_tables(
                tuple((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in files)
            )
            st.dataframe(
                pd.DataFrame([
                    {"Tablo": info.name, "Satır": info.rows, "Sütun": info.columns,
                     "Benzersiz anahtarlar": ", ".join(info.unique_keys) or "—"}
                    for info in table_infos
                ]),
                width="stretch", hide_index=True,
            )
            if relations:
                st.plotly_chart(relation_figure(table_infos, relations), width="stretch")
                st.dataframe(
                    display_frame(pd.DataFrame([
                        {"Tür": item.kind, "Çok taraf": f"{item.left_table}.{item.left_column}",
                         "Bir taraf": f"{item.right_table}.{item.right_column}",
                         "Kapsama": item.containment, "Ortak değer": item.shared_values,
                         "Öneri": "evet" if item.suggested else "hayır", "Not": item.note}
                        for item in relations
                    ])),
                    width="stretch", hide_index=True,
                )
                table_names = list(tables)
                join_base = st.selectbox(
                    "Taban tablo (birleşik verinin satır düzeyi bu tablonun düzeyi olur)", table_names,
                    index=table_names.index(default_base(relations, table_names)), key="join_base",
                )
                join_plan = suggest_plan(join_base, relations, tables)
                if not join_plan.steps:
                    st.info("Bu tablodan güvenli biçimde birleştirilebilecek bir tablo bulunamadı.")
                else:
                    st.write("**Önerilen birleştirme planı** — onaylamadığınız adımların işaretini kaldırın:")
                    st.caption(
                        "Eşleştir: karşı tabloda anahtar benzersiz, satır sayısı değişmez. "
                        "Topla: karşı tabloda anahtar başına birden çok satır var; önce anahtar başına "
                        "sayım/ortalama/toplam alınır. Çoktan-çoğa ilişkiler satırları çoğaltacağı için önerilmez."
                    )
                    approved_tables = {
                        step.table for step in join_plan.steps
                        if st.checkbox(
                            f"{step.table} — {step.mode}: {step.left_table}.{step.left_column} = "
                            f"{step.table}.{step.right_column}",
                            value=True, key=f"join_step_{join_base}_{step.table}",
                        )
                    }
                    valid_plan, dropped_steps = prune_plan(join_plan, approved_tables)
                    for reason in dropped_steps:
                        st.caption(reason)
                    if st.button("Birleştir ve analiz et", key="join_button", disabled=not valid_plan.steps):
                        try:
                            with st.spinner("Tablolar birleştiriliyor..."):
                                joined = apply_plan(tables, valid_plan)
                        except ValueError as exc:
                            st.error(f"Birleştirme yapılamadı: {exc}")
                        else:
                            st.session_state.dataframe = joined.frame
                            # İmza birleşik verinin içeriğinden: plan veya kaynak dosyalar değişirse önbellek yenilenir.
                            st.session_state.data_signature = f"join:{frame_signature(joined.frame)}"
                            st.session_state.data_name = f"birlesik_{join_base}.csv"
                            st.session_state.join_logs = joined.logs
                            st.session_state.autopilot_result = None
                            st.rerun()
            else:
                st.info("Tablolar arasında ortak anahtar bulunamadı.")
    if joined_active and st.session_state.get("join_logs"):
        with st.expander("Birleştirme kaydı"):
            st.dataframe(
                display_frame(pd.DataFrame([
                    {"Tablo": log.table, "Tür": log.mode, "Anahtar": log.key, "Satır (önce)": log.rows_before,
                     "Satır (sonra)": log.rows_after, "Eşleşme oranı": log.match_rate, "Not": log.note}
                    for log in st.session_state.join_logs
                ])),
                width="stretch", hide_index=True,
            )
    if not joined_active:
        st.stop()

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
if result.plan_origin is not None:
    plan_text = f"Temizlik planı: {result.plan_origin.source} — {result.plan_origin.reason}"
    (st.warning if result.plan_origin.error else st.caption)(plan_text)
for rejected in result.rejected_actions:
    st.warning(
        f"Plan önerisi uygulanmadı — “{rejected.column}” için {rejected.action}: {rejected.rejection} "
        f"(Plan gerekçesi: {rejected.plan_reason})"
    )
if result.outliers:
    with st.expander(f"Şüpheli aykırı değerler: {len(result.outliers)} sütun (değiştirilmedi)"):
        st.dataframe(
            display_frame(pd.DataFrame([_model_dict(item) for item in result.outliers]).rename(columns={
                "column": "Sütun", "count": "Aykırı Sayısı", "ratio": "Oran",
                "lower_bound": "Alt Sınır", "upper_bound": "Üst Sınır",
                "min": "En Küçük", "max": "En Büyük", "method": "Yöntem",
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

timeseries = result.timeseries
ts_request: tuple | None = None
timeseries_error = result.timeseries_message
ts_dates = date_columns(cleaned)
if ts_dates:
    st.header("📈 Zaman Serisi Otopilotu")
    default_date = timeseries.summary.date_column if timeseries is not None else ts_dates[0]
    metric_options = {"Kayıt sayısı": None, **{f"{column} toplamı": column for column in value_columns(cleaned)}}
    frequency_options = {"Otomatik": "auto", "Günlük": "D", "Haftalık": "W", "Aylık": "M"}
    date_column_ui, metric_ui, frequency_ui = st.columns(3)
    ts_date = date_column_ui.selectbox("Tarih sütunu", ts_dates, index=ts_dates.index(default_date), key="ts_date")
    ts_metric = metric_ui.selectbox("Ölçü", list(metric_options), key="ts_metric")
    ts_frequency = frequency_ui.selectbox("Frekans", list(frequency_options), key="ts_frequency")
    ts_request = (ts_date, metric_options[ts_metric], frequency_options[ts_frequency])
    if ts_request != (default_date, None, "auto"):
        # Kullanıcının seçimi ayrı hesaplanır ve önbelleğe alınır; her yeniden çizimde tekrarlanmaz.
        ts_cache = st.session_state.setdefault("timeseries_cache", {})
        ts_key = (st.session_state.data_signature, *ts_request)
        if ts_key not in ts_cache:
            try:
                ts_cache[ts_key] = TimeSeriesEngine().analyze(
                    cleaned, date_column=ts_request[0], value_column=ts_request[1], frequency=ts_request[2]
                )
            except ValueError as exc:
                ts_cache[ts_key] = f"Zaman serisi analizi yapılamadı: {exc}"
        cached_ts = ts_cache[ts_key]
        timeseries, timeseries_error = (None, cached_ts) if isinstance(cached_ts, str) else (cached_ts, "")
    if timeseries is None:
        st.warning(timeseries_error or "Zaman serisi analizi yapılamadı.")
    else:
        ts_summary = timeseries.summary
        ts_metrics = st.columns(4)
        ts_metrics[0].metric("Frekans", ts_summary.frequency_label.capitalize())
        ts_metrics[1].metric("Dönem", integer_tr(ts_summary.periods))
        ts_metrics[2].metric(
            "Seviye Değişimi",
            "—" if ts_summary.level_change is None else f"%{decimal_tr(ts_summary.level_change * 100)}",
        )
        ts_metrics[3].metric(
            "Mevsimsellik Gücü",
            "—" if ts_summary.seasonality_strength is None else decimal_tr(ts_summary.seasonality_strength, 2),
        )
        st.write(ts_summary.trend_text)
        st.write(ts_summary.seasonality_text)
        (st.success if ts_summary.beats_baseline else st.warning)(ts_summary.verdict)
        st.plotly_chart(timeseries.forecast_figure, width="stretch")
        st.caption(f"{ts_summary.forecast_text} Model: {ts_summary.model_name}.")
        if timeseries.seasonal_figure is not None:
            st.plotly_chart(timeseries.seasonal_figure, width="stretch")
        if ts_summary.anomalies:
            st.subheader("Olağan dışı dönemler")
            st.dataframe(
                display_frame(pd.DataFrame([
                    {"Dönem": item.label, "Gözlenen": item.observed, "Beklenen": item.expected,
                     "Sapma (z)": item.z, "Yön": item.direction}
                    for item in ts_summary.anomalies
                ])),
                width="stretch",
                hide_index=True,
            )
        for note in ts_summary.notes:
            st.caption(f"Not: {note}")

st.header("🔀 Dönem Karşılaştırması")
st.caption(
    "İki dönem ya da iki dosya arasında şema, eksik oranı ve dağılım farkı. "
    "Trafik ışığı etki büyüklüğüne (PSI) göre verilir; ham satırlar karşılaştırılmaz."
)
drift_modes = (["İki yıl", "Kesim tarihi"] if ts_dates else []) + ["İkinci dosya"]
drift_mode = st.radio("Karşılaştırma türü", drift_modes, horizontal=True, key="drift_mode")
drift_request: tuple | None = None
drift_upload = None
if drift_mode in ("İki yıl", "Kesim tarihi"):
    drift_date = st.selectbox("Tarih sütunu", ts_dates, key="drift_date")
    drift_dates = cleaned[drift_date].dropna()
    if drift_mode == "İki yıl":
        drift_years = sorted(int(year) for year in drift_dates.dt.year.unique())
        if len(drift_years) < 2:
            st.info("Bu tarih sütununda iki farklı yıl yok; kesim tarihiyle karşılaştırın.")
        else:
            reference_ui, current_ui = st.columns(2)
            reference_year = reference_ui.selectbox(
                "Referans yıl", drift_years, index=len(drift_years) - 2, key="drift_reference_year"
            )
            current_year = current_ui.selectbox(
                "Güncel yıl", drift_years, index=len(drift_years) - 1, key="drift_current_year"
            )
            drift_request = ("years", drift_date, reference_year, current_year)
    else:
        ordered_dates = drift_dates.sort_values()
        drift_cut = st.date_input(
            "Kesim tarihi (bu tarih ve sonrası güncel dönem)",
            value=ordered_dates.iloc[len(ordered_dates) // 2].date(),
            min_value=ordered_dates.iloc[0].date(),
            max_value=ordered_dates.iloc[-1].date(),
            key="drift_cut",
        )
        drift_request = ("cut", drift_date, str(drift_cut))
else:
    drift_upload = st.file_uploader(
        "Karşılaştırılacak dosya (CSV veya Excel)", type=["csv", "xlsx"], key="drift_file"
    )
    if drift_upload is not None:
        drift_request = ("file", drift_upload.name, content_signature(drift_upload))

if drift_request is not None and st.button("Karşılaştır", key="drift_button"):
    with st.spinner("Dönemler karşılaştırılıyor..."):
        try:
            if drift_request[0] == "years":
                drift_parts = split_by_years(cleaned, drift_request[1], drift_request[2], drift_request[3])
            elif drift_request[0] == "cut":
                drift_parts = split_by_date(cleaned, drift_request[1], drift_request[2])
            else:
                # İki dosya da aynı yapısal hijyenden geçer; istatistiksel dönüşüm yapılmaz.
                drift_parts = (
                    prepare_for_comparison(st.session_state.dataframe),
                    prepare_for_comparison(read_tabular_file(drift_upload, drift_upload.name)),
                    st.session_state.data_name,
                    drift_upload.name,
                )
            drift_outcome = DriftEngine().compare(*drift_parts)
        except ValueError as exc:
            drift_outcome = f"Karşılaştırma yapılamadı: {exc}"
        st.session_state.drift_result = (st.session_state.data_signature, drift_request, drift_outcome)

drift_stored = st.session_state.get("drift_result")
if drift_stored and drift_stored[0] == st.session_state.data_signature and drift_stored[1] == drift_request:
    drift_outcome = drift_stored[2]
    if isinstance(drift_outcome, str):
        st.warning(drift_outcome)
    else:
        drift_report = drift_outcome.report
        st.write(drift_report.summary)
        light_columns = st.columns(3)
        light_columns[0].metric("🔴 Kırmızı", drift_report.counts["kırmızı"])
        light_columns[1].metric("🟡 Sarı", drift_report.counts["sarı"])
        light_columns[2].metric("🟢 Yeşil", drift_report.counts["yeşil"])
        for change in drift_report.schema_changes:
            st.warning(f"Şema: “{change.column}” {change.change} ({change.detail})")
        if drift_report.columns:
            lights = {"kırmızı": "🔴", "sarı": "🟡", "yeşil": "🟢"}
            st.dataframe(
                display_frame(pd.DataFrame([
                    {
                        "Durum": f"{lights[item.status]} {item.status}", "Sütun": item.column, "Tür": item.kind,
                        "PSI": item.psi, drift_report.reference_label: item.reference_summary,
                        drift_report.current_label: item.current_summary, "Açıklama": item.explanation,
                        "Test": item.test, "p-değeri": item.p_value,
                    }
                    for item in drift_report.columns
                ])),
                width="stretch",
                hide_index=True,
            )
        if drift_outcome.figure is not None:
            st.plotly_chart(drift_outcome.figure, width="stretch")
        if drift_outcome.detail_figures:
            with st.expander("Değişen sütunların dağılımları"):
                for drift_figure in drift_outcome.detail_figures.values():
                    st.plotly_chart(drift_figure, width="stretch")
        if drift_report.skipped:
            with st.expander(f"Karşılaştırılmayan sütunlar ({len(drift_report.skipped)})"):
                for note in drift_report.skipped:
                    st.caption(note)
        st.caption(drift_report.method_note)

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
automl_settings: tuple = (None, (), ())
report_explanation = None
if selected_target != "-- Hedef seçin --":
    with st.expander("Değerlendirme ve tahmin anı ayarları"):
        st.caption(
            "Model, eğitim bölmesinde çapraz doğrulamayla seçilir; test bölmesi seçime katılmaz. Tahmin anında "
            "hangi bilgilerin bilindiğini belirterek sızıntıyı önleyin ve gerekirse zamana göre değerlendirin."
        )
        time_options = ["Yok (rastgele bölme)", *date_columns(cleaned)]
        evaluation_time = st.selectbox(
            "Zaman sırası sütunu (test = en son %20 dönem)", time_options, key=f"automl_time_{selected_target}"
        )
        feature_options = [column for column in target_options if column != selected_target]
        excluded_columns = st.multiselect(
            "Tahmin anında bilinmeyen sütunlar (modelden çıkarılır)", feature_options,
            key=f"automl_exclude_{selected_target}",
        )
        known_columns = st.multiselect(
            "Tahmin anında bilindiğini onayladığınız sütunlar (sızıntı uyarısı verilmez)", feature_options,
            key=f"automl_known_{selected_target}",
        )
    automl_settings = (
        None if evaluation_time == time_options[0] else evaluation_time,
        tuple(sorted(excluded_columns)), tuple(sorted(known_columns)),
    )
if selected_target != "-- Hedef seçin --" and (
    result.suggested_target is not None and selected_target != result.suggested_target
    or result.suggested_target is None and selected_target != target_options_for_ui[0]
    or automl_settings != (None, (), ())
):
    # Aynı veri, hedef ve ayarlar için model yalnızca bir kez eğitilir; her yeniden çizimde tekrarlanmaz.
    cache = st.session_state.setdefault("automl_cache", {})
    cache_key = (st.session_state.data_signature, selected_target, automl_settings)
    if cache_key not in cache:
        sample = cleaned.sample(10_000, random_state=42) if len(cleaned) > 10_000 else cleaned
        try:
            with st.spinner("Modeller çapraz doğrulamayla karşılaştırılıyor..."):
                cache[cache_key] = AutoMLEngine().run(
                    sample, selected_target, time_column=automl_settings[0],
                    exclude_columns=automl_settings[1], known_columns=automl_settings[2],
                )
        except (ValueError, TypeError, KeyError) as exc:
            cache[cache_key] = f"'{selected_target}' hedefi için AutoML çalıştırılamadı: {exc}"
    cached = cache[cache_key]
    automl, automl_error = (None, cached) if isinstance(cached, str) else (cached, "")
if selected_target != "-- Hedef seçin --":
    st.write(f"**Tahmin edilen hedef sütun:** `{selected_target}`")
if automl is not None:
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
    if automl.model is not None and automl.background is not None:
        st.subheader("Model neden böyle tahmin ediyor? (Kısmi bağımlılık)")
        explain_cache = st.session_state.setdefault("explain_cache", {})
        explain_key = (st.session_state.data_signature, selected_target, automl_settings, automl.best_model)
        if explain_key not in explain_cache:
            try:
                explain_cache[explain_key] = ModelExplainer().explain(automl)
            except ValueError as exc:
                explain_cache[explain_key] = f"Açıklama üretilemedi: {exc}"
        explanation = explain_cache[explain_key]
        if isinstance(explanation, str):
            st.info(explanation)
        else:
            report_explanation = explanation
            st.caption(PDP_NOTE)
            for curve, curve_figure in zip(explanation.curves, explanation.figures):
                st.plotly_chart(curve_figure, width="stretch")
                (st.warning if curve.interpretation.startswith("Uyarı") else st.write)(curve.interpretation)
            for note in explanation.skipped:
                st.caption(note)
            if explanation.what_if_features:
                st.subheader("What-if: değerleri değiştirip tahmini görün")
                st.caption(
                    "Diğer özellikler tipik değerlerinde (sayısalda medyan, kategoride en sık değer) sabit tutulur."
                )
                with st.form(key=f"what_if_{selected_target}"):
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
                                    [{"Sınıf": label, "Olasılık": value}
                                     for label, value in prediction.probabilities.items()]
                                )),
                                width="stretch", hide_index=True,
                            )
else:
    st.info(automl_error or result.automl_message or "Uygun hedef sütun bulunamadığı için AutoML atlandı.")

st.header("📄 Yönetici Raporu")
st.caption(
    "Özet, öneriler, grafikler ve model sonucu tek dosyada. "
    "Ham satırlar ve audit log örnekleri rapora girmez."
)
report_key = (st.session_state.data_signature, selected_target, ts_request, automl_settings, str(drift_request))
if st.button("Raporu Hazırla", key="report_button"):
    with st.spinner("Rapor hazırlanıyor..."):
        ts_figures = [] if timeseries is None else [
            figure for figure in (timeseries.forecast_figure, timeseries.seasonal_figure) if figure is not None
        ]
        stored_drift = st.session_state.get("drift_result")
        report_drift = (
            stored_drift[2].report
            if stored_drift and stored_drift[0] == st.session_state.data_signature
            and stored_drift[1] == drift_request and not isinstance(stored_drift[2], str)
            else None
        )
        report_joins = (
            st.session_state.get("join_logs")
            if str(st.session_state.data_signature or "").startswith("join:") else None
        )
        pdp_figures = list(report_explanation.figures) if report_explanation is not None else []
        report = build_report(
            replace(
                result,
                suggested_target=selected_target if automl is not None else result.suggested_target,
                automl_result=automl,
                automl_message=automl_error or result.automl_message,
                timeseries=timeseries,
                timeseries_message=timeseries_error,
            ),
            st.session_state.data_name,
            drift=report_drift,
            explanation=report_explanation,
            join_logs=report_joins,
        )
        st.session_state.report_files = {
            "key": report_key,
            "html": render_html(
                report, [chart.figure for chart in result.charts], automl.figure if automl is not None else None,
                timeseries_figures=ts_figures, pdp_figures=pdp_figures,
            ).encode("utf-8"),
            "pdf": render_pdf(
                report, [chart.figure for chart in result.charts], timeseries_figures=ts_figures,
                pdp_figures=pdp_figures,
            ),
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
    st.caption(
        "Koşul ve zaman kırılımı da sorulabilir: “İstanbul için ortalama harcama”, "
        "“2018'de aylık sipariş sayısı”, “yaşı 40'tan büyük müşterilerin şehirlere göre harcaması”."
    )
    if st.button("Soruyu Analiz Et", key="autopilot_chat_button"):
        try:
            chat_engine = DataChatEngine(api_key)
            chat_result, summary, figure, query = chat_engine.answer(cleaned, question, result.metadata)
            st.write(f"**{summary}**")
            st.caption(f"Yapılandırılmış işlem: `{query.operation}` · Sorgu planı: {chat_engine.origin.source}")
            if chat_engine.origin.error:
                st.warning(chat_engine.origin.reason)
            st.plotly_chart(figure, width="stretch")
            st.dataframe(display_frame(chat_result), width="stretch")
            query_code = chat_engine.explain(cleaned, query)
            st.markdown("**Bu cevap nasıl hesaplandı?**")
            st.caption(
                f"{integer_tr(query_code.rows_used)} kayıt kullanıldı"
                + (f"; filtre: {'; '.join(query_code.filters)}." if query_code.filters else ".")
                + " Kod yalnızca gösterim içindir: uygulama bu metinleri çalıştırmaz, aynı hesabı "
                "doğrulanmış işlemlerle yapar."
            )
            pandas_tab, sql_tab = st.tabs(["pandas", "SQL (PostgreSQL)"])
            pandas_tab.code(query_code.pandas, language="python")
            sql_tab.code(query_code.sql, language="sql")
        except (ValueError, TypeError) as exc:
            st.error(str(exc))
