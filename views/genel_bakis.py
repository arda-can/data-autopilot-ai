"""Genel Bakış: çoklu tablo birleştirme, yönetici özeti, seçilen grafikler ve veriyle sohbet."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.chat import DataChatEngine
from core.formatting import decimal_tr, display_frame, integer_tr
from core.ingestion import frame_signature, list_local_csv_files
from core.relations import apply_plan, default_base, prune_plan, relation_figure, suggest_plan
from views.common import (
    SINGLE_TABLE, discover_tables, joined_active, require_result, rule_evaluation, set_dataset,
)


def _multi_table_section() -> None:
    files = list_local_csv_files()
    with st.expander("🔗 Çoklu Tablo Keşfi ve Birleştirme", expanded=not joined_active()):
        if len(files) < 2:
            st.info("Çoklu tablo için data/ klasöründe en az iki CSV gerekir.")
            return
        tables, table_infos, relations = discover_tables(
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
        if not relations:
            st.info("Tablolar arasında ortak anahtar bulunamadı.")
            return
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
            return
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
                # İmza birleşik verinin içeriğinden: plan veya kaynak dosyalar değişirse önbellek yenilenir.
                set_dataset(joined.frame, f"join:{frame_signature(joined.frame)}", f"birlesik_{join_base}.csv")
                st.session_state.join_logs = joined.logs
                st.rerun()
    if joined_active() and st.session_state.get("join_logs"):
        with st.expander("Birleştirme kaydı"):
            st.dataframe(
                display_frame(pd.DataFrame([
                    {"Tablo": log.table, "Tür": log.mode, "Anahtar": log.key, "Satır (önce)": log.rows_before,
                     "Satır (sonra)": log.rows_after, "Eşleşme oranı": log.match_rate, "Not": log.note}
                    for log in st.session_state.join_logs
                ])),
                width="stretch", hide_index=True,
            )
    if not joined_active():
        st.stop()


if st.session_state.get("work_mode", SINGLE_TABLE) != SINGLE_TABLE:
    _multi_table_section()

result = require_result()
before, after, cleaned = result.before_report, result.after_report, result.cleaned_frame

st.header("🚀 Otopilot Yönetici Özeti")
rule_report, _ = rule_evaluation(result)
metrics = st.columns(5)
metrics[0].metric(
    "Yapısal Hijyen",
    f"{decimal_tr(after.kalite_skoru)}/100",
    delta=decimal_tr(after.kalite_skoru - before.kalite_skoru),
    help="Boş/sabit sütun, mükerrer ve tip sorunlarını ölçer; değerlerin doğruluğunu ölçmez.",
)
metrics[1].metric(
    "İş Kuralı Uyumu",
    "—" if rule_report is None or rule_report.compliance_score is None
    else f"%{decimal_tr(rule_report.compliance_score)}",
    help="Hiçbir kuralı ihlal etmeyen kayıtların oranı (Kalite sayfasındaki kurallar).",
)
metrics[2].metric("Satır Değişimi", f"{before.satır_sayısı} → {after.satır_sayısı}")
metrics[3].metric("Sütun Değişimi", f"{before.sütun_sayısı} → {after.sütun_sayısı}")
metrics[4].metric("Otomatik İşlem", len(result.audit_logs))

st.write("**Başlangıçta tespit edilen veri sorunları**")
issues = before.sorunlar if before.sorunlar != ["Belirgin bir kalite sorunu tespit edilmedi."] else [
    "Belirgin bir başlangıç kalite sorunu tespit edilmedi; standart güvenlik kontrolleri uygulandı."
]
for issue in issues:
    st.write(f"• {issue}")
finding_count = (
    len(after.domain_findings) + len(after.category_variants) + len(result.outliers) + len(result.rejected_actions)
)
if finding_count:
    st.caption(
        f"Kalite sayfasında {finding_count} bulgu var (alan kuralı, kategori kopyası, aykırı değer, "
        "uygulanmayan plan önerisi); değerler değiştirilmedi."
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

st.header("💬 Veriyle Sohbet")
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
        chat_engine = DataChatEngine(st.session_state.get("api_key", ""))
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

with st.expander("Temizlenmiş veri önizlemesini göster"):
    st.dataframe(display_frame(cleaned.head(100)), width="stretch")
