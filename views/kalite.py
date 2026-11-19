"""Kalite: temizlik planı, alan kuralı ve kategori bulguları, aykırı değerler, audit log ve KVKK kartı."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core.formatting import decimal_tr, display_frame
from views.common import require_result

result = require_result()
before, after, cleaned = result.before_report, result.after_report, result.cleaned_frame

st.header("🧹 Veri Kalitesi ve Yapısal Hijyen")
metrics = st.columns(3)
metrics[0].metric(
    "Kalite Skoru",
    f"{decimal_tr(after.kalite_skoru)}/100",
    delta=decimal_tr(after.kalite_skoru - before.kalite_skoru),
)
metrics[1].metric("Satır Değişimi", f"{before.satır_sayısı} → {after.satır_sayısı}")
metrics[2].metric("Otomatik İşlem", len(result.audit_logs))
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
            display_frame(pd.DataFrame([item.model_dump() for item in result.outliers]).rename(columns={
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

st.header("🔍 Detaylar")
with st.expander("Audit Log'u göster"):
    st.dataframe(
        display_frame(pd.DataFrame([item.model_dump() for item in result.audit_logs])),
        width="stretch", hide_index=True,
    )
with st.expander("KVKK maskeli Metadata Kartını göster"):
    st.json(result.metadata.model_dump())
