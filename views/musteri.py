"""Müşteri: RFM segmentasyonu, kohort tutundurma ve tarihsel müşteri değeri."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from core.columns import is_text_series
from core.customers import DEFINITIONS, WINDOW_OPTIONS
from core.formatting import decimal_tr, display_frame, integer_tr
from views.common import (
    customer_analysis_for, customer_request, default_customer_request, index_of, remember, require_result,
)

NONE = "— yok —"

result = require_result()
cleaned = result.cleaned_frame

st.header("👥 Müşteri Analizi")
st.caption(
    "RFM segmentleri, kohort tutundurma ve edinimden sonraki tarihsel müşteri değeri. "
    "Tanımlar aşağıda açıkça yazılıdır; hiçbir değer tahmin edilmez ya da doldurulmaz."
)

text_columns = [str(column) for column in cleaned.columns if is_text_series(cleaned[column])]
date_columns = [str(column) for column in cleaned.columns if pd.api.types.is_datetime64_any_dtype(cleaned[column])]
number_columns = [
    str(column) for column in cleaned.columns
    if pd.api.types.is_numeric_dtype(cleaned[column]) and not pd.api.types.is_bool_dtype(cleaned[column])
]
if not (text_columns and date_columns and number_columns):
    st.info("Müşteri analizi için bir müşteri kimliği (metin), bir tarih ve bir tutar (sayı) sütunu gerekir.")
    st.stop()

current = customer_request(result) or (None, None, None, None, "Tüm veri", None)
if default_customer_request(result) is None:
    st.info("Müşteri, tarih ve tutar sütunları otomatik bulunamadı; aşağıdan seçin.")
group_options = [
    str(column) for column in text_columns
    if column != current[0] and 1 < cleaned[column].nunique() <= 20
]
with st.expander("Sütunlar ve pencere", expanded=default_customer_request(result) is None):
    left, middle, right = st.columns(3)
    customer = left.selectbox("Müşteri kimliği", text_columns, index=index_of(text_columns, current[0]),
                              key="cust_customer")
    date = middle.selectbox("İşlem tarihi", date_columns, index=index_of(date_columns, current[1]), key="cust_date")
    amount = right.selectbox("Tutar", number_columns, index=index_of(number_columns, current[2]), key="cust_amount")
    left, middle, right = st.columns(3)
    order_options = [NONE, *[column for column in text_columns if column != customer]]
    order = left.selectbox("Sipariş numarası (isteğe bağlı)", order_options,
                           index=index_of(order_options, current[3]), key="cust_order")
    windows = list(WINDOW_OPTIONS)
    window = middle.selectbox("RFM penceresi", windows, index=index_of(windows, current[4]), key="cust_window")
    group_choices = [NONE, *group_options]
    group = right.selectbox("Değer kırılımı (ilk işlemdeki değer)", group_choices,
                            index=index_of(group_choices, current[5]), key="cust_group")
request = (customer, date, amount, None if order == NONE else order, window, None if group == NONE else group)
remember("customer_request", request)

analysis = customer_analysis_for(result, request)
if isinstance(analysis, str):
    st.warning(analysis)
    st.stop()
summary = analysis.summary

with st.expander("Tanımlar ve pencere"):
    st.write(f"**Analiz tarihi:** {summary.analysis_date} · **RFM penceresi:** {summary.window_label} "
             f"({summary.window_start} – {summary.analysis_date})")
    for name, text in (("İşlem", DEFINITIONS["işlem"]), ("RFM", DEFINITIONS["rfm"]),
                       ("Kohort", DEFINITIONS["kohort"]), ("Müşteri değeri", DEFINITIONS["clv"])):
        st.caption(f"**{name}:** {text}")
for note in summary.notes:
    st.caption(f"Not: {note}")

year_value = next((row for row in summary.clv if row.horizon_months == 12 and row.customers), None)
metrics = st.columns(4)
metrics[0].metric("Müşteri", integer_tr(summary.customers))
metrics[1].metric("İşlem", integer_tr(summary.transactions))
metrics[2].metric("Tekrar Eden Müşteri", f"%{decimal_tr(summary.repeat_customer_share * 100)}")
metrics[3].metric(
    "12 Aylık Müşteri Değeri", "—" if year_value is None else decimal_tr(year_value.revenue_per_customer, 0),
    help="Edinimden sonraki ilk 12 ayda müşteri başına ortalama gelir (tarihsel, yalnızca 12 ayı gözlenmiş müşteriler).",
)

st.subheader("RFM segmentleri")
st.plotly_chart(analysis.segment_figure, width="stretch")
st.dataframe(
    display_frame(pd.DataFrame([
        {"Segment": item.segment, "Müşteri": item.customers, "Müşteri payı": item.customer_share,
         "Gelir payı": item.revenue_share, "Ort. yakınlık (gün)": item.recency_days,
         "Ort. sıklık": item.frequency, "Ort. parasal": item.monetary, "Önerilen aksiyon": item.action}
        for item in summary.segments
    ])),
    width="stretch", hide_index=True,
)
buffer = io.BytesIO()
analysis.rfm_table.to_csv(buffer, index=False, encoding="utf-8-sig")
st.download_button("Müşteri bazında RFM tablosunu indir (CSV)", buffer.getvalue(), "rfm_musteriler.csv", "text/csv")

st.subheader("Kohort tutundurma")
st.plotly_chart(analysis.cohort_figure, width="stretch")
if summary.retention:
    st.write("Kohort büyüklüğüyle ağırlıklı ortalama tutundurma: " + ", ".join(
        f"{month}. ay %{decimal_tr(value * 100)}" for month, value in summary.retention.items()
    ) + ".")

st.subheader("Tarihsel müşteri değeri")
st.dataframe(
    display_frame(pd.DataFrame([
        {"Edinimden sonraki ay": row.horizon_months, "Gözlenen müşteri": row.customers,
         "Müşteri başına gelir": row.revenue_per_customer, "Müşteri başına işlem": row.orders_per_customer}
        for row in summary.clv
    ])),
    width="stretch", hide_index=True,
)
st.plotly_chart(analysis.clv_figure, width="stretch")
if analysis.clv_by_group is not None and not analysis.clv_by_group.empty:
    st.write(f"**İlk işlemdeki “{request[5]}” değerine göre** (en az 30 müşterili gruplar):")
    st.dataframe(display_frame(analysis.clv_by_group), width="stretch", hide_index=True)
