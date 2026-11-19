"""Data Autopilot AI — Streamlit giriş noktası: veri kaynağı, ortak oturum durumu ve sayfa gezintisi.

Her sayfa `views/` altında yalnızca sunum yapar; iş mantığı `core/` içindedir. Otopilot sonucu, veri imzası
ve önbellekler oturum durumunda tutulur ve tüm sayfalarca paylaşılır (bkz. `views/common.py`).
"""

from __future__ import annotations

import streamlit as st

from core.ingestion import (
    content_signature,
    generate_sample_dirty_data,
    list_local_csv_files,
    read_local_csv,
    read_tabular_file,
)
from views.common import SINGLE_TABLE, init_state, load_dataset

st.set_page_config(page_title="Data Autopilot AI", page_icon="🚀", layout="wide")
init_state()

PAGES = [
    st.Page("views/genel_bakis.py", title="Genel Bakış", icon="🚀", default=True),
    st.Page("views/kalite.py", title="Kalite", icon="🧹"),
    st.Page("views/zaman_serisi.py", title="Zaman Serisi", icon="📈"),
    st.Page("views/model.py", title="Model", icon="🤖"),
    st.Page("views/rapor.py", title="Rapor", icon="📄"),
]
navigation = st.navigation(PAGES)

with st.sidebar:
    st.header("Veri Kaynağı")
    st.text_input(
        "Gemini API Key (isteğe bağlı)",
        type="password",
        key="api_key",
        help="Boş bırakılırsa KVKK maskeli metadata üzerinden güvenli yerel fallback kullanılır.",
    )
    work_mode = st.radio("Çalışma modu", [SINGLE_TABLE, "Çoklu tablo (data/ klasörü)"], key="work_mode")
    files = list_local_csv_files()
    selected, uploaded = "-- Seçin --", None
    if work_mode == SINGLE_TABLE:
        selected = st.selectbox("Yerel CSV seçin", ["-- Seçin --", *[p.name for p in files]])
        uploaded = st.file_uploader("CSV veya Excel (.xlsx) yükleyin", type=["csv", "xlsx"])
    else:
        st.caption("Tablolar arası ilişkiler Genel Bakış sayfasında keşfedilir; onayladığınız plan tek tabloya birleştirilir.")
    if work_mode == SINGLE_TABLE and st.button("Örnek Kirli Türkçe Veri Üret ve Yükle", width="stretch"):
        path = generate_sample_dirty_data()
        load_dataset(lambda: read_local_csv(path), f"sample:{path.stat().st_mtime_ns}", path.name)
        st.rerun()

if uploaded is not None:
    load_dataset(
        # Önbellek anahtarı içerikten türetilir: aynı ad ve boyutta farklı dosya eski sonuçları göstermez.
        lambda: read_tabular_file(uploaded, uploaded.name), f"upload:{content_signature(uploaded)}", uploaded.name
    )
elif selected != "-- Seçin --":
    path = next(path for path in files if path.name == selected)
    file_stat = path.stat()
    load_dataset(
        lambda: read_local_csv(path), f"local:{path}:{file_stat.st_mtime_ns}:{file_stat.st_size}", path.name
    )

st.title("🚀 Data Autopilot AI")
st.caption("Sıfır tıklamalı, güvenli ve açıklanabilir otonom veri bilimi")
if st.session_state.get("load_error"):
    st.error(f"Dosya yüklenemedi: {st.session_state.load_error}")

navigation.run()
