"""Uçtan uca arayüz testleri (Streamlit AppTest): elle yapılan Olist denemelerinin tekrarlanabilir karşılığı.

Veri, `AUTOPILOT_DATA_DIR` ile geçici bir klasöre üretilen küçük, Olist benzeri sentetik tablolardır:
siparişler (iki yıl, teslim edilmeyenlerde teslim tarihi boş), müşteriler ve gecikmeyle düşen yorum puanları.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT

TIMEOUT = 240


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(21)
    folder = tmp_path_factory.mktemp("veri")
    orders_count = 700
    purchase = pd.Timestamp("2017-01-02") + pd.to_timedelta(np.sort(rng.integers(0, 720, orders_count)), unit="D")
    status = rng.choice(["delivered", "shipped", "canceled"], orders_count, p=[0.9, 0.06, 0.04])
    estimated = purchase + pd.Timedelta(days=15)
    delay = rng.integers(-8, 12, orders_count)
    delivered = pd.Series(estimated + pd.to_timedelta(delay, unit="D"))
    delivered[status != "delivered"] = pd.NaT
    orders = pd.DataFrame({
        "order_id": [f"o{index:05d}" for index in range(orders_count)],
        "customer_id": [f"c{index % 500:04d}" for index in range(orders_count)],
        "order_status": status,
        "order_purchase_timestamp": purchase.strftime("%Y-%m-%d %H:%M:%S"),
        "order_estimated_delivery_date": estimated.strftime("%Y-%m-%d"),
        "order_delivered_customer_date": delivered.dt.strftime("%Y-%m-%d"),
        "price": np.round(rng.uniform(20, 200, orders_count), 2),
    })
    customers = pd.DataFrame({
        "customer_id": [f"c{index:04d}" for index in range(500)],
        "customer_city": rng.choice(["sao paulo", "rio", "curitiba", "recife"], 500),
    })
    score = np.clip(np.round(4.6 - 0.25 * np.clip(delay, 0, None) + rng.normal(0, 0.5, orders_count)), 1, 5)
    reviews = pd.DataFrame({"review_id": [f"r{index}" for index in range(orders_count)],
                            "order_id": orders["order_id"], "review_score": score.astype(int)})
    # Olist'teki gibi bazı siparişlerin ikinci yorumu var: ilişki 1-N olur ve "topla" önerilir.
    extra = reviews.head(20).assign(review_id=lambda frame: frame["review_id"] + "b")
    reviews = pd.concat([reviews, extra], ignore_index=True)
    orders.to_csv(folder / "olist_orders_dataset.csv", index=False)
    customers.to_csv(folder / "olist_customers_dataset.csv", index=False)
    reviews.to_csv(folder / "olist_order_reviews_dataset.csv", index=False)
    return folder


@pytest.fixture
def app(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(data_dir))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    return application


def _ok(application: AppTest) -> None:
    assert not application.exception, [str(item.value)[:300] for item in application.exception]


GENEL, KALITE, ZAMAN, MODEL, RAPOR = (
    "views/genel_bakis.py", "views/kalite.py", "views/zaman_serisi.py", "views/model.py", "views/rapor.py",
)


def test_single_table_flow_timeseries_drift_chat_and_report(app: AppTest) -> None:
    app.sidebar.selectbox[0].select("olist_orders_dataset.csv").run()
    _ok(app)
    assert "🚀 Otopilot Yönetici Özeti" in [item.value for item in app.header]

    app.switch_page(ZAMAN).run()
    _ok(app)
    headers = [item.value for item in app.header]
    assert "📈 Zaman Serisi Otopilotu" in headers and "🔀 Dönem Karşılaştırması" in headers

    app.selectbox(key="ts_frequency").select("Haftalık").run()
    _ok(app)
    assert ("Frekans", "Haftalık") in [(item.label, item.value) for item in app.metric]

    assert app.selectbox(key="drift_reference_year").value == 2017
    app.button(key="drift_button").click().run()
    _ok(app)
    assert any(item.value.startswith("“2017” ile “2018” arasında") for item in app.markdown)

    app.switch_page(GENEL).run()
    app.text_input(key="autopilot_chat_question").set_value("canceled siparişlerin aylık sayısı").run()
    app.button(key="autopilot_chat_button").click().run()
    _ok(app)
    assert any("Filtre: order_status == canceled" in item.value for item in app.markdown)
    assert any("SELECT DATE_TRUNC('month'" in item.value for item in app.code)

    app.switch_page(RAPOR).run()
    _ok(app)
    app.button(key="report_button").click().run()
    _ok(app)
    page = app.session_state["report_files"]["html"].decode("utf-8")
    for heading in ("Zaman Serisi", "Dönem Karşılaştırması", "Tahmin Modeli", "Uygulanan İşlemler"):
        assert f"<h2>{heading}</h2>" in page
    # Rapor, Zaman Serisi sayfasındaki haftalık seçimi kullanır (sayfa değişse de seçim kaybolmaz).
    assert "haftalık" in page.split("<h2>Zaman Serisi</h2>")[1].split("</section>")[0]
    assert app.session_state["report_files"]["pdf"].startswith(b"%PDF")


def test_selections_survive_page_changes(app: AppTest) -> None:
    """Streamlit, çizilmeyen widget'ın durumunu siler; seçimler başka sayfaya gidip dönünce korunmalı."""
    app.sidebar.selectbox[0].select("olist_orders_dataset.csv").run()
    app.switch_page(ZAMAN).run()
    app.selectbox(key="ts_frequency").select("Aylık").run()
    app.radio(key="drift_mode").set_value("Kesim tarihi").run()
    _ok(app)
    app.switch_page(MODEL).run()
    app.selectbox(key="autopilot_target").select("price").run()
    _ok(app)
    app.switch_page(KALITE).run()
    _ok(app)
    app.switch_page(ZAMAN).run()
    _ok(app)
    assert app.selectbox(key="ts_frequency").value == "Aylık"
    assert app.radio(key="drift_mode").value == "Kesim tarihi"
    app.switch_page(MODEL).run()
    _ok(app)
    assert app.selectbox(key="autopilot_target").value == "price"
    app.switch_page(RAPOR).run()
    _ok(app)
    assert any("model hedefi: price" in item.value for item in app.caption)


def test_prediction_time_settings_and_what_if(app: AppTest) -> None:
    app.sidebar.selectbox[0].select("olist_orders_dataset.csv").run()
    app.switch_page(MODEL).run()
    _ok(app)
    app.selectbox(key="autopilot_target").select("order_status").run()
    _ok(app)
    assert any("olası hedef sızıntısı" in item.value.lower() for item in app.warning)
    app.multiselect(key="automl_known_order_status").set_value(["order_delivered_customer_date"]).run()
    _ok(app)
    assert any("tahmin anında bilindiği onaylanan: order_delivered_customer_date" in item.value for item in app.caption)
    assert not any(item.value.startswith("Olası hedef sızıntısı") for item in app.warning)

    app.selectbox(key="automl_time_order_status").select("order_purchase_timestamp").run()
    _ok(app)
    assert any(item.value.startswith("Değerlendirme: zamana göre (order_purchase_timestamp") for item in app.caption)

    submit = next(button for button in app.button if button.label == "Tahmin et")
    submit.click().run()
    _ok(app)
    assert any(item.value.startswith("Tahmin: “") for item in app.success)


def test_multi_table_join_flow(app: AppTest) -> None:
    app.sidebar.radio(key="work_mode").set_value("Çoklu tablo (data/ klasörü)").run()
    _ok(app)
    assert app.selectbox(key="join_base").value == "orders"
    labels = [item.label for item in app.checkbox]
    assert any(label.startswith("customers — eşleştir") for label in labels)
    assert any(label.startswith("order_reviews — topla") for label in labels)
    app.switch_page(MODEL).run()
    _ok(app)
    assert any("önce Genel Bakış sayfasında" in item.value for item in app.info)
    app.switch_page(GENEL).run()
    app.button(key="join_button").click().run()
    _ok(app)
    assert ("Satır Değişimi", "700 → 700") in [(item.label, item.value) for item in app.metric]
    app.switch_page(MODEL).run()
    _ok(app)
    assert app.selectbox(key="autopilot_target").value == "order_reviews_review_score_ort"
    app.switch_page(RAPOR).run()
    app.button(key="report_button").click().run()
    _ok(app)
    page = app.session_state["report_files"]["html"].decode("utf-8")
    assert "<h2>Veri Birleştirme</h2>" in page and "Model Neden Böyle Tahmin Ediyor?" in page


def test_changed_file_content_refreshes_results(data_dir: Path, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Önbellek anahtarı içerikle değişir: aynı ada yeni içerik yazılınca eski sonuçlar gösterilmez."""
    path = tmp_path / "tablo.csv"
    pd.DataFrame({"şehir": ["A", "B"] * 30, "tutar": range(60)}).to_csv(path, index=False)
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.selectbox[0].select("tablo.csv").run()
    _ok(application)
    assert ("Satır Değişimi", "60 → 60") in [(item.label, item.value) for item in application.metric]
    pd.DataFrame({"şehir": ["A", "B"] * 45, "tutar": range(90)}).to_csv(path, index=False)
    application.run()
    _ok(application)
    assert ("Satır Değişimi", "90 → 90") in [(item.label, item.value) for item in application.metric]


def test_demo_button_loads_demo_dataset(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.button(key="demo_button").click().run()
    _ok(application)
    assert (tmp_path / "demo_eticaret_siparisleri.csv").exists()
    assert "🚀 Otopilot Yönetici Özeti" in [item.value for item in application.header]
    application.switch_page(KALITE).run()
    _ok(application)
    assert any(item.value.startswith("tutar: 4 negatif değer") for item in application.warning)


def test_business_rules_on_quality_page(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.button(key="demo_button").click().run()
    application.switch_page(KALITE).run()
    _ok(application)
    metrics = {item.label: item.value for item in application.metric}
    assert metrics["İş Kuralı Uyumu"] == "%99,5" and metrics["Kural İhlali Olan Kayıt"] == "37"
    assert metrics["Yapısal Hijyen Skoru"].endswith("/100")

    tutar_rule = next(box for box in application.checkbox if box.label.startswith("“tutar” en az 0 olmalı"))
    tutar_rule.uncheck().run()
    _ok(application)
    assert {item.label: item.value for item in application.metric}["Kural İhlali Olan Kayıt"] == "33"

    application.selectbox(key="rule_kind").select("İzinli değerler").run()
    application.selectbox(key="rule_column").select("kanal")
    application.text_area(key="rule_values").input("Web\nMobil Uygulama")
    next(button for button in application.button if button.label == "Kuralı ekle").click().run()
    _ok(application)
    assert any("“kanal” yalnızca şu değerleri alabilir: Web, Mobil Uygulama" in item.value for item in application.markdown)
    violating = int({item.label: item.value for item in application.metric}["Kural İhlali Olan Kayıt"].replace(".", ""))
    assert violating > 1000  # mağaza siparişleri bu kuralı ihlal eder

    application.switch_page(GENEL).run()
    _ok(application)
    assert {item.label: item.value for item in application.metric}["İş Kuralı Uyumu"] != "%99,5"
    application.switch_page(RAPOR).run()
    application.button(key="report_button").click().run()
    _ok(application)
    page = application.session_state["report_files"]["html"].decode("utf-8")
    assert "<h2>İş Kuralı Uyumu</h2>" in page and "“kanal” yalnızca" in page
    assert "“tutar” en az 0 olmalı" not in page  # onayı kaldırılan öneri rapora girmez


def test_customer_page_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.button(key="demo_button").click().run()
    application.switch_page("views/musteri.py").run()
    _ok(application)
    metrics = {item.label: item.value for item in application.metric}
    assert metrics["Müşteri"] == "3.998" and metrics["Tekrar Eden Müşteri"].startswith("%40")
    assert application.selectbox(key="cust_customer").value == "müşteri_no"
    application.selectbox(key="cust_window").select("Son 6 ay").run()
    application.selectbox(key="cust_group").select("kanal").run()
    _ok(application)
    assert any("İlk işlemdeki “kanal” değerine göre" in item.value for item in application.markdown)
    application.switch_page(RAPOR).run()
    _ok(application)
    assert any("müşteri analizi: var" in item.value for item in application.caption)
    application.button(key="report_button").click().run()
    _ok(application)
    page = application.session_state["report_files"]["html"].decode("utf-8")
    assert "<h2>Müşteri Analizi</h2>" in page and "RFM penceresi: Son 6 ay" in page


def test_root_cause_section_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.button(key="demo_button").click().run()
    application.switch_page(ZAMAN).run()
    _ok(application)
    assert "🔎 Neden Değişti?" in [item.value for item in application.header]
    assert str(application.date_input(key="rc_cur_start").value) == "2025-10-01"
    application.selectbox(key="rc_kind").select("Ortalama").run()
    application.selectbox(key="rc_column").select("değerlendirme_puanı").run()
    application.button(key="rc_button").click().run()
    _ok(application)
    assert any("en büyük katkı “Ankara”" in item.value for item in application.success)
    application.switch_page(RAPOR).run()
    application.button(key="report_button").click().run()
    _ok(application)
    page = application.session_state["report_files"]["html"].decode("utf-8")
    assert "<h2>Neden Değişti?</h2>" in page


def test_experiment_page_and_report(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.run()
    application.sidebar.button(key="demo_button").click().run()
    application.switch_page("views/deney.py").run()
    _ok(application)
    assert application.selectbox(key="ab_group").value == "deney_grubu"
    assert application.selectbox(key="ab_unit").value == "müşteri_no"
    application.button(key="ab_button").click().run()
    _ok(application)
    assert any("“B” grubunda" in item.value and "anlamlı biçimde yüksek" in item.value for item in application.success)
    application.switch_page(RAPOR).run()
    _ok(application)
    assert any("A/B testi: var" in item.value for item in application.caption)
    application.button(key="report_button").click().run()
    _ok(application)
    assert "<h2>A/B Testi</h2>" in application.session_state["report_files"]["html"].decode("utf-8")


def test_demo_link_opens_with_demo_data(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Paylaşılan ?demo=1 bağlantısı tıklama gerektirmeden demo veriyle açılır."""
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    application = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    application.query_params["demo"] = "1"
    application.run()
    _ok(application)
    assert application.session_state["data_name"] == "demo_eticaret_siparisleri.csv"
    assert "🚀 Otopilot Yönetici Özeti" in [item.value for item in application.header]
