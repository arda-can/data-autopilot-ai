"""Sayfaların paylaştığı oturum durumu, hatırlanan seçimler ve önbellekli hesaplar.

Streamlit bir sayfada çizilmeyen widget'ların durumunu siler. Sayfalar arasında kaybolmaması gereken
seçimler (zaman serisi ölçüsü, hedef sütun, tahmin anı ayarları…) bu yüzden ayrı bir anahtarda
hatırlanır ve widget yeniden çizilirken varsayılan değer olarak verilir. Veri değişince hepsi unutulur.

Bir sonucu etkileyen her seçim önbellek anahtarına girer; anahtarın ilk öğesi her zaman veri imzasıdır.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core.autopilot import AutopilotEngine, AutopilotResult
from core.automl import AutoMLEngine
from core.customers import CustomerAnalysis, CustomerAnalyzer, CustomerColumns, detect_customer_columns
from core.explain import ModelExplainer
from core.relations import RelationFinder, load_tables
from core.rules import DataRule, RuleReport, evaluate_rules, suggest_rules
from core.timeseries import TimeSeriesEngine, date_columns

MEMORY_PREFIX = "_hatirla_"
NO_TARGET = "-- Hedef seçin --"
DEFAULT_AUTOML_SETTINGS: tuple = (None, (), ())
SINGLE_TABLE = "Tek tablo"


def init_state() -> None:
    for key, value in (
        ("dataframe", None), ("autopilot_result", None), ("data_signature", None), ("data_name", "veri"),
    ):
        st.session_state.setdefault(key, value)


def remembered(key: str, default: Any = None) -> Any:
    return st.session_state.get(MEMORY_PREFIX + key, default)


def remember(key: str, value: Any) -> None:
    st.session_state[MEMORY_PREFIX + key] = value


def forget_selections() -> None:
    """Yeni veri yüklenince önceki verinin seçimleri yeni veriye taşınmaz."""
    for key in [key for key in st.session_state if str(key).startswith(MEMORY_PREFIX)]:
        del st.session_state[key]


def index_of(options: list, value: Any, default: int = 0) -> int:
    return options.index(value) if value in options else default


def set_dataset(frame, signature: str, name: str) -> None:
    st.session_state.dataframe = frame
    st.session_state.data_signature = signature
    st.session_state.data_name = name
    st.session_state.autopilot_result = None
    forget_selections()


def load_dataset(load, signature: str, name: str) -> None:
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
    set_dataset(frame, signature, name)


def joined_active() -> bool:
    return str(st.session_state.data_signature or "").startswith("join:")


@st.cache_resource(show_spinner="Tablolar okunuyor ve ilişkiler aranıyor...")
def discover_tables(signature: tuple[tuple[str, int, int], ...]) -> tuple:
    """`data/` klasöründeki tabloları bir kez okur; dosyalar değişince imza değişir ve yeniden okunur."""
    tables = load_tables([Path(item[0]) for item in signature])
    finder = RelationFinder()
    return tables, finder.describe(tables), finder.discover(tables)


# Sunucudaki dosyalardan (demo, data/ klasörü, örnek veri) gelen ve LLM kullanmayan sonuçlar oturumlar arasında
# paylaşılır: imza dosya yolu ve değiştirilme zamanını içerir, dosya değişince yeniden hesaplanır. Yüklenen
# dosyalar ve Gemini anahtarlı çalışmalar paylaşılmaz. Sonuç nesnesi hiçbir yerde yerinde değiştirilmez.
SHARED_SOURCES = ("demo:", "local:", "sample:")


@st.cache_resource(max_entries=4, show_spinner=False)
def _shared_autopilot(signature: str, _frame: pd.DataFrame) -> AutopilotResult:
    return AutopilotEngine().run_full_autopilot(_frame, api_key="")


def require_result() -> AutopilotResult:
    """Otopilot sonucunu döndürür; veri yoksa ya da çoklu tablo henüz birleştirilmediyse sayfayı durdurur."""
    if st.session_state.get("work_mode", SINGLE_TABLE) != SINGLE_TABLE and not joined_active():
        st.info("Çoklu tablo modunda önce Genel Bakış sayfasında tabloları birleştirin.")
        st.stop()
    frame = st.session_state.dataframe
    if frame is None:
        st.info("Başlamak için sol menüden bir CSV seçin, Excel/CSV yükleyin veya örnek veri üretin.")
        st.stop()
    if st.session_state.autopilot_result is None:
        with st.spinner("Otopilot veriyi temizliyor ve en mantıklı analizleri hazırlıyor..."):
            try:
                api_key = st.session_state.get("api_key", "")
                signature = str(st.session_state.data_signature or "")
                if not api_key and signature.startswith(SHARED_SOURCES):
                    st.session_state.autopilot_result = _shared_autopilot(signature, frame)
                else:
                    st.session_state.autopilot_result = AutopilotEngine().run_full_autopilot(frame, api_key=api_key)
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Otopilot çalıştırılamadı: {exc}")
                st.stop()
    return st.session_state.autopilot_result


# --- Zaman serisi -------------------------------------------------------------------------------

def default_timeseries_request(result: AutopilotResult) -> tuple | None:
    dates = date_columns(result.cleaned_frame)
    if not dates:
        return None
    default_date = result.timeseries.summary.date_column if result.timeseries is not None else dates[0]
    return (default_date, None, "auto")


def timeseries_request(result: AutopilotResult) -> tuple | None:
    """Kullanıcının Zaman Serisi sayfasındaki son seçimi; sayfa açılmadıysa otopilotun varsayılanı."""
    return remembered("ts_request", default_timeseries_request(result))


def timeseries_for(result: AutopilotResult, request: tuple | None) -> tuple:
    """(analiz, hata metni). Varsayılan seçim otopilot sonucundan gelir; diğerleri bir kez hesaplanır."""
    if request is None or request == default_timeseries_request(result):
        return result.timeseries, result.timeseries_message
    cache = st.session_state.setdefault("timeseries_cache", {})
    key = (st.session_state.data_signature, *request)
    if key not in cache:
        try:
            cache[key] = TimeSeriesEngine().analyze(
                result.cleaned_frame, date_column=request[0], value_column=request[1], frequency=request[2]
            )
        except ValueError as exc:
            cache[key] = f"Zaman serisi analizi yapılamadı: {exc}"
    cached = cache[key]
    return (None, cached) if isinstance(cached, str) else (cached, "")


# --- Dönem karşılaştırması ----------------------------------------------------------------------

def stored_drift(request: tuple | None):
    """Bu veri ve bu istek için yapılmış karşılaştırma (sonuç ya da hata metni); yoksa None."""
    stored = st.session_state.get("drift_result")
    if request is not None and stored and stored[0] == st.session_state.data_signature and stored[1] == request:
        return stored[2]
    return None


def stored_root_cause(request: tuple | None):
    """Bu veri ve bu istek için yapılmış kök neden ayrıştırması (sonuç ya da hata metni); yoksa None."""
    stored = st.session_state.get("root_cause_result")
    if request is not None and stored and stored[0] == st.session_state.data_signature and stored[1] == request:
        return stored[2]
    return None


def stored_experiment(spec):
    """Bu veri ve bu deney tanımı için yapılmış analiz (sonuç ya da hata metni); yoksa None."""
    stored = st.session_state.get("experiment_result")
    if spec is not None and stored and stored[0] == st.session_state.data_signature \
            and stored[1] == spec.model_dump_json():
        return stored[2]
    return None


# --- İş kuralları -------------------------------------------------------------------------------

def rule_id(rule: DataRule) -> str:
    return hashlib.sha1(rule.key.encode("utf-8")).hexdigest()[:10]


def rule_candidates(result: AutopilotResult) -> list[DataRule]:
    """Onay bekleyen öneriler: yerel öneriler (veri başına bir kez hesaplanır) ve varsa Gemini önerileri."""
    cache = st.session_state.setdefault("rule_suggestion_cache", {})
    signature = st.session_state.data_signature
    if signature not in cache:
        cache[signature] = suggest_rules(result.cleaned_frame)
    gemini = [DataRule.model_validate(item) for item in remembered("gemini_rules", [])]
    seen: set[str] = set()
    return [rule for rule in [*cache[signature], *gemini] if not (rule.key in seen or seen.add(rule.key))]


def custom_rules() -> list[DataRule]:
    return [DataRule.model_validate(item) for item in remembered("custom_rules", [])]


def add_custom_rules(rules: list[DataRule]) -> int:
    """Aynı kural iki kez eklenmez; eklenen kural sayısını döndürür."""
    current = remembered("custom_rules", [])
    keys = {DataRule.model_validate(item).key for item in current}
    fresh = [rule.model_dump() for rule in rules if rule.key not in keys]
    remember("custom_rules", [*current, *fresh])
    return len(fresh)


def active_rules(result: AutopilotResult) -> list[DataRule]:
    """Uygulanan kurallar: onaylı öneriler (varsayılan: onaylı) ve kullanıcının eklediği kurallar."""
    # Onay kutusunun güncel değeri (bu çalışmada tıklandıysa) hatırlanandan önce gelir.
    approved = [
        rule for rule in rule_candidates(result)
        if st.session_state.get(f"rule_on_{rule_id(rule)}", remembered(f"rule_on_{rule_id(rule)}", True))
    ]
    keys = {rule.key for rule in approved}
    return approved + [rule for rule in custom_rules() if rule.key not in keys]


def rule_evaluation(result: AutopilotResult) -> tuple[RuleReport | None, pd.DataFrame | None]:
    """(kural raporu, ihlal matrisi); kural yoksa (None, None). Aynı veri ve kural kümesi için bir kez hesaplanır."""
    rules = active_rules(result)
    if not rules:
        return None, None
    cache = st.session_state.setdefault("rule_cache", {})
    key = (st.session_state.data_signature, tuple(rule.key for rule in rules))
    if key not in cache:
        cache.clear()  # yalnızca son değerlendirme tutulur; ihlal matrisi veri boyutundadır
        cache[key] = evaluate_rules(result.cleaned_frame, rules)
    return cache[key]


# --- Müşteri analizi ----------------------------------------------------------------------------

def default_customer_request(result: AutopilotResult) -> tuple | None:
    columns = detect_customer_columns(result.cleaned_frame)
    if columns is None:
        return None
    return (columns.customer, columns.date, columns.amount, columns.order, "Tüm veri", None)


def customer_request(result: AutopilotResult) -> tuple | None:
    """(müşteri, tarih, tutar, sipariş, pencere, kırılım); Müşteri sayfası açılmadıysa otomatik bulunan sütunlar."""
    return remembered("customer_request", default_customer_request(result))


def customer_analysis_for(result: AutopilotResult, request: tuple | None) -> CustomerAnalysis | str | None:
    """Analiz, Türkçe hata metni ya da (uygun sütun yoksa) None. Aynı istek için bir kez hesaplanır."""
    if request is None:
        return None
    cache = st.session_state.setdefault("customer_cache", {})
    key = (st.session_state.data_signature, *request)
    if key not in cache:
        customer, date, amount, order, window, group = request
        try:
            cache[key] = CustomerAnalyzer().analyze(
                result.cleaned_frame, CustomerColumns(customer=customer, date=date, amount=amount, order=order),
                window_label=window, group_column=group,
            )
        except ValueError as exc:
            cache[key] = f"Müşteri analizi yapılamadı: {exc}"
    return cache[key]


# --- AutoML ve açıklama -------------------------------------------------------------------------

def automl_selection(result: AutopilotResult) -> tuple[str | None, tuple]:
    """(hedef, ayarlar). Hedef None ise kullanıcı hedef seçmemiştir."""
    default_target = result.suggested_target if result.suggested_target in result.cleaned_frame.columns else None
    return remembered("automl_target", default_target), remembered("automl_settings", DEFAULT_AUTOML_SETTINGS)


def automl_for(result: AutopilotResult, target: str | None, settings: tuple) -> tuple:
    """(AutoML sonucu, hata metni). Aynı veri, hedef ve ayarlar için model yalnızca bir kez eğitilir."""
    if target is None or (target == result.suggested_target and settings == DEFAULT_AUTOML_SETTINGS):
        return result.automl_result, ""
    cache = st.session_state.setdefault("automl_cache", {})
    key = (st.session_state.data_signature, target, settings)
    if key not in cache:
        cleaned = result.cleaned_frame
        sample = cleaned.sample(10_000, random_state=42) if len(cleaned) > 10_000 else cleaned
        try:
            with st.spinner("Modeller çapraz doğrulamayla karşılaştırılıyor..."):
                cache[key] = AutoMLEngine().run(
                    sample, target, time_column=settings[0], exclude_columns=settings[1], known_columns=settings[2],
                )
        except (ValueError, TypeError, KeyError) as exc:
            cache[key] = f"'{target}' hedefi için AutoML çalıştırılamadı: {exc}"
    cached = cache[key]
    return (None, cached) if isinstance(cached, str) else (cached, "")


def explanation_for(target: str | None, settings: tuple, automl):
    """Kısmi bağımlılık ve what-if hazırlığı; açıklanamıyorsa Türkçe neden metni, model yoksa None."""
    if automl is None or automl.model is None or automl.background is None:
        return None
    cache = st.session_state.setdefault("explain_cache", {})
    key = (st.session_state.data_signature, target, settings, automl.best_model)
    if key not in cache:
        try:
            cache[key] = ModelExplainer().explain(automl)
        except ValueError as exc:
            cache[key] = f"Açıklama üretilemedi: {exc}"
    return cache[key]
