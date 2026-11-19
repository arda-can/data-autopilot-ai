"""Kalite: yapısal hijyen (boşluk, tekrar, tip) ve iş kuralı uyumu (kullanıcı tanımlı kurallar) ayrı ölçülür."""

from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from core.formatting import decimal_tr, display_frame, integer_tr
from core.ingestion import content_signature
from core.rules import (
    PATTERN_PRESETS, RULE_KIND_LABELS, DataRule, GeminiRuleAdvisor, violating_records,
)
from views.common import (
    add_custom_rules, custom_rules, remember, remembered, require_result, rule_candidates, rule_evaluation, rule_id,
)

result = require_result()
before, after, cleaned = result.before_report, result.after_report, result.cleaned_frame
columns = [str(column) for column in cleaned.columns]


def _csv(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(str(item["msg"]).removeprefix("Value error, ") for item in exc.errors())


st.header("🧹 Veri Kalitesi")
st.caption(
    "İki ayrı ölçü: **yapısal hijyen** verinin biçimini (boş/sabit sütun, mükerrer, tip) ölçer ve değerlerin "
    "doğru olup olmadığını bilmez; **iş kuralı uyumu** tanımladığınız kurallara uyan kayıtların oranıdır."
)
rule_report, rule_matrix = rule_evaluation(result)
metrics = st.columns(3)
metrics[0].metric(
    "Yapısal Hijyen Skoru",
    f"{decimal_tr(after.kalite_skoru)}/100",
    delta=decimal_tr(after.kalite_skoru - before.kalite_skoru),
)
metrics[1].metric(
    "İş Kuralı Uyumu",
    "—" if rule_report is None or rule_report.compliance_score is None
    else f"%{decimal_tr(rule_report.compliance_score)}",
    help="Hiçbir kuralı ihlal etmeyen kayıtların oranı. Kural yoksa skor gösterilmez.",
)
metrics[2].metric(
    "Kural İhlali Olan Kayıt", "—" if rule_report is None else integer_tr(rule_report.violating_rows),
)

# --- Yapısal hijyen ---------------------------------------------------------------------------
st.subheader("Yapısal hijyen")
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
st.download_button("Temizlenmiş CSV'yi indir", _csv(cleaned), "otopilot_temizlenmis_veri.csv", "text/csv")

# --- İş kuralları -----------------------------------------------------------------------------
st.subheader("📏 İş kuralları")
st.caption(
    "Kurallar serbest kod değil, altı sınırlı türden biridir: aralık, benzersiz, boş olamaz, izinli değerler, "
    "desen, iki sütun karşılaştırması. Kurallar veriyi değiştirmez; yalnızca ihlal eden kayıtları gösterir."
)

candidates = rule_candidates(result)
if candidates:
    st.write("**Önerilen kurallar** — veriden çıkarıldı; uygulamak istemediklerinizin işaretini kaldırın:")
    for rule in candidates:
        key = f"rule_on_{rule_id(rule)}"
        enabled = st.checkbox(
            f"{rule.describe()} · {rule.source}", value=remembered(key, True), key=key, help=rule.reason or None,
        )
        remember(key, enabled)

api_key = st.session_state.get("api_key", "")
if api_key and st.button("Gemini ile kural öner", key="rules_gemini_button"):
    advisor = GeminiRuleAdvisor(api_key)
    with st.spinner("Gemini'ye yalnızca KVKK maskeli metadata kartı gönderiliyor..."):
        accepted, rejected_rules = advisor.suggest(cleaned, result.metadata)
    remember("gemini_rules", [rule.model_dump() for rule in accepted])
    remember("gemini_origin", advisor.origin.model_dump())
    remember("gemini_rejected", [item.model_dump() for item in rejected_rules])
    st.rerun()
if remembered("gemini_origin"):
    origin = remembered("gemini_origin")
    (st.warning if origin["error"] else st.caption)(f"Kural önerisi: {origin['source']} — {origin['reason']}")
    for item in remembered("gemini_rejected", []):
        st.caption(f"Reddedilen öneri — {item['description']}: {item['reason']}")

with st.expander("Kural ekle"):
    kind_labels = {label: kind for kind, label in RULE_KIND_LABELS.items()}
    kind = kind_labels[st.selectbox("Kural türü", list(kind_labels), key="rule_kind")]
    with st.form("rule_form", clear_on_submit=True):
        column = st.selectbox("Sütun", columns, key="rule_column")
        params: dict = {}
        if kind == "aralık":
            low, high = st.columns(2)
            params["minimum"] = low.number_input("En küçük (boş = sınır yok)", value=None, key="rule_min")
            params["maximum"] = high.number_input("En büyük (boş = sınır yok)", value=None, key="rule_max")
        elif kind == "izinli_değerler":
            params["values_text"] = st.text_area(
                "İzinli değerler (her satıra bir değer; büyük/küçük harf ve aksan önemlidir)", key="rule_values"
            )
        elif kind == "desen":
            preset = st.selectbox("Hazır desen", [*PATTERN_PRESETS, "Özel düzenli ifade"], key="rule_preset")
            custom = st.text_input("Özel düzenli ifade (değerin tamamı eşleşmeli)", key="rule_pattern")
            params["pattern"] = custom if preset == "Özel düzenli ifade" else PATTERN_PRESETS[preset]
        elif kind == "karşılaştırma":
            op_ui, other_ui = st.columns([1, 3])
            params["operator"] = op_ui.selectbox("İşleç", ["<", "<=", ">", ">=", "==", "!="], key="rule_operator")
            params["other_column"] = other_ui.selectbox("İkinci sütun", columns, key="rule_other")
        reason = st.text_input("Gerekçe (isteğe bağlı)", key="rule_reason")
        submitted = st.form_submit_button("Kuralı ekle")
    if submitted:
        values_text = params.pop("values_text", "")
        if kind == "izinli_değerler":
            params["values"] = [line.strip() for line in values_text.splitlines() if line.strip()]
        try:
            new_rule = DataRule(kind=kind, column=column, reason=reason, **params)
        except ValidationError as exc:
            st.error(f"Kural eklenemedi: {_validation_message(exc)}")
        else:
            if add_custom_rules([new_rule]):
                st.rerun()
            st.info("Bu kural zaten ekli.")

own_rules = custom_rules()
if own_rules:
    st.write("**Eklediğiniz kurallar**")
    for rule in own_rules:
        text_ui, button_ui = st.columns([5, 1])
        text_ui.write(f"• {rule.describe()}" + (f" — {rule.reason}" if rule.reason else ""))
        if button_ui.button("Kaldır", key=f"rule_remove_{rule_id(rule)}"):
            remember("custom_rules", [item.model_dump() for item in own_rules if item.key != rule.key])
            st.rerun()

io_left, io_right = st.columns(2)
active = ([] if rule_report is None else [item.rule for item in rule_report.results])
io_left.download_button(
    "Kuralları JSON olarak indir",
    json.dumps({"rules": [rule.model_dump() for rule in active]}, ensure_ascii=False, indent=2).encode("utf-8"),
    "veri_kurallari.json", "application/json", disabled=not active,
)
rule_file = io_right.file_uploader("Kural dosyası yükle (JSON)", type=["json"], key="rule_file")
if rule_file is not None and remembered("rule_file_signature") != content_signature(rule_file):
    remember("rule_file_signature", content_signature(rule_file))
    try:
        payload = json.loads(rule_file.getvalue().decode("utf-8"))
        loaded = [DataRule.model_validate(item) for item in payload.get("rules", [])]
    except (ValueError, AttributeError) as exc:
        message = _validation_message(exc) if isinstance(exc, ValidationError) else str(exc)
        st.error(f"Kural dosyası okunamadı: {message}")
    else:
        add_custom_rules(loaded)
        st.rerun()

def _show_rule_report() -> None:
    st.dataframe(
        display_frame(pd.DataFrame([
            {
                "Kural": item.description, "Tür": RULE_KIND_LABELS[item.rule.kind], "Kaynak": item.rule.source,
                "Kontrol edilen": item.checked, "İhlal": item.violations, "Uyum": item.compliance,
                "Not": item.error or item.rule.reason,
            }
            for item in rule_report.results
        ])),
        width="stretch", hide_index=True,
    )
    for item in rule_report.results:
        if item.error:
            st.warning(f"Uygulanamayan kural — {item.description}: {item.error}")
    st.caption(rule_report.method_note)
    records = violating_records(cleaned, rule_matrix)
    if len(records):
        st.download_button(
            f"İhlal eden {integer_tr(len(records))} kaydı indir (CSV)", _csv(records), "kural_ihlalleri.csv",
            "text/csv", key="rules_download",
        )
        with st.expander("İhlal eden kayıtlar (ilk 100)"):
            st.dataframe(display_frame(records.head(100)), width="stretch", hide_index=True)


if rule_report is None:
    st.info("Henüz uygulanan kural yok; iş kuralı uyumu skoru kural tanımlanınca hesaplanır.")
else:
    _show_rule_report()

st.header("🔍 Detaylar")
with st.expander("Audit Log'u göster"):
    st.dataframe(
        display_frame(pd.DataFrame([item.model_dump() for item in result.audit_logs])),
        width="stretch", hide_index=True,
    )
with st.expander("KVKK maskeli Metadata Kartını göster"):
    st.json(result.metadata.model_dump())
