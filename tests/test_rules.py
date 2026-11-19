"""Kullanıcı tanımlı veri kuralları: sınırlı türler, eval yok, öneriler ve rapordaki uyum bölümü."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest
from pydantic import ValidationError

from core.autopilot import AutopilotEngine, AutopilotResult
from core.demo_data import DEMO_FILENAME, write_demo_data
from core.ingestion import read_local_csv
from core.metadata import MetadataGenerator
from core.report import build_report, render_html, render_pdf
from core.rules import (
    PATTERN_PRESETS, DataRule, GeminiRuleAdvisor, evaluate_rules, suggest_rules, violating_records, violation_mask,
)
from tests.conftest import ROOT


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "no": ["a1", "a2", "a2", None, "a5"],
        "tutar": [10.0, -1.0, 5.0, None, 200.0],
        "şehir": ["İstanbul", "istanbul", "Ankara", "Ankara", None],
        "eposta": ["x@ornek.com", "bozuk", None, "y@ornek.com.tr", "z@ornek.com"],
        "sipariş": pd.to_datetime(["2025-01-01 14:00", "2025-01-02 10:00", "2025-01-03 09:00", None, "2025-01-05 08:00"]),
        "teslim": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-06", "2025-01-07", None]),
    })


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> AutopilotResult:
    path = write_demo_data(tmp_path_factory.mktemp("kurallar") / DEMO_FILENAME)
    return AutopilotEngine().run_full_autopilot(read_local_csv(path))


def _count(frame: pd.DataFrame, **rule) -> int:
    return int(violation_mask(frame, DataRule(**rule))[0].sum())


def test_each_rule_kind_counts_violations(frame: pd.DataFrame) -> None:
    assert _count(frame, kind="aralık", column="tutar", minimum=0) == 1
    assert _count(frame, kind="aralık", column="tutar", minimum=0, maximum=100) == 2
    assert _count(frame, kind="benzersiz", column="no") == 2  # tekrarlanan değerin iki kaydı da
    assert _count(frame, kind="boş_olamaz", column="no") == 1
    assert _count(frame, kind="izinli_değerler", column="şehir", values=["İstanbul", "Ankara"]) == 1
    assert _count(frame, kind="desen", column="eposta", pattern=PATTERN_PRESETS["E-posta"]) == 1
    assert _count(frame, kind="karşılaştırma", column="teslim", operator=">=", other_column="sipariş") == 1


def test_nulls_only_violate_not_null_rule(frame: pd.DataFrame) -> None:
    mask, checked = violation_mask(frame, DataRule(kind="aralık", column="tutar", minimum=0))
    assert checked == 4 and not mask[frame["tutar"].isna()].any()


def test_same_day_delivery_is_not_before_order(frame: pd.DataFrame) -> None:
    """Saatsiz teslim tarihi, aynı gün 14:00'teki siparişten "önce" sayılmamalı (gün düzeyinde kıyas)."""
    mask, checked = violation_mask(frame, DataRule(kind="karşılaştırma", column="teslim", operator=">=",
                                                   other_column="sipariş"))
    assert checked == 3 and mask.tolist() == [False, True, False, False, False]


@pytest.mark.parametrize(("rule", "message"), [
    ({"kind": "aralık", "column": "x"}, "en az bir sınır"),
    ({"kind": "aralık", "column": "x", "minimum": 5, "maximum": 1}, "en küçük değer"),
    ({"kind": "izinli_değerler", "column": "x"}, "en az bir değer"),
    ({"kind": "desen", "column": "x", "pattern": "(a+)+$"}, "İç içe niceleyici"),
    ({"kind": "desen", "column": "x", "pattern": "[a-"}, "geçerli bir düzenli ifade değil"),
    ({"kind": "karşılaştırma", "column": "x", "operator": "<", "other_column": "x"}, "iki farklı sütun"),
    ({"kind": "karşılaştırma", "column": "x", "other_column": "y"}, "işleç"),
])
def test_invalid_rules_are_rejected_in_turkish(rule: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        DataRule(**rule)


def test_unknown_kind_and_operator_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DataRule(kind="ifade", column="x")
    with pytest.raises(ValidationError):
        DataRule(kind="karşılaştırma", column="x", operator="__import__", other_column="y")


def test_inapplicable_rules_are_reported_not_skipped(frame: pd.DataFrame) -> None:
    report, _ = evaluate_rules(frame, [
        DataRule(kind="aralık", column="yok", minimum=0),
        DataRule(kind="aralık", column="şehir", minimum=0),
        DataRule(kind="karşılaştırma", column="tutar", operator="<", other_column="sipariş"),
        DataRule(kind="aralık", column="tutar", minimum=0),
    ])
    errors = [item.error for item in report.results]
    assert "veride yok" in errors[0] and "sayısal değil" in errors[1] and "karşılaştırılamaz" in errors[2]
    assert errors[3] == "" and len(report.applied) == 1


def test_compliance_is_row_level_and_absent_without_rules(frame: pd.DataFrame) -> None:
    empty, _ = evaluate_rules(frame, [])
    assert empty.compliance_score is None and empty.violating_rows == 0
    report, matrix = evaluate_rules(frame, [
        DataRule(kind="aralık", column="tutar", minimum=0),          # satır 1
        DataRule(kind="izinli_değerler", column="şehir", values=["İstanbul", "Ankara"]),  # satır 1
        DataRule(kind="boş_olamaz", column="no"),                    # satır 3
    ])
    assert report.violating_rows == 2 and report.compliance_score == pytest.approx(60.0)
    records = violating_records(frame, matrix)
    assert records["kayıt_no"].tolist() == [1, 3]
    assert records.iloc[0]["ihlal_edilen_kurallar"].count(";") == 1


def test_repeated_alternation_pattern_is_rejected() -> None:
    """Regresyon: (a|aa)+$ iç içe niceleyici süzgecinden geçiyor ve felaket geri izlemeyle arayüzü kilitliyordu."""
    with pytest.raises(ValidationError, match="Tekrarlanan alternatif"):
        DataRule(kind="desen", column="x", pattern="(a|aa)+$")
    DataRule(kind="desen", column="x", pattern=r"(\+90|0)?5\d{9}")  # isteğe bağlı alternatif sorun değil


def test_slow_custom_pattern_is_stopped_and_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Süzgeçten geçen ama yavaş bir desen (art arda niceleyiciler) süre sınırında durdurulur, hata olarak raporlanır."""
    import time

    import core.rules as rules_module

    monkeypatch.setattr(rules_module, "PATTERN_TIMEOUT_SECONDS", 1.5)
    frame = pd.DataFrame({"kod": ["1" * 3000, "123"]})
    started = time.perf_counter()
    report, matrix = evaluate_rules(frame, [DataRule(kind="desen", column="kod", pattern=r"\d*\d*\d*\d*\d*\d*!")])
    assert time.perf_counter() - started < 15
    assert "saniyede değerlendirilemedi" in report.results[0].error and report.compliance_score is None


def test_custom_pattern_checks_the_whole_value() -> None:
    """Regresyon: desen yalnızca ilk 256 karaktere uygulanıyordu; sondaki geçersiz karakter görülmüyordu."""
    frame = pd.DataFrame({"kod": ["a" * 256 + "!", "a" * 300, "aa"]})
    mask, checked = violation_mask(frame, DataRule(kind="desen", column="kod", pattern="a+"))
    assert checked == 3 and mask.tolist() == [True, False, False]
    long_email = "x" * 300 + "@ornek.com!"
    mask, _ = violation_mask(pd.DataFrame({"e": [long_email]}), DataRule(kind="desen", column="e",
                                                                          pattern=PATTERN_PRESETS["E-posta"]))
    assert mask.tolist() == [True]


def test_rules_with_the_same_description_do_not_overwrite_each_other() -> None:
    """Regresyon: maskeler kısaltılmış açıklamayla tutuluyordu; ilk 8 değeri aynı iki kural birbirini eziyordu."""
    first8 = [f"d{i}" for i in range(8)]
    frame = pd.DataFrame({"x": ["d9", "d10"]})
    rules = [DataRule(kind="izinli_değerler", column="x", values=[*first8, "d9"]),
             DataRule(kind="izinli_değerler", column="x", values=[*first8, "d10"])]
    assert rules[0].describe() == rules[1].describe()
    report, matrix = evaluate_rules(frame, rules)
    assert matrix.shape[1] == 2 and report.violating_rows == 2 and report.compliance_score == 0
    labels = violating_records(frame, matrix)["ihlal_edilen_kurallar"].tolist()
    assert labels[0] == rules[1].describe() + " (2)" and labels[1] == rules[0].describe()


def test_rules_round_trip_through_json() -> None:
    rule = DataRule(kind="karşılaştırma", column="teslim", operator=">=", other_column="sipariş", reason="r")
    assert DataRule.model_validate_json(rule.model_dump_json()) == rule
    assert rule.key == rule.model_copy(update={"reason": "başka", "source": "öneri"}).key


def test_rules_module_has_no_dynamic_evaluation() -> None:
    source = (ROOT / "core" / "rules.py").read_text(encoding="utf-8")
    assert "eval(" not in source and ".query(" not in source and "exec(" not in source


def test_suggestions_find_every_injected_demo_violation(demo_result: AutopilotResult) -> None:
    rules = suggest_rules(demo_result.cleaned_frame)
    report, _ = evaluate_rules(demo_result.cleaned_frame, rules)
    found = {item.description: item.violations for item in report.results}
    assert found["“tutar” en az 0 olmalı"] == 4
    assert found["“indirim_oranı” 0 ile 1 arasında olmalı"] == 3
    assert found["“teslim_tarihi” >= “sipariş_tarihi” olmalı"] == 5
    assert found["“değerlendirme_puanı” 1 ile 5 arasında olmalı"] == 3  # 2–5 değil: geçerli 1 puanlar seyrek
    assert found["“eposta” geçerli e-posta biçiminde olmalı"] == 6
    assert found["“sipariş_no” her kayıtta farklı olmalı"] == 4
    assert next(value for key, value in found.items() if key.startswith("“şehir” yalnızca")) == 13
    assert found["“tahmini_teslim_tarihi” >= “sipariş_tarihi” olmalı"] == 0
    assert all(item.rule.source == "öneri" and item.rule.reason for item in report.results)


def test_suggestions_skip_person_and_contact_columns(demo_result: AutopilotResult) -> None:
    columns = {(rule.kind, rule.column) for rule in suggest_rules(demo_result.cleaned_frame)}
    assert ("benzersiz", "müşteri_adı") not in columns and ("benzersiz", "eposta") not in columns
    assert ("benzersiz", "müşteri_no") not in columns  # tekrar eden müşteriler: benzersiz değil


class _FakeRuleClient:
    text = ""
    prompts: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.models = self

    def generate_content(self, model: str, contents: str, config) -> SimpleNamespace:
        _FakeRuleClient.prompts.append(contents)
        return SimpleNamespace(parsed=None, text=_FakeRuleClient.text)


@pytest.fixture
def fake_rule_sdk(monkeypatch: pytest.MonkeyPatch):
    import google.genai

    _FakeRuleClient.prompts = []
    monkeypatch.setattr(google.genai, "Client", _FakeRuleClient)
    return _FakeRuleClient


def test_gemini_rules_are_validated_against_data(fake_rule_sdk, frame: pd.DataFrame) -> None:
    frame = frame.assign(ad=["Ayşe Yılmaz", "Ali Kaya", "Can Demir", "Ece Koç", "Emre Kurt"])
    fake_rule_sdk.text = json.dumps({"rules": [
        {"kind": "aralık", "column": "tutar", "minimum": 0, "reason": "Tutar negatif olamaz."},
        {"kind": "benzersiz", "column": "olmayan_sütun", "reason": "x"},
    ]}, ensure_ascii=False)
    advisor = GeminiRuleAdvisor("sahte-anahtar")
    accepted, rejected = advisor.suggest(frame, MetadataGenerator().generate(frame))
    assert [(rule.column, rule.source) for rule in accepted] == [("tutar", "Gemini")]
    assert rejected[0].reason == "“olmayan_sütun” sütunu veride yok."
    assert advisor.origin.source == "Gemini" and not advisor.origin.error
    assert "Ayşe Yılmaz" not in fake_rule_sdk.prompts[0] and "x@ornek.com" not in fake_rule_sdk.prompts[0]


def test_gemini_free_form_rule_is_refused_visibly(fake_rule_sdk, frame: pd.DataFrame) -> None:
    fake_rule_sdk.text = json.dumps({"rules": [{"kind": "ifade", "column": "tutar", "pattern": "tutar > 0"}]})
    advisor = GeminiRuleAdvisor("sahte-anahtar")
    accepted, rejected = advisor.suggest(frame, MetadataGenerator().generate(frame))
    assert accepted == [] and rejected == []
    assert advisor.origin.error and "ValidationError" in advisor.origin.reason


def test_report_has_rule_section_without_raw_values(demo_result: AutopilotResult) -> None:
    rules, _ = evaluate_rules(demo_result.cleaned_frame, suggest_rules(demo_result.cleaned_frame))
    report = build_report(demo_result, "demo.csv", rules=rules)
    assert report.rules is not None and json.loads(report.model_dump_json())["rules"]["violating_rows"] == 37
    assert any(item.startswith("İş kuralı uyumu %99,5") for item in report.headline)
    assert any(item.startswith("Kural ihlali:") for item in report.recommendations)
    page = render_html(report, embed_plotly=False)
    assert "<h2>İş Kuralı Uyumu</h2>" in page and "Yapısal Hijyen" in page
    assert "@ornek.com.tr" not in page and " at ornek" not in page
    assert render_pdf(report).startswith(b"%PDF")
