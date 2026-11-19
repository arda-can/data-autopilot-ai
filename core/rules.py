"""Kullanıcı tanımlı veri kuralları (iş kuralı uyumu).

Kurallar serbest ifade metni değildir; altı sınırlı türden biridir ve Pydantic ile doğrulanır:
aralık, benzersiz, boş olamaz, izinli değerler, desen, iki sütun karşılaştırması. Karşılaştırma işleçleri
`operator` modülüne eşlenir; `eval`/`DataFrame.query` kullanılmaz. LLM yalnızca bu şemaya uyan kural
önerir; her öneri veriye karşı doğrulanır, geçmeyen reddedilir ve nedeni gösterilir.

Yapısal hijyen skoru (`core/health.py`) boşluk, tekrar ve tip sorunlarını ölçer; değerlerin iş açısından
doğru olup olmadığını bilmez. Bu modülün uyum skoru o boşluğu doldurur: en az bir kuralı ihlal eden
kayıtların payı. Kural tanımlanmadıysa skor yoktur; "100" uydurulmaz.
"""

from __future__ import annotations

import operator
import re
from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from core.columns import _PERSON_NAME_TOKENS, is_identifier_name, is_text_series, name_tokens
from core.health import is_nonnegative_name
from core.metadata import DatasetMetadataCard
from core.planner import PlanOrigin, failure_reason, request_structured

RuleKind = Literal["aralık", "benzersiz", "boş_olamaz", "izinli_değerler", "desen", "karşılaştırma"]
CompareOperator = Literal["<", "<=", ">", ">=", "==", "!="]
RuleSource = Literal["kullanıcı", "öneri", "Gemini"]

RULE_KIND_LABELS: dict[str, str] = {
    "aralık": "Aralık", "benzersiz": "Benzersiz", "boş_olamaz": "Boş olamaz",
    "izinli_değerler": "İzinli değerler", "desen": "Desen", "karşılaştırma": "İki sütun karşılaştırması",
}
_OPERATORS: dict[str, Callable] = {
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne,
}
PATTERN_PRESETS: dict[str, str] = {
    "E-posta": r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}",
    "Türkiye cep telefonu": r"(\+90|0)?[ -]?5\d{2}[ -]?\d{3}[ -]?\d{2}[ -]?\d{2}",
    "Posta kodu (5 hane)": r"\d{5}",
}
_PRESET_PHRASES = {"E-posta": "e-posta", "Türkiye cep telefonu": "Türkiye cep telefonu",
                   "Posta kodu (5 hane)": "posta kodu (5 hane)"}
MAX_PATTERN_LENGTH = 200
MAX_TEXT_LENGTH = 256  # desen yalnızca değerin ilk 256 karakterine uygulanır
# İç içe niceleyici ((a+)+ gibi) felaket geri izlemeye yol açar; bu biçim reddedilir.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)\s*[+*{]")
_RATIO_TOKENS = frozenset({"oran", "oranı", "orani", "rate", "ratio"})
_EMAIL_TOKENS = frozenset({"email", "eposta", "mail"})
_PHONE_TOKENS = frozenset({"telefon", "tel", "phone", "gsm", "cep"})
_RATING_TOKENS = frozenset({"puan", "puanı", "puani", "skor", "skoru", "score", "rating", "yıldız", "yildiz"})
RARE_SHARE = 0.005  # metin önerileri için "seyrek" eşiği: kayıtların %0,5'i
RARE_SCALE_SHARE = 0.001  # tam sayı ölçekleri için: uç ama geçerli değerler (ör. 1 puan) seyrek olabilir


class DataRule(BaseModel):
    """Tek bir veri kuralı. Türüne göre gereken alanlar doğrulanır; JSON'a çevrilip geri yüklenebilir."""

    kind: RuleKind
    column: str
    minimum: float | None = None
    maximum: float | None = None
    values: list[str] = Field(default_factory=list)
    pattern: str | None = None
    operator: CompareOperator | None = None
    other_column: str | None = None
    reason: str = ""
    source: RuleSource = "kullanıcı"

    @model_validator(mode="after")
    def _check_parameters(self) -> DataRule:
        if self.kind == "aralık":
            if self.minimum is None and self.maximum is None:
                raise ValueError("Aralık kuralı için en az bir sınır (en küçük ya da en büyük) gerekir.")
            if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
                raise ValueError("Aralık kuralında en küçük değer en büyükten büyük olamaz.")
        elif self.kind == "izinli_değerler" and not self.values:
            raise ValueError("İzinli değerler kuralı için en az bir değer gerekir.")
        elif self.kind == "desen":
            if not self.pattern:
                raise ValueError("Desen kuralı için bir düzenli ifade gerekir.")
            if len(self.pattern) > MAX_PATTERN_LENGTH:
                raise ValueError(f"Desen en fazla {MAX_PATTERN_LENGTH} karakter olabilir.")
            if _NESTED_QUANTIFIER.search(self.pattern):
                raise ValueError("İç içe niceleyici içeren desenler (ör. (a+)+) yavaşlamaya yol açtığı için kabul edilmez.")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(f"Desen geçerli bir düzenli ifade değil: {exc}") from exc
        elif self.kind == "karşılaştırma":
            if self.operator is None or not self.other_column:
                raise ValueError("Karşılaştırma kuralı için işleç ve ikinci sütun gerekir.")
            if self.other_column == self.column:
                raise ValueError("Karşılaştırma kuralı iki farklı sütun ister.")
        return self

    @property
    def key(self) -> str:
        """Kuralın içerik anahtarı (gerekçe ve kaynak hariç): aynı kural iki kez eklenmez."""
        return self.model_dump_json(exclude={"reason", "source"})

    def describe(self) -> str:
        column = f"“{self.column}”"
        if self.kind == "aralık":
            if self.minimum is not None and self.maximum is not None:
                return f"{column} {_number(self.minimum)} ile {_number(self.maximum)} arasında olmalı"
            if self.minimum is not None:
                return f"{column} en az {_number(self.minimum)} olmalı"
            return f"{column} en çok {_number(self.maximum)} olmalı"
        if self.kind == "benzersiz":
            return f"{column} her kayıtta farklı olmalı"
        if self.kind == "boş_olamaz":
            return f"{column} boş olamaz"
        if self.kind == "izinli_değerler":
            shown = ", ".join(self.values[:8]) + (f" (+{len(self.values) - 8})" if len(self.values) > 8 else "")
            return f"{column} yalnızca şu değerleri alabilir: {shown}"
        if self.kind == "desen":
            preset = next((name for name, pattern in PATTERN_PRESETS.items() if pattern == self.pattern), None)
            if preset:
                return f"{column} geçerli {_PRESET_PHRASES[preset]} biçiminde olmalı"
            return f"{column} şu desene uymalı: {self.pattern}"
        return f"{column} {self.operator} “{self.other_column}” olmalı"


class RuleResult(BaseModel):
    """Bir kuralın sonucu. Yalnızca sayılar tutulur; ihlal eden değerler (PII olabilir) rapora girmez."""

    rule: DataRule
    description: str
    checked: int = 0
    violations: int = 0
    compliance: float | None = None
    error: str = ""


class RuleReport(BaseModel):
    rows: int
    results: list[RuleResult]
    violating_rows: int = 0
    compliance_score: float | None = None  # 0–100; kural yoksa None
    method_note: str = (
        "İş kuralı uyumu = hiçbir kuralı ihlal etmeyen kayıtların oranı. Boş değerler yalnızca “boş olamaz” "
        "kuralında ihlal sayılır; diğer kurallar dolu değerlere uygulanır. Kurallar temizlenmiş veride "
        "(mükerrerler çıkarıldıktan sonra) değerlendirilir."
    )

    @property
    def applied(self) -> list[RuleResult]:
        return [item for item in self.results if not item.error]


def _number(value: float) -> str:
    text = f"{value:,.4f}".rstrip("0").rstrip(".")
    return text.replace(",", "_").replace(".", ",").replace("_", ".")


def _date_only(series: pd.Series) -> bool:
    present = series.dropna()
    return bool(len(present)) and bool((present == present.dt.normalize()).all())


def _comparable(frame: pd.DataFrame, rule: DataRule) -> tuple[pd.Series, pd.Series]:
    left, right = frame[rule.column], frame[rule.other_column]
    if pd.api.types.is_datetime64_any_dtype(left) and pd.api.types.is_datetime64_any_dtype(right):
        # Biri yalnızca tarih (saat yok) ise gün düzeyinde kıyaslanır: 14:00'teki sipariş aynı gün teslim edilebilir.
        if _date_only(left) or _date_only(right):
            return left.dt.normalize(), right.dt.normalize()
        return left, right
    if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right) \
            and not pd.api.types.is_bool_dtype(left) and not pd.api.types.is_bool_dtype(right):
        return left, right
    raise ValueError(f"“{rule.column}” ile “{rule.other_column}” karşılaştırılamaz: ikisi de sayı ya da ikisi de tarih olmalı.")


def check_rule(frame: pd.DataFrame, rule: DataRule) -> str:
    """Kural bu veriye uygulanabilir mi? Uygulanamıyorsa Türkçe neden, uygulanabiliyorsa boş metin."""
    for column in (rule.column, rule.other_column):
        if column is not None and column not in frame.columns:
            return f"“{column}” sütunu veride yok."
    series = frame[rule.column]
    if rule.kind == "aralık" and (
        not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)
    ):
        return f"“{rule.column}” sayısal değil; aralık kuralı yalnızca sayısal sütunlara uygulanır."
    if rule.kind == "desen" and not is_text_series(series):
        return f"“{rule.column}” metin değil; desen kuralı yalnızca metin sütunlarına uygulanır."
    if rule.kind == "karşılaştırma":
        try:
            _comparable(frame, rule)
        except ValueError as exc:
            return str(exc)
    return ""


def violation_mask(frame: pd.DataFrame, rule: DataRule) -> tuple[pd.Series, int]:
    """(ihlal maskesi, kontrol edilen kayıt sayısı). Kural uygulanamıyorsa ValueError."""
    reason = check_rule(frame, rule)
    if reason:
        raise ValueError(reason)
    series = frame[rule.column]
    present = series.notna()
    if rule.kind == "boş_olamaz":
        return ~present, len(frame)
    if rule.kind == "benzersiz":
        return present & series.duplicated(keep=False), int(present.sum())
    if rule.kind == "aralık":
        outside = pd.Series(False, index=frame.index)
        if rule.minimum is not None:
            outside |= series < rule.minimum
        if rule.maximum is not None:
            outside |= series > rule.maximum
        return present & outside.fillna(False).astype(bool), int(present.sum())
    if rule.kind == "izinli_değerler":
        allowed = set(rule.values)
        return present & ~series.astype(str).isin(allowed), int(present.sum())
    if rule.kind == "desen":
        compiled = re.compile(rule.pattern)
        matches = series[present].astype(str).str.slice(0, MAX_TEXT_LENGTH).map(lambda text: bool(compiled.fullmatch(text)))
        return present & ~matches.reindex(frame.index, fill_value=True), int(present.sum())
    left, right = _comparable(frame, rule)
    both = left.notna() & right.notna()
    holds = pd.Series(True, index=frame.index)
    holds[both] = _OPERATORS[rule.operator](left[both], right[both])
    return both & ~holds, int(both.sum())


def evaluate_rules(frame: pd.DataFrame, rules: list[DataRule]) -> tuple[RuleReport, pd.DataFrame]:
    """(rapor, kayıt × kural ihlal matrisi). Uygulanamayan kural hata metniyle raporlanır, sessizce atlanmaz."""
    results: list[RuleResult] = []
    masks: dict[str, pd.Series] = {}
    for rule in rules:
        description = rule.describe()
        try:
            mask, checked = violation_mask(frame, rule)
        except ValueError as exc:
            results.append(RuleResult(rule=rule, description=description, error=str(exc)))
            continue
        violations = int(mask.sum())
        masks[description] = mask
        results.append(RuleResult(
            rule=rule, description=description, checked=checked, violations=violations,
            compliance=1 - violations / checked if checked else None,
        ))
    matrix = pd.DataFrame(masks, index=frame.index) if masks else pd.DataFrame(index=frame.index)
    violating = int(matrix.any(axis=1).sum()) if masks else 0
    score = 100 * (1 - violating / len(frame)) if masks and len(frame) else None
    return RuleReport(rows=len(frame), results=results, violating_rows=violating, compliance_score=score), matrix


def violating_records(frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """İndirilecek tablo: en az bir kuralı ihlal eden kayıtlar ve ihlal ettikleri kurallar."""
    if matrix.empty or not matrix.any(axis=None):
        return frame.iloc[0:0].assign(ihlal_edilen_kurallar=pd.Series(dtype=str))
    hits = matrix[matrix.any(axis=1)]
    labels = hits.apply(lambda row: "; ".join(column for column, hit in row.items() if hit), axis=1)
    records = frame.loc[hits.index].copy()
    records.insert(0, "ihlal_edilen_kurallar", labels)
    records.insert(0, "kayıt_no", hits.index)
    return records


# --- Yerel öneriler ------------------------------------------------------------------------------

def _rare_and_core(series: pd.Series, share: float = RARE_SHARE) -> tuple[list, list]:
    counts = series.value_counts()
    threshold = share * len(series)
    return list(counts[counts < threshold].index), list(counts[counts >= threshold].index)


def suggest_rules(frame: pd.DataFrame) -> list[DataRule]:
    """Veriden deterministik kural önerileri; her birinin gerekçesi yazılıdır. Öneriler onay bekler."""
    suggestions: list[DataRule] = []
    rows = len(frame)
    if rows == 0:
        return suggestions
    datetimes = [column for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])]
    for column in frame.columns:
        series, name = frame[column], str(column)
        tokens = set(name_tokens(column))
        present = series.dropna()
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        if is_identifier_name(column) and is_text_series(series) and len(present) \
                and present.nunique() / len(present) >= 0.99 \
                and not tokens & (_EMAIL_TOKENS | _PHONE_TOKENS | _PERSON_NAME_TOKENS):
            suggestions.append(DataRule(kind="benzersiz", column=name, source="öneri",
                                        reason="Ad bir kimlik/numara sütununa işaret ediyor ve değerlerin %99'undan fazlası tekil."))
            if series.isna().mean() < 0.01:
                suggestions.append(DataRule(kind="boş_olamaz", column=name, source="öneri",
                                            reason="Kimlik sütunu; kayıtların neredeyse tamamında dolu."))
        if numeric and len(present):
            if tokens & _RATIO_TOKENS and present.between(0, 1).mean() >= 0.99:
                suggestions.append(DataRule(kind="aralık", column=name, minimum=0, maximum=1, source="öneri",
                                            reason="Adı oran belirtiyor ve değerlerin en az %99'u 0–1 arasında."))
            elif is_nonnegative_name(column):
                suggestions.append(DataRule(kind="aralık", column=name, minimum=0, source="öneri",
                                            reason="Adı negatif olamayacak bir ölçüye (tutar, fiyat, adet…) işaret ediyor."))
            elif tokens & _RATING_TOKENS and (scale := next(
                (top for top in (5, 10) if present.between(1, top).mean() >= 0.99), None
            )):
                suggestions.append(DataRule(
                    kind="aralık", column=name, minimum=1, maximum=scale, source="öneri",
                    reason=f"Adı puan/skor belirtiyor ve değerlerin en az %99'u 1–{scale} ölçeğinde.",
                ))
            elif (present % 1 == 0).all() and present.nunique() <= 12:
                rare, core = _rare_and_core(present, RARE_SCALE_SHARE)
                core_values = sorted(int(value) for value in core)
                if rare and core_values and core_values == list(range(core_values[0], core_values[-1] + 1)) \
                        and any(value < core_values[0] or value > core_values[-1] for value in rare):
                    suggestions.append(DataRule(
                        kind="aralık", column=name, minimum=core_values[0], maximum=core_values[-1], source="öneri",
                        reason=f"Değerler {core_values[0]}–{core_values[-1]} ölçeğinde; bu aralığın dışındaki değerler "
                               "kayıtların %0,1'inden azında görülüyor.",
                    ))
        if is_text_series(series) and len(present):
            if tokens & _EMAIL_TOKENS:
                suggestions.append(DataRule(kind="desen", column=name, pattern=PATTERN_PRESETS["E-posta"],
                                            source="öneri", reason="Adı e-posta sütununa işaret ediyor."))
            elif tokens & _PHONE_TOKENS:
                suggestions.append(DataRule(kind="desen", column=name, pattern=PATTERN_PRESETS["Türkiye cep telefonu"],
                                            source="öneri", reason="Adı telefon sütununa işaret ediyor."))
            elif not is_identifier_name(column) and present.nunique() <= 15 and rows >= 50:
                rare, core = _rare_and_core(present.astype(str))
                if rare and sum(present.astype(str).isin(rare)) <= 0.02 * len(present):
                    suggestions.append(DataRule(
                        kind="izinli_değerler", column=name, values=sorted(map(str, core)), source="öneri",
                        reason=f"{len(core)} değer kayıtların büyük kısmını oluşturuyor; {len(rare)} değer çok seyrek "
                               "(yazım farkı ya da hatalı giriş olabilir).",
                    ))
    for first_index, earlier in enumerate(datetimes[:10]):
        for later in datetimes[first_index + 1:10]:
            for left, right in ((later, earlier), (earlier, later)):
                rule = DataRule(kind="karşılaştırma", column=str(left), operator=">=", other_column=str(right),
                                source="öneri")
                mask, checked = violation_mask(frame, rule)
                if checked >= 20 and mask.sum() <= 0.01 * checked:
                    suggestions.append(rule.model_copy(update={
                        "reason": f"Birlikte dolu kayıtların en az %99'unda “{left}”, “{right}” tarihinden önce değil."
                    }))
                    break
    return suggestions


# --- LLM önerisi ---------------------------------------------------------------------------------

class RuleSuggestions(BaseModel):
    rules: list[DataRule] = Field(default_factory=list)


class RejectedRule(BaseModel):
    description: str
    reason: str


def _rule_prompt(card: DatasetMetadataCard) -> str:
    return (
        "Bir veri kalitesi uzmanısın. Aşağıdaki KVKK maskeli metadata kartına bakarak iş kuralları öner. "
        "Yalnızca şu türleri kullan: aralık (minimum/maximum), benzersiz, boş_olamaz, izinli_değerler (values), "
        "desen (pattern, Python düzenli ifadesi), karşılaştırma (operator ve other_column). Kod yazma; "
        "yalnızca şemaya uyan JSON döndür. Her kural için Türkçe kısa bir gerekçe (reason) yaz. "
        "Sütun adlarını karttaki gibi aynen kullan.\n\n"
        f"Metadata kartı:\n{card.model_dump_json()}"
    )


class GeminiRuleAdvisor:
    """Gemini'den yalnızca `DataRule` şemasıyla kural önerisi alır; her öneri veriye karşı doğrulanır."""

    def __init__(self, api_key: str | None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model
        self.origin = PlanOrigin(source="yerel kurallar", reason="Gemini anahtarı yok; yalnızca yerel öneriler.")

    def suggest(self, frame: pd.DataFrame, card: DatasetMetadataCard) -> tuple[list[DataRule], list[RejectedRule]]:
        if not self.api_key:
            return [], []
        try:
            response = request_structured(self.api_key, self.model, _rule_prompt(card), RuleSuggestions)
        except Exception as exc:  # SDK, ağ ve doğrulama hataları görünür kılınır, yutulmaz
            self.origin = PlanOrigin(source="yerel kurallar", reason=f"Gemini kural önerisi alınamadı: {failure_reason(exc)}",
                                     error=True)
            return [], []
        accepted: list[DataRule] = []
        rejected: list[RejectedRule] = []
        for rule in response.rules:
            rule = rule.model_copy(update={"source": "Gemini"})
            reason = check_rule(frame, rule)
            if reason:
                rejected.append(RejectedRule(description=rule.describe(), reason=reason))
            else:
                accepted.append(rule)
        self.origin = PlanOrigin(
            source="Gemini", reason=f"{len(accepted)} kural önerildi, {len(rejected)} öneri veriye uymadığı için reddedildi."
        )
        return accepted, rejected
