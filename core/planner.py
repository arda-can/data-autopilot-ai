"""Gemini tabanlı yapılandırılmış planlama ve güvenli kurallı fallback."""

from __future__ import annotations

import logging
from typing import Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, Field

from core.columns import is_text_dtype_name
from core.metadata import DatasetMetadataCard, MetadataGenerator

AllowedAction = Literal[
    "drop_column", "parse_currency", "parse_datetime", "impute_median",
    "impute_mode", "cap_outliers_iqr", "cast_categorical",
    "strip_whitespace", "keep_as_is",
]

# Veriden istatistik öğrenen aksiyonlar. Otopilot bunları ön işlemede çalıştırmaz; aynı işi
# modelleme hattı yalnızca eğitim bölmesi üzerinde yapar (.cursorrules madde 4).
STATISTICAL_ACTIONS: frozenset[str] = frozenset({"impute_median", "impute_mode", "cap_outliers_iqr"})


logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class PlanOrigin(BaseModel):
    """Planın nereden geldiği; arayüz ve rapor bunu gösterir. LLM şemasının parçası değildir."""

    source: Literal["Gemini", "yerel kurallar"]
    reason: str
    error: bool = False


def request_structured(api_key: str, model: str, contents: str, schema: type[SchemaT]) -> SchemaT:
    """Gemini'den şemaya uyan JSON ister ve doğrular. Hata yutulmaz; çağıran yakalar ve görünür kılar."""
    from google import genai
    from google.genai import types

    response = genai.Client(api_key=api_key).models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema),
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    # `parsed` yoksa metin bir JSON belgesidir: model_validate(str) her zaman reddederdi.
    return schema.model_validate_json(response.text)


def failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:200]}"


class ColumnPlan(BaseModel):
    column: str
    action: AllowedAction
    reason: str


class AnalysisPlan(BaseModel):
    domain: str
    suggested_target: str | None = None
    problem_type: str
    actions: list[ColumnPlan] = Field(default_factory=list)


class GeminiPlanner:
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model
        self.origin = PlanOrigin(source="yerel kurallar", reason="Henüz plan üretilmedi.")

    def create_plan(self, frame: pd.DataFrame, metadata: DatasetMetadataCard | None = None) -> AnalysisPlan:
        card = metadata or MetadataGenerator().generate(frame)
        if not self.api_key:
            self.origin = PlanOrigin(source="yerel kurallar", reason="Gemini API anahtarı girilmedi.")
            return self._fallback(card)
        try:
            plan = request_structured(
                self.api_key, self.model,
                "Aşağıdaki KVKK maskeli metadata kartına göre Türkçe, güvenli ve "
                "deterministik bir veri temizleme planı üret. Yalnızca izin verilen "
                "aksiyonları kullan; ham veri isteme veya üretme. Eksik doldurma ve "
                "aykırı değer sınırlama önerme; bunlar modelleme hattında yapılır.\n"
                f"{card.model_dump_json()}",
                AnalysisPlan,
            )
        except Exception as exc:  # SDK; ağ, kota, anahtar ve şema hataları için farklı türler fırlatır
            # Hata yutulmaz: loglanır ve planın yerel kurallardan geldiği nedeniyle birlikte gösterilir.
            logger.warning("Gemini planı alınamadı (%s); yerel kurallar kullanıldı.", failure_reason(exc))
            self.origin = PlanOrigin(
                source="yerel kurallar", error=True,
                reason=f"Gemini planı alınamadı ({failure_reason(exc)}); yerel kurallar kullanıldı.",
            )
            return self._fallback(card)
        self.origin = PlanOrigin(source="Gemini", reason=f"{self.model} planı şema doğrulamasından geçti.")
        return plan

    def _fallback(self, card: DatasetMetadataCard) -> AnalysisPlan:
        actions: list[ColumnPlan] = []
        for column in card.columns:
            if column.null_ratio > 0.90:
                action: AllowedAction = "drop_column"
                reason = "Sütunun eksik değer oranı %90 üzerinde."
            elif is_text_dtype_name(column.dtype):
                samples = " ".join(column.sample_values)
                if any(token in samples for token in ("TL", "₺", "$", "€", "%")):
                    action = "parse_currency"
                    reason = "Örnek değerler para veya sayısal metin biçiminde."
                else:
                    action = "strip_whitespace"
                    reason = "Metin değerlerinde standart boşluk temizliği uygulanabilir."
            elif column.null_ratio > 0:
                action = "keep_as_is"
                reason = "Eksik değerler korunur; doldurma yalnızca modelleme hattında eğitim verisinden öğrenilir."
            else:
                action = "keep_as_is"
                reason = "Belirgin bir deterministik dönüşüm gerekmiyor."
            actions.append(ColumnPlan(column=column.name, action=action, reason=reason))
        return AnalysisPlan(
            domain="Genel yapılandırılmış veri",
            suggested_target=None,
            problem_type="Belirlenemedi",
            actions=actions,
        )
