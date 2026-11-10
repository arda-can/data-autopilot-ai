"""Gemini tabanlı yapılandırılmış planlama ve güvenli kurallı fallback."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from core.metadata import DatasetMetadataCard, MetadataGenerator

AllowedAction = Literal[
    "drop_column", "parse_currency", "parse_datetime", "impute_median",
    "impute_mode", "cap_outliers_iqr", "cast_categorical",
    "strip_whitespace", "keep_as_is",
]


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

    def create_plan(self, frame: pd.DataFrame, metadata: DatasetMetadataCard | None = None) -> AnalysisPlan:
        card = metadata or MetadataGenerator().generate(frame)
        if not self.api_key:
            return self._fallback(card)
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=(
                    "Aşağıdaki KVKK maskeli metadata kartına göre Türkçe, güvenli ve "
                    "deterministik bir veri temizleme planı üret. Yalnızca izin verilen "
                    "aksiyonları kullan; ham veri isteme veya üretme.\n"
                    f"{card.model_dump_json()}"
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisPlan,
                ),
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, AnalysisPlan):
                return parsed
            return AnalysisPlan.model_validate(response.text)
        except Exception:
            # API erişimi, anahtar veya SDK sürümü sorunlarında güvenli yerel plan.
            return self._fallback(card)

    def _fallback(self, card: DatasetMetadataCard) -> AnalysisPlan:
        actions: list[ColumnPlan] = []
        for column in card.columns:
            if column.null_ratio > 0.90:
                action: AllowedAction = "drop_column"
                reason = "Sütunun eksik değer oranı %90 üzerinde."
            elif column.dtype in {"object", "string"}:
                samples = " ".join(column.sample_values)
                if any(token in samples for token in ("TL", "₺", "$", "€", "%")):
                    action = "parse_currency"
                    reason = "Örnek değerler para veya sayısal metin biçiminde."
                elif column.null_ratio > 0:
                    action = "impute_mode"
                    reason = "Metin sütunundaki eksik değerler mod ile doldurulabilir."
                else:
                    action = "strip_whitespace"
                    reason = "Metin değerlerinde standart boşluk temizliği uygulanabilir."
            elif column.null_ratio > 0:
                action = "impute_median"
                reason = "Sayısal eksik değerler medyan ile doldurulabilir."
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
