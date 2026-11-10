"""LLM'e gönderilmek üzere KVKK/PII maskeli veri özeti.

Koruma üç katmanlıdır:
1. Hassas adlı sütunlar (kimlik, ad, telefon, adres...) için örnekler ve istatistikler tümüyle gizlenir.
2. Neredeyse her değeri benzersiz olan metin sütunlarında içerik yerine yalnızca biçim gönderilir.
3. Kalan tüm metinlerde e-posta, telefon, TC kimlik, IBAN ve kart numarası desenleri maskelenir.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from core.columns import is_identifier_name, is_text_series, name_tokens

PII_MASK = "[PII_MASKED]"
_SAMPLE_COUNT = 5
_SENSITIVE_TOKENS = frozenset({
    "email", "eposta", "mail", "telefon", "tel", "phone", "gsm", "cep", "mobile", "iban",
    "adres", "address", "kart", "card", "pasaport", "passport", "doğum", "dogum", "birth",
    "birthday", "dob", "ssn", "şifre", "sifre", "password", "plaka",
})


class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    null_ratio: float
    nunique_ratio: float
    min: float | None = None
    max: float | None = None
    median: float | None = None
    sample_values: list[str] = Field(default_factory=list)


class DatasetMetadataCard(BaseModel):
    row_count: int
    column_count: int
    columns: list[ColumnMetadata]


class MetadataGenerator:
    """Ham satırları göndermeden veri setinin kompakt profilini üretir."""

    _patterns = (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),  # e-posta
        re.compile(r"\bTR\d{2}(?:[ -]?\d{4}){5}[ -]?\d{2}\b", re.IGNORECASE),  # TR IBAN
        re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){3,7}(?:[ -]?[A-Z0-9]{1,3})?\b"),  # diğer IBAN
        re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,7}(?!\d)"),  # kart numarası
        re.compile(r"(?<!\d)\d{11}(?!\d)"),  # TC kimlik
        re.compile(  # telefon
            r"(?<!\d)(?:\+?90[\s.-]?)?(?:0[\s.-]?)?\(?(?:5\d{2}|[2-4]\d{2})\)?[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"
        ),
    )

    @classmethod
    def mask_pii(cls, value: Any) -> str:
        text = str(value)
        for pattern in cls._patterns:
            text = pattern.sub(PII_MASK, text)
        return text

    @staticmethod
    def is_sensitive_name(name: object) -> bool:
        """Sütun adı kişisel veri ya da kimlik taşıdığına işaret ediyor mu?"""
        return is_identifier_name(name) or bool(set(name_tokens(name)) & _SENSITIVE_TOKENS)

    @staticmethod
    def _shape(value: Any) -> str:
        """İçeriği gizleyip biçimi korur: `Ayşe 12` → `Aaaa 99`."""
        text = re.sub(r"[^\W\d_]", lambda match: "A" if match.group().isupper() else "a", str(value)[:40])
        return re.sub(r"\d", "9", text)

    @staticmethod
    def _looks_like_numeric_identifier(values: pd.Series) -> bool:
        """10+ haneli tam sayılar (telefon, TC, kart) ölçüm değil kimliktir; min/max bile kişiyi ifşa eder."""
        if values.empty:
            return False
        return bool((values % 1 == 0).all() and values.abs().max() >= 1e9)

    @staticmethod
    def _number(value: Any) -> float | None:
        return float(value) if pd.notna(value) else None

    @staticmethod
    def _sample_text(value: Any) -> str:
        if isinstance(value, float) and not value.is_integer():
            return f"{value:.6g}"
        return str(value)

    def generate(self, frame: pd.DataFrame) -> DatasetMetadataCard:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Metadata üretimi için pandas.DataFrame gereklidir.")
        row_count, column_count = frame.shape
        columns: list[ColumnMetadata] = []
        for name in frame.columns:
            series = frame[name]
            present = series.dropna()
            samples = present.head(_SAMPLE_COUNT).tolist()
            unique_ratio = float(series.nunique(dropna=True) / max(row_count, 1))
            numeric = pd.api.types.is_numeric_dtype(series)
            numeric_values = pd.to_numeric(present, errors="coerce") if numeric else pd.Series(dtype=float)
            hidden = self.is_sensitive_name(name) or (
                numeric and self._looks_like_numeric_identifier(numeric_values)
            )
            if hidden:
                sample_values = [PII_MASK] * len(samples)
            elif is_text_series(series) and row_count >= 20 and unique_ratio >= 0.5:
                sample_values = [self._shape(value) for value in samples]
            else:
                sample_values = [self.mask_pii(self._sample_text(value)) for value in samples]
            show_stats = numeric and not hidden
            columns.append(
                ColumnMetadata(
                    name=self.mask_pii(name),
                    dtype=str(series.dtype),
                    null_ratio=round(float(series.isna().mean()), 6),
                    nunique_ratio=round(unique_ratio, 6),
                    min=self._number(numeric_values.min()) if show_stats else None,
                    max=self._number(numeric_values.max()) if show_stats else None,
                    median=self._number(numeric_values.median()) if show_stats else None,
                    sample_values=sample_values,
                )
            )
        return DatasetMetadataCard(row_count=row_count, column_count=column_count, columns=columns)
