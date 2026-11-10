"""LLM'e gönderilmek üzere KVKK/PII maskeli veri özeti."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


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

    _email = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
    _phone = re.compile(r"(?<!\d)(?:\+?90[\s.-]?)?(?:0[\s.-]?)?(?:5\d{2}|[2-4]\d{2})[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)")
    _tc = re.compile(r"(?<!\d)\d{11}(?!\d)")

    @classmethod
    def mask_pii(cls, value: Any) -> str:
        text = str(value)
        text = cls._email.sub("[PII_MASKED]", text)
        text = cls._phone.sub("[PII_MASKED]", text)
        return cls._tc.sub("[PII_MASKED]", text)

    @staticmethod
    def _number(value: Any) -> float | None:
        return float(value) if pd.notna(value) else None

    def generate(self, frame: pd.DataFrame) -> DatasetMetadataCard:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Metadata üretimi için pandas.DataFrame gereklidir.")
        row_count, column_count = frame.shape
        columns: list[ColumnMetadata] = []
        for name in frame.columns:
            series = frame[name]
            numeric = pd.api.types.is_numeric_dtype(series)
            numeric_values = pd.to_numeric(series, errors="coerce") if numeric else pd.Series(dtype=float)
            columns.append(
                ColumnMetadata(
                    name=str(name),
                    dtype=str(series.dtype),
                    null_ratio=round(float(series.isna().mean()), 6),
                    nunique_ratio=round(float(series.nunique(dropna=True) / max(row_count, 1)), 6),
                    min=self._number(numeric_values.min()) if numeric else None,
                    max=self._number(numeric_values.max()) if numeric else None,
                    median=self._number(numeric_values.median()) if numeric else None,
                    sample_values=[
                        self.mask_pii(value)
                        for value in series.dropna().head(5).tolist()
                    ],
                )
            )
        return DatasetMetadataCard(row_count=row_count, column_count=column_count, columns=columns)
