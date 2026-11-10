"""Veri kalitesi ölçümü ve Pydantic rapor şemaları."""

from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel, Field


class ColumnHealth(BaseModel):
    sütun: str
    veri_tipi: str
    eksik_oranı: float
    benzersiz_değer: int
    sabit_mi: bool
    olası_id_mi: bool
    sayısal_görünümlü_kirli_mi: bool


class HealthReport(BaseModel):
    kalite_skoru: float = Field(ge=0, le=100)
    satır_sayısı: int
    sütun_sayısı: int
    eksik_veri_oranı: float
    mükerrer_satır_sayısı: int
    yüksek_eksik_sütunlar: list[str]
    sabit_sütunlar: list[str]
    olası_id_sütunları: list[str]
    kirli_sayısal_sütunlar: list[str]
    sorunlar: list[str]
    sütun_raporları: list[ColumnHealth]


class DataHealthAnalyzer:
    """DataFrame üzerinde yalnızca gözlemlenebilir kalite metrikleri hesaplar."""

    _numeric_pattern = re.compile(r"^[\s\d.,%₺$€TL-]+$", re.IGNORECASE)

    def analyze(self, frame: pd.DataFrame) -> HealthReport:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Analiz için pandas.DataFrame gereklidir.")
        rows, columns = frame.shape
        missing_ratio = float(frame.isna().mean().mean()) if rows and columns else 0.0
        duplicates = int(frame.duplicated().sum())
        constant = [str(c) for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
        high_null = [str(c) for c in frame.columns if frame[c].isna().mean() > 0.90]
        possible_ids: list[str] = []
        dirty_numeric: list[str] = []
        column_reports: list[ColumnHealth] = []

        for column in frame.columns:
            series = frame[column]
            name = str(column)
            unique = int(series.nunique(dropna=True))
            is_id = unique == rows and rows > 0 and (
                name.lower().endswith(("id", "_id")) or "id" in name.lower()
            )
            if is_id:
                possible_ids.append(name)
            dirty = False
            if series.dtype == "object":
                values = series.dropna().astype(str).str.strip()
                numeric_like = values.str.replace(r"[\d\s.,%₺$€TL-]", "", regex=True).eq("").mean()
                dirty = bool(len(values) and numeric_like >= 0.60 and not pd.to_numeric(values, errors="coerce").notna().all())
            if dirty:
                dirty_numeric.append(name)
            column_reports.append(
                ColumnHealth(
                    sütun=name,
                    veri_tipi=str(series.dtype),
                    eksik_oranı=round(float(series.isna().mean()), 4),
                    benzersiz_değer=unique,
                    sabit_mi=name in constant,
                    olası_id_mi=is_id,
                    sayısal_görünümlü_kirli_mi=dirty,
                )
            )

        penalties = min(100.0, missing_ratio * 45 + (len(constant) / max(columns, 1)) * 15
                        + (len(high_null) / max(columns, 1)) * 15
                        + (duplicates / max(rows, 1)) * 25)
        score = round(max(0.0, 100.0 - penalties), 2)
        issues = []
        if missing_ratio:
            issues.append(f"Toplam eksik veri oranı %{missing_ratio * 100:.2f}.")
        if constant:
            issues.append(f"Sabit sütunlar: {', '.join(constant)}.")
        if high_null:
            issues.append(f"%90 üzeri boş sütunlar: {', '.join(high_null)}.")
        if duplicates:
            issues.append(f"{duplicates} mükerrer satır tespit edildi.")
        if possible_ids:
            issues.append(f"Olası kimlik sütunları: {', '.join(possible_ids)}.")
        if dirty_numeric:
            issues.append(f"Kirli sayısal sütunlar: {', '.join(dirty_numeric)}.")
        return HealthReport(
            kalite_skoru=score, satır_sayısı=rows, sütun_sayısı=columns,
            eksik_veri_oranı=round(missing_ratio, 4), mükerrer_satır_sayısı=duplicates,
            yüksek_eksik_sütunlar=high_null, sabit_sütunlar=constant,
            olası_id_sütunları=possible_ids, kirli_sayısal_sütunlar=dirty_numeric,
            sorunlar=issues or ["Belirgin bir kalite sorunu tespit edilmedi."],
            sütun_raporları=column_reports,
        )
