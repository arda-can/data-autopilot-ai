"""Önceden tanımlı, deterministik veri temizleme eylemleri."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.columns import (
    is_identifier_name,
    is_text_series,
    numeric_text_ratio,
    parse_datetime_text,
    parse_numeric_text,
    text_columns,
)
from core.formatting import decimal_tr


class AuditLog(BaseModel):
    column: str
    action: str
    reason: str
    affected_count: int
    before_sample: list[Any] = Field(default_factory=list)
    after_sample: list[Any] = Field(default_factory=list)


class ActionExecutor:
    def __init__(self) -> None:
        self.audit_logs: list[AuditLog] = []

    def _record(self, column: str, action: str, reason: str, count: int,
                before: pd.Series | list[Any], after: pd.Series | list[Any]) -> None:
        self.audit_logs.append(AuditLog(
            column=column, action=action, reason=reason, affected_count=int(count),
            before_sample=list(before[:5]), after_sample=list(after[:5]),
        ))

    def drop_constant_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in list(result.columns):
            if result[column].nunique(dropna=False) <= 1:
                values = result[column].head(5)
                result = result.drop(columns=[column])
                self._record(str(column), "Sabit sütunu kaldır", "Tek değer içerdiği için kaldırıldı.",
                             len(frame), values, [])
        return result

    def drop_column(self, frame: pd.DataFrame, column: str, reason: str = "") -> pd.DataFrame:
        result = frame.copy()
        if column in result.columns:
            missing = float(result[column].isna().mean())
            explanation = (
                f"%90'dan fazlası boş olduğu için otomatik kaldırıldı (eksik oranı %{decimal_tr(missing * 100)})."
                if missing > 0.90 else f"Plan aksiyonu uygulandı: {reason or 'Sütunun kaldırılması önerildi.'}"
            )
            self._record(column, "Sütunu kaldır", explanation,
                         len(result), result[column].head(5), [])
            return result.drop(columns=[column])
        return result

    def drop_high_null_columns(self, frame: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
        result = frame.copy()
        for column in list(result.columns):
            if result[column].isna().mean() > threshold:
                self._record(str(column), "Boş sütunu kaldır", f"%{decimal_tr(threshold * 100, 0)}'dan fazlası boş olduğu için otomatik kaldırıldı.",
                             int(result[column].isna().sum()), result[column].head(5), [])
                result = result.drop(columns=[column])
        return result

    def remove_duplicates(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        mask = result.duplicated()
        count = int(mask.sum())
        if count:
            self._record("Tüm sütunlar", "Mükerrerleri kaldır", "Tamamen aynı satırlar kaldırıldı.",
                         count, [" | ".join(map(str, row)) for row in result.loc[mask].head(5).itertuples(index=False)], [])
            result = result.loc[~mask].reset_index(drop=True)
        return result

    def strip_whitespace(self, frame: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        result = frame.copy()
        columns = [column] if column else text_columns(result)
        for current_column in columns:
            if current_column not in result.columns:
                continue
            original = result[current_column].copy()
            result[current_column] = result[current_column].map(lambda value: value.strip() if isinstance(value, str) else value)
            changed = int((original.fillna("").astype(str) != result[current_column].fillna("").astype(str)).sum())
            if changed:
                self._record(str(current_column), "Boşlukları temizle", "Metin başı ve sonundaki boşluklar kaldırıldı.",
                             changed, original, result[current_column])
        return result

    def impute_median(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        result = frame.copy()
        if column not in result.columns or not pd.api.types.is_numeric_dtype(result[column]):
            return result
        missing = result[column].isna()
        if missing.any():
            median = result[column].median()
            if pd.isna(median):
                return result
            before = result[column].copy()
            result[column] = result[column].fillna(median)
            self._record(column, "Eksikleri medyan ile doldur", "Sayısal eksikler sütun medyanı ile dolduruldu.",
                         int(missing.sum()), before, result[column])
        return result

    def impute_mode(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        result = frame.copy()
        if column not in result.columns:
            return result
        missing = result[column].isna()
        modes = result[column].mode(dropna=True)
        if missing.any() and not modes.empty:
            before = result[column].copy()
            result[column] = result[column].fillna(modes.iloc[0])
            self._record(column, "Eksikleri mod ile doldur", "Eksikler en sık görülen değer ile dolduruldu.",
                         int(missing.sum()), before, result[column])
        return result

    def cast_categorical(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        result = frame.copy()
        if column in result.columns and result[column].dtype != "category":
            before = result[column].copy()
            result[column] = result[column].astype("category")
            self._record(column, "Kategorik türe dönüştür", "Düşük kardinaliteli sütun kategorik yapıya çevrildi.",
                         len(result), before, result[column])
        return result

    def clean_currency_and_numeric_strings(self, frame: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        """Tamamı sayı biçiminde olan metin sütunlarını sayıya çevirir.

        Otomatik modda kimlik adlı sütunlara dokunmaz ve değerlerin en az %90'ı sayı
        biçiminde olmalıdır; açıkça sütun verildiğinde eşik %60'tır.
        """
        result = frame.copy()
        explicit = column is not None
        threshold = 0.60 if explicit else 0.90
        columns = [column] if explicit else text_columns(result)
        for current_column in columns:
            if current_column not in result.columns or not is_text_series(result[current_column]):
                continue
            if not explicit and is_identifier_name(current_column):
                continue
            series = result[current_column]
            present = int(series.notna().sum())
            if not present or numeric_text_ratio(series) < threshold - 0.10:
                continue
            numbers, valid = parse_numeric_text(series)
            converted = int(valid.sum())
            if converted / present < threshold:
                continue
            reason = "Para birimi ve yerel sayı biçimi sayısala çevrildi."
            if converted < present:
                reason += f" Sayı biçiminde olmayan {present - converted} değer boş bırakıldı."
            result[current_column] = numbers
            self._record(str(current_column), "Para ve sayısal metinleri dönüştür", reason,
                         converted, series, result[current_column])
        return result

    def parse_datetime_columns(self, frame: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
        result = frame.copy()
        columns = [column] if column else text_columns(result)
        for current_column in columns:
            if current_column not in result.columns:
                continue
            series = result[current_column]
            if pd.api.types.is_datetime64_any_dtype(series):
                continue
            parsed = parse_datetime_text(series)
            if parsed is None:
                continue
            valid = series.notna() & parsed.notna()
            result[current_column] = parsed
            self._record(str(current_column), "Tarihleri dönüştür", "Tarih biçimi standartlaştırıldı.",
                         int(valid.sum()), series, result[current_column])
        return result

    def cap_outliers_iqr(self, frame: pd.DataFrame, multiplier: float = 1.5,
                         column: str | None = None) -> pd.DataFrame:
        result = frame.copy()
        columns = [column] if column else list(result.select_dtypes(include=[np.number]).columns)
        for current_column in columns:
            if current_column not in result.columns:
                continue
            if is_identifier_name(current_column):
                continue
            series = result[current_column]
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            mask = (series < lower) | (series > upper)
            if mask.any():
                before = series.copy()
                result[current_column] = series.clip(lower, upper)
                self._record(str(current_column), "Aykırı değerleri sınırla", "IQR yöntemiyle uç değerler sınırlandı.",
                             int(mask.sum()), before, result[current_column])
        return result

    def execute_plan(self, frame: pd.DataFrame, approved_actions: list[Any]) -> pd.DataFrame:
        """Onaylanan planı beyaz listedeki deterministik metotlarla sırayla uygular."""
        result = frame.copy()
        for item in approved_actions:
            if hasattr(item, "column") and hasattr(item, "action"):
                column, action = item.column, item.action
                reason = getattr(item, "reason", "")
                approved = True
            else:
                column = str(item.get("column", ""))
                action = str(item.get("action", "keep_as_is"))
                reason = str(item.get("reason", ""))
                approved = bool(item.get("onay", item.get("approved", True)))
            if not approved or action == "keep_as_is":
                continue
            if action == "drop_column":
                result = self.drop_column(result, column, reason)
            elif action == "parse_currency":
                result = self.clean_currency_and_numeric_strings(result, column)
            elif action == "parse_datetime":
                result = self.parse_datetime_columns(result, column)
            elif action == "impute_median":
                result = self.impute_median(result, column)
            elif action == "impute_mode":
                result = self.impute_mode(result, column)
            elif action == "cap_outliers_iqr":
                result = self.cap_outliers_iqr(result, column=column)
            elif action == "cast_categorical":
                result = self.cast_categorical(result, column)
            elif action == "strip_whitespace":
                result = self.strip_whitespace(result, column)
            else:
                raise ValueError(f"Desteklenmeyen plan aksiyonu: {action}")
        return result
