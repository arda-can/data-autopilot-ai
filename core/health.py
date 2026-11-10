"""Veri kalitesi ölçümü ve Pydantic rapor şemaları."""

from __future__ import annotations

import unicodedata

import pandas as pd
from pydantic import BaseModel, Field

from core.columns import is_identifier_name, is_text_series, name_tokens, numeric_text_ratio
from core.formatting import decimal_tr


class DomainFinding(BaseModel):
    column: str
    count: int
    rule: str = "Negatif değer içermemeli."


class CategoryVariantFinding(BaseModel):
    column: str
    group_count: int
    affected_count: int
    # Ham değerler rapora taşınmaz; yalnızca toplulaştırılmış sayılar tutulur.


_MAX_CATEGORY_VARIANT_VALUES = 5_000

_NONNEGATIVE_TOKENS = frozenset({
    "yaş", "yas", "age", "fiyat", "price", "adet", "miktar", "quantity", "qty",
    "count", "sayı", "sayisi", "sayısı", "ücret", "ucret", "tutar", "harcama",
    "cost", "fee", "stok", "stock", "freight", "taksit", "installment", "installments",
    "payment", "ödeme", "odeme", "weight", "ağırlık", "agirlik",
})


def is_nonnegative_name(name: object) -> bool:
    """Sütun adı, doğası gereği negatif olamayan bir ölçüye (tutar, taksit, adet, yaş) işaret ediyor mu?"""
    return bool(set(name_tokens(name)) & _NONNEGATIVE_TOKENS)


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
    domain_findings: list[DomainFinding] = Field(default_factory=list)
    category_variants: list[CategoryVariantFinding] = Field(default_factory=list)


class OutlierFinding(BaseModel):
    """IQR sınırlarının dışında kalan değerlerin özeti. Yalnızca raporlanır, veri değiştirilmez."""

    column: str
    count: int
    ratio: float
    lower_bound: float
    upper_bound: float
    min: float
    max: float


class DataHealthAnalyzer:
    """DataFrame üzerinde yalnızca gözlemlenebilir kalite metrikleri hesaplar."""

    def find_domain_violations(self, frame: pd.DataFrame) -> list[DomainFinding]:
        findings = []
        for column in frame.select_dtypes(include="number").columns:
            if is_identifier_name(column) or not is_nonnegative_name(column):
                continue
            count = int((frame[column] < 0).sum())
            if count:
                findings.append(DomainFinding(column=str(column), count=count))
        return findings

    def find_category_variants(self, frame: pd.DataFrame) -> list[CategoryVariantFinding]:
        findings = []
        for column in frame.columns:
            series = frame[column]
            if is_identifier_name(column) or not (is_text_series(series) or isinstance(series.dtype, pd.CategoricalDtype)):
                continue
            # Çok benzersiz değerli sütunlar (serbest metin, hash) kategori değildir; Python döngüsü
            # milyon satırda dakikalar sürer, bu yüzden hiç taranmaz.
            if series.nunique(dropna=True) > _MAX_CATEGORY_VARIANT_VALUES:
                continue
            counts = series.dropna().value_counts()
            groups: dict[str, list[int]] = {}
            for value, count in counts.items():
                if not isinstance(value, str) or not count:
                    continue
                folded = unicodedata.normalize("NFKD", value.casefold())
                key = "".join(char for char in folded if not unicodedata.combining(char))
                groups.setdefault(key, []).append(int(count))
            duplicates = [counts for counts in groups.values() if len(counts) > 1]
            if duplicates:
                findings.append(CategoryVariantFinding(
                    column=str(column), group_count=len(duplicates),
                    affected_count=sum(sum(counts) for counts in duplicates),
                ))
        return findings

    def find_outliers(self, frame: pd.DataFrame, multiplier: float = 1.5) -> list[OutlierFinding]:
        """Sürekli sayısal sütunlardaki şüpheli uç değerleri bulur; kimlik ve ayrık ölçekleri atlar."""
        findings: list[OutlierFinding] = []
        for column in frame.select_dtypes(include="number").columns:
            values = frame[column].dropna()
            if is_identifier_name(column) or values.nunique() < 10:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            if not iqr > 0:
                continue
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            # Negatif olamayan bir ölçü için "beklenen aralık" sıfırın altına inemez: ya ad bunu söyler
            # (yaş, tutar, taksit) ya da veride hiç negatif değer yoktur.
            if lower < 0 and (is_nonnegative_name(column) or values.min() >= 0):
                lower = 0.0
            count = int(((values < lower) | (values > upper)).sum())
            if count:
                findings.append(OutlierFinding(
                    column=str(column), count=count, ratio=round(count / len(values), 4),
                    lower_bound=float(lower), upper_bound=float(upper),
                    min=float(values.min()), max=float(values.max()),
                ))
        return findings

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
            is_id = unique == rows and rows > 0 and is_identifier_name(name)
            if is_id:
                possible_ids.append(name)
            dirty = bool(
                is_text_series(series)
                and not is_identifier_name(name)
                and numeric_text_ratio(series) >= 0.60
            )
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
            issues.append(f"Toplam eksik veri oranı %{decimal_tr(missing_ratio * 100, 2)}.")
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
        domain_findings = self.find_domain_violations(frame)
        category_variants = self.find_category_variants(frame)
        for finding in domain_findings:
            issues.append(f"{finding.column}: {finding.count} negatif değer şüpheli; alan kuralı gereği negatif olmamalı. Değerler değiştirilmedi.")
        for finding in category_variants:
            issues.append(f"{finding.column}: aksan veya büyük/küçük harf farkıyla oluşmuş {finding.group_count} olası kategori kopyası grubu ({finding.affected_count} kayıt). Birleştirme öncesinde incelenmesi önerilir; değerler değiştirilmedi.")
        return HealthReport(
            kalite_skoru=score, satır_sayısı=rows, sütun_sayısı=columns,
            eksik_veri_oranı=round(missing_ratio, 4), mükerrer_satır_sayısı=duplicates,
            yüksek_eksik_sütunlar=high_null, sabit_sütunlar=constant,
            olası_id_sütunları=possible_ids, kirli_sayısal_sütunlar=dirty_numeric,
            sorunlar=issues or ["Belirgin bir kalite sorunu tespit edilmedi."],
            sütun_raporları=column_reports,
            domain_findings=domain_findings, category_variants=category_variants,
        )
