"""Güvenli doğal dil veri sorgulama ve Plotly görselleştirme motoru."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field

from core.metadata import DatasetMetadataCard, MetadataGenerator
from core.formatting import localize_figure


class StructuredDataQuery(BaseModel):
    operation: Literal[
        "groupby_agg", "value_counts", "top_n", "describe",
        "correlation", "histogram",
    ]
    group_col: str | None = None
    target_col: str | None = None
    agg_func: Literal["mean", "sum", "count", "min", "max", "median"] = "mean"
    chart_type: Literal[
        "bar", "pie", "line", "histogram", "box", "scatter", "heatmap",
    ] = "bar"
    limit: int = Field(default=10, ge=1, le=100)
    explanation: str


class DataChatResult(BaseModel):
    query: StructuredDataQuery
    summary: str


class DataChatEngine:
    """LLM'i yalnızca doğrulanmış sorgu planı üretmek için kullanan motor."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model

    def answer(
        self,
        frame: pd.DataFrame,
        question: str,
        metadata: DatasetMetadataCard | None = None,
    ) -> tuple[pd.DataFrame, str, object, StructuredDataQuery]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Sorgulama için boş olmayan bir pandas DataFrame gereklidir.")
        if not question.strip():
            raise ValueError("Lütfen bir veri sorusu yazın.")
        card = metadata or MetadataGenerator().generate(frame)
        query = self._plan(question, card, frame)
        result, summary, figure = self._execute(frame, query)
        return result, summary, localize_figure(figure), query

    def _plan(
        self,
        question: str,
        card: DatasetMetadataCard,
        frame: pd.DataFrame,
    ) -> StructuredDataQuery:
        if self.api_key:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model,
                    contents=self._build_prompt(question, card),
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=StructuredDataQuery,
                    ),
                )
                parsed = getattr(response, "parsed", None)
                if isinstance(parsed, StructuredDataQuery):
                    return self._validate_columns(parsed, frame)
                return self._validate_columns(
                    StructuredDataQuery.model_validate_json(response.text), frame
                )
            except Exception:
                pass
        return self._fallback(question, frame)

    @staticmethod
    def _build_prompt(question: str, card: DatasetMetadataCard) -> str:
        """LLM'e giden metin: soru da metadata gibi PII maskesinden geçer."""
        return (
            "Türkçe kullanıcı sorusunu yalnızca aşağıdaki KVKK/PII maskeli "
            "metadata ile analiz et. Ham veri isteme veya üretme. Sadece "
            "doğrulanmış StructuredDataQuery JSON'u döndür.\n"
            f"Kullanıcı sorusu: {MetadataGenerator.mask_pii(question)}\n"
            f"Metadata: {card.model_dump_json()}"
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9çğıöşü]", "", value.lower().replace("İ", "i"))

    def _find_column(self, question: str, columns: list[str], numeric_only: bool = False,
                     frame: pd.DataFrame | None = None) -> str | None:
        normalized = self._normalize(question)
        candidates = [
            column for column in columns
            if self._normalize(str(column)) in normalized
            and (not numeric_only or frame is None or pd.api.types.is_numeric_dtype(frame[column]))
        ]
        return max(candidates, key=lambda value: len(str(value)), default=None)

    def _fallback(self, question: str, frame: pd.DataFrame) -> StructuredDataQuery:
        numeric = [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        target = self._find_column(question, [str(c) for c in frame.columns], True, frame)
        group = self._find_column(question, categorical or [str(c) for c in frame.columns])
        lowered = question.lower()
        limit_match = re.search(r"\b(\d{1,3})\b", lowered)
        limit = max(1, min(int(limit_match.group(1)), 100)) if limit_match else 10
        if "korelasyon" in lowered:
            operation, chart, explanation = "correlation", "heatmap", "Sayısal sütunlar arasındaki korelasyon gösteriliyor."
        elif "histogram" in lowered:
            operation, chart, explanation = "histogram", "histogram", "Sayısal değerlerin histogram dağılımı gösteriliyor."
        elif "dağılım" in lowered:
            operation, chart, explanation = "value_counts", "pie", "Değerlerin dağılımı gösteriliyor."
        elif any(word in lowered for word in ("en yüksek", "en çok", "top")):
            operation, chart, explanation = "top_n", "bar", f"En yüksek {limit} değer gösteriliyor."
        elif any(word in lowered for word in ("sayı", "kaç", "frekans")):
            operation, chart, explanation = "value_counts", "bar", "Değerlerin sayısal dağılımı gösteriliyor."
        elif any(word in lowered for word in ("açıklama", "özet", "istatistik", "describe")):
            operation, chart, explanation = "describe", "bar", "Temel istatistiksel özet hazırlanıyor."
        else:
            operation, chart, explanation = "groupby_agg", "bar", "Gruplara göre ortalama hesaplanıyor."
        agg = "sum" if "toplam" in lowered else "mean"
        if "medyan" in lowered:
            agg = "median"
        elif "minimum" in lowered or "en düşük" in lowered:
            agg = "min"
        elif "maksimum" in lowered:
            agg = "max"
        if not target:
            target = numeric[0] if numeric else (str(frame.columns[0]))
        if not group and categorical:
            group = categorical[0]
        if operation == "correlation":
            target = None
        if operation in {"value_counts", "top_n"} and not group:
            group = str(frame.columns[0])
        return StructuredDataQuery(
            operation=operation,
            group_col=group,
            target_col=target,
            agg_func=agg,
            chart_type=chart,
            limit=limit,
            explanation=explanation,
        )

    @staticmethod
    def _validate_columns(query: StructuredDataQuery, frame: pd.DataFrame) -> StructuredDataQuery:
        columns = {str(column) for column in frame.columns}
        for name in (query.group_col, query.target_col):
            if name is not None and name not in columns:
                raise ValueError(f"Plan sütun içermiyor veya geçersiz sütun seçti: {name}")
        return query

    def _execute(self, frame: pd.DataFrame, query: StructuredDataQuery) -> tuple[pd.DataFrame, str, object]:
        group = query.group_col
        target = query.target_col
        if query.operation == "correlation":
            result = frame.select_dtypes(include="number").corr().round(3)
            return result, query.explanation, px.imshow(result, text_auto=True, color_continuous_scale="RdBu_r")
        if query.operation == "describe":
            result = frame.describe(include="all").transpose().reset_index(names="Sütun")
            return result, query.explanation, px.bar(result, x="Sütun", y="count", title="Sütun istatistikleri")
        if query.operation == "histogram":
            histogram_col = target or group
            if histogram_col is None or histogram_col not in frame:
                raise ValueError("Histogram için geçerli bir hedef veya grup sütunu gerekir.")
            values = frame[histogram_col].dropna()
            if pd.api.types.is_numeric_dtype(values):
                bin_count = min(10, max(1, values.nunique()))
                bins = pd.cut(values, bins=bin_count, include_lowest=True, duplicates="drop")
                result = bins.value_counts(sort=False).rename_axis("Aralık").reset_index(name="Sayı")
                # Yalnızca sunum etiketleri değişir; pd.cut sınırları ve frekanslar korunur.
                result["Aralık"] = result["Aralık"].map(
                    lambda interval: f"({interval.left:,.3f} … {interval.right:,.3f}]".translate(
                        str.maketrans({",": ".", ".": ","})
                    )
                )
            else:
                result = values.value_counts().rename_axis(histogram_col).reset_index(name="Sayı")
            figure = px.histogram(
                frame[[histogram_col]].dropna(),
                x=histogram_col,
                title=query.explanation,
            )
            return result, query.explanation, figure
        if group is None or group not in frame:
            raise ValueError("Bu soru için geçerli bir grup sütunu bulunamadı.")
        if query.operation == "value_counts":
            result = frame[group].value_counts(dropna=False).head(query.limit).rename_axis(group).reset_index(name="Sayı")
            figure = px.pie(result, names=group, values="Sayı") if query.chart_type == "pie" else px.bar(result, x=group, y="Sayı")
            return result, query.explanation, figure
        if query.operation == "top_n":
            if target is None or not pd.api.types.is_numeric_dtype(frame[target]):
                raise ValueError("En yüksek değer sorgusu için sayısal hedef sütun gerekir.")
            result = frame[[group, target]].dropna().sort_values(target, ascending=False).head(query.limit)
            return result, query.explanation, px.bar(result, x=group, y=target, title=query.explanation)
        if target is None or not pd.api.types.is_numeric_dtype(frame[target]):
            raise ValueError("Gruplu hesaplama için sayısal hedef sütun gerekir.")
        result = (
            frame.groupby(group, dropna=False)[target].agg(query.agg_func)
            .sort_values(ascending=False).head(query.limit).rename("Değer").reset_index()
        )
        figure = px.line(result, x=group, y="Değer") if query.chart_type == "line" else px.bar(result, x=group, y="Değer")
        return result, query.explanation, figure
