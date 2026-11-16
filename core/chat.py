"""Güvenli doğal dil veri sorgulama ve Plotly görselleştirme motoru.

LLM (ya da anahtar yoksa kurallı ayrıştırıcı) yalnızca `StructuredDataQuery` planı üretir. Plan,
sütun ve tip doğrulamasından geçtikten sonra önceden yazılmış pandas işlemleriyle çalıştırılır;
`eval`/`DataFrame.query` gibi metin yorumlayan hiçbir yol yoktur. Her cevap için eşdeğer pandas ve
SQL kodu yalnızca gösterim amacıyla üretilir, çalıştırılmaz.
"""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field

from core.columns import is_identifier_column, is_text_series, parse_numeric_text
from core.formatting import decimal_tr, integer_tr, localize_figure
from core.metadata import DatasetMetadataCard, MetadataGenerator
from core.planner import PlanOrigin, failure_reason, request_structured

Operator = Literal["==", "!=", ">", ">=", "<", "<=", "in"]
TimeGrain = Literal["day", "week", "month", "year"]

MONTHS_TR = ("ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
             "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık")
_MONTH_LABELS = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
                 "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
_GRAIN_WORDS = {"günlük": "day", "haftalık": "week", "aylık": "month", "yıllık": "year"}
_GRAIN_LABELS = {"day": "gün", "week": "hafta", "month": "ay", "year": "yıl"}
_SQL_AGG = {"mean": "AVG", "sum": "SUM", "count": "COUNT", "min": "MIN", "max": "MAX"}
_AGG_TR = {"mean": "ortalama", "sum": "toplam", "count": "sayı", "min": "en düşük", "max": "en yüksek",
           "median": "medyan"}
_MAX_FILTER_CATEGORIES = 50


class QueryFilter(BaseModel):
    """Tek bir filtre koşulu. Değerler metin olarak taşınır, sütun tipine göre çevrilir."""

    column: str
    operator: Operator = "=="
    value: str = ""
    values: list[str] = Field(default_factory=list)


class StructuredDataQuery(BaseModel):
    operation: Literal[
        "groupby_agg", "value_counts", "top_n", "describe",
        "correlation", "histogram", "time_trend",
    ]
    group_col: str | None = None
    target_col: str | None = None
    agg_func: Literal["mean", "sum", "count", "min", "max", "median"] = "mean"
    chart_type: Literal[
        "bar", "pie", "line", "histogram", "box", "scatter", "heatmap",
    ] = "bar"
    limit: int = Field(default=10, ge=1, le=100)
    filters: list[QueryFilter] = Field(default_factory=list)
    time_col: str | None = None
    time_grain: TimeGrain | None = None
    explanation: str


class QueryCode(BaseModel):
    """Cevabın nasıl hesaplandığı: eşdeğer pandas ve SQL (yalnızca gösterim, çalıştırılmaz)."""

    filters: list[str]
    rows_used: int
    pandas: str
    sql: str


class DataChatEngine:
    """LLM'i yalnızca doğrulanmış sorgu planı üretmek için kullanan motor."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key.strip() if api_key else ""
        self.model = model
        self.origin = PlanOrigin(source="yerel kurallar", reason="Henüz soru sorulmadı.")

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
        filtered, descriptions = self.apply_filters(frame, query)
        result, summary, figure = self._execute(filtered, query)
        if descriptions:
            summary = f"{summary} Filtre: {'; '.join(descriptions)} ({integer_tr(len(filtered))} kayıt)."
        return result, summary, localize_figure(figure), query

    # --- Planlama ----------------------------------------------------------------------------

    def _plan(
        self,
        question: str,
        card: DatasetMetadataCard,
        frame: pd.DataFrame,
    ) -> StructuredDataQuery:
        if not self.api_key:
            self.origin = PlanOrigin(source="yerel kurallar", reason="Gemini API anahtarı girilmedi.")
            return self._fallback(question, frame)
        try:
            query = self._validate_columns(
                request_structured(self.api_key, self.model, self._build_prompt(question, card), StructuredDataQuery),
                frame,
            )
        except Exception as exc:  # SDK hataları ve geçersiz sütun seçen planlar
            self.origin = PlanOrigin(
                source="yerel kurallar", error=True,
                reason=f"Gemini sorgu planı kullanılamadı ({failure_reason(exc)}); yerel ayrıştırıcı kullanıldı.",
            )
            return self._fallback(question, frame)
        self.origin = PlanOrigin(source="Gemini", reason=f"{self.model} sorgu planı doğrulandı.")
        return query

    @staticmethod
    def _build_prompt(question: str, card: DatasetMetadataCard) -> str:
        """LLM'e giden metin: soru da metadata gibi PII maskesinden geçer."""
        return (
            "Türkçe kullanıcı sorusunu yalnızca aşağıdaki KVKK/PII maskeli "
            "metadata ile analiz et. Ham veri isteme veya üretme. Sadece "
            "doğrulanmış StructuredDataQuery JSON'u döndür. Koşullar `filters` içinde "
            "(tarihler YYYY-MM-DD), zaman kırılımı `time_trend` + `time_col` + `time_grain` ile verilir.\n"
            f"Kullanıcı sorusu: {MetadataGenerator.mask_pii(question)}\n"
            f"Metadata: {card.model_dump_json()}"
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9çğıöşü]", "", value.replace("İ", "i").lower())

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
        columns = [str(c) for c in frame.columns]
        dates = [c for c in columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
        numeric = [c for c in columns if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [c for c in columns if c not in numeric and c not in dates]
        lowered = question.replace("İ", "i").lower()

        filters, consumed = self._parse_filters(question, lowered, frame, numeric, categorical, dates)
        remaining = lowered
        for start, end in sorted(consumed, reverse=True):
            remaining = remaining[:start] + " " + remaining[end:]
        filter_columns = {item.column for item in filters}
        single_value_columns = {item.column for item in filters if item.operator == "=="}

        # Filtrenin kullandığı sütun adı `remaining` metninden çıkarıldı; ad yeniden geçiyorsa ("yaşı 40'tan büyük
        # müşterilerin ortalama yaşı") aynı sütun hedef olabilir.
        target = self._find_column(remaining, numeric, True, frame)
        group = self._find_column(remaining, categorical) or None
        limit_match = re.search(r"\b(\d{1,3})\b", remaining)
        limit = max(1, min(int(limit_match.group(1)), 100)) if limit_match else 10
        grain = next((value for word, value in _GRAIN_WORDS.items() if word in lowered), None)
        if grain is None and dates and any(word in lowered for word in ("zamanla", "trend", "aylara göre", "ay ay")):
            grain = "month"

        if grain and dates:
            operation, chart = "time_trend", "line"
            explanation = f"{_GRAIN_LABELS[grain].capitalize()} bazında değişim gösteriliyor."
        elif "korelasyon" in lowered:
            operation, chart, explanation = "correlation", "heatmap", "Sayısal sütunlar arasındaki korelasyon gösteriliyor."
        elif "histogram" in lowered:
            operation, chart, explanation = "histogram", "histogram", "Sayısal değerlerin histogram dağılımı gösteriliyor."
        elif "dağılım" in lowered:
            operation, chart, explanation = "value_counts", "pie", "Değerlerin dağılımı gösteriliyor."
        elif "en yüksek" in lowered or "en çok" in lowered or re.search(r"\btop\b", lowered):
            # "top" kelime sınırıyla aranır: "toplam" içinde geçtiği için yanlış işleme gidiyordu.
            operation, chart, explanation = "top_n", "bar", f"En yüksek {limit} değer gösteriliyor."
        elif any(word in remaining for word in ("sayı", "kaç", "frekans")) and not target:
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

        time_col = None
        if operation == "time_trend":
            time_col = self._find_column(remaining, dates) or dates[0]
            if not target and (any(word in lowered for word in ("sayı", "kaç", "adet")) or "toplam" not in lowered):
                agg = "count"
            explanation = (
                f"{_GRAIN_LABELS[grain].capitalize()} bazında "
                + ("kayıt sayısı" if agg == "count" else f"{_AGG_TR[agg]} {target or numeric[0]}")
                + " gösteriliyor."
            )
        if not target and operation not in ("time_trend", "correlation"):
            choices = [c for c in numeric if c not in filter_columns] or numeric
            target = choices[0] if choices else columns[0]
        wants_grouping = any(word in lowered for word in ("göre", "bazında", "başına", "kırılım", "her "))
        multi_value = [item.column for item in filters if item.operator == "in"]
        if not group and operation == "groupby_agg" and multi_value:
            group = multi_value[0]  # "Ankara ve İzmir için toplam harcama": değerler karşılaştırılır
        elif not group and operation == "groupby_agg" and not wants_grouping:
            group = None  # gruplama istenmediyse ("ortalama yaşı nedir?") genel değer hesaplanır
        elif not group and operation in ("groupby_agg", "value_counts", "top_n"):
            # Grup belirtilmemişse: kimlik olmayan, tek değere filtrelenmemiş ilk kategorik sütun.
            group = next(
                (c for c in categorical
                 if c not in single_value_columns and not is_identifier_column(frame[c], c)),
                None,
            )
            if group is None and operation in ("value_counts", "top_n"):
                group = next((c for c in categorical if c not in single_value_columns), columns[0])
        if operation == "groupby_agg" and group is None:
            explanation = f"Seçilen kayıtlar için {_AGG_TR[agg]} {target} hesaplanıyor."
        if operation == "correlation":
            target = None
        return StructuredDataQuery(
            operation=operation, group_col=group, target_col=target, agg_func=agg, chart_type=chart,
            limit=limit, filters=filters, time_col=time_col, time_grain=grain if time_col else None,
            explanation=explanation,
        )

    def _parse_filters(
        self, question: str, lowered: str, frame: pd.DataFrame,
        numeric: list[str], categorical: list[str], dates: list[str],
    ) -> tuple[list[QueryFilter], list[tuple[int, int]]]:
        """Sorudaki koşulları çıkarır: ay/yıl, kategori değeri ve sayısal karşılaştırma."""
        filters: list[QueryFilter] = []
        consumed: list[tuple[int, int]] = []

        # 1. Tarih: "Kasım 2017'de", "2018'de", "2017 ile 2018 arasında"
        if dates:
            date_col = self._find_column(question, dates) or dates[0]
            month_pattern = re.compile(rf"\b({'|'.join(MONTHS_TR)})\s+((?:19|20)\d{{2}})\b")
            month_match = month_pattern.search(lowered)
            years = [(match.start(), match.end(), int(match.group(0)))
                     for match in re.finditer(r"\b(?:19|20)\d{2}\b", lowered)]
            if month_match:
                year, month = int(month_match.group(2)), MONTHS_TR.index(month_match.group(1)) + 1
                start = pd.Timestamp(year=year, month=month, day=1)
                filters += [QueryFilter(column=date_col, operator=">=", value=start.date().isoformat()),
                            QueryFilter(column=date_col, operator="<",
                                        value=(start + pd.offsets.MonthBegin(1)).date().isoformat())]
                consumed.append(month_match.span())
            elif years:
                first, last = min(year for *_, year in years), max(year for *_, year in years)
                filters += [QueryFilter(column=date_col, operator=">=", value=f"{first}-01-01"),
                            QueryFilter(column=date_col, operator="<", value=f"{last + 1}-01-01")]
                consumed += [(start, end) for start, end, _ in years]

        # 2. Sayısal karşılaştırma: "yaşı 40'tan büyük", "en az 100", "100 üzeri"
        number = r"(\d+(?:[.,]\d+)*)"
        patterns = (
            (re.compile(rf"{number}\s*['’]?\s*(?:den|dan|ten|tan)\s+(büyük|fazla|yüksek|küçük|az|düşük)"), None),
            (re.compile(rf"{number}\s+(üzeri|üstü|altı)"), None),
            (re.compile(rf"en\s+(az|fazla)\s+{number}"), "prefix"),
        )
        for pattern, kind in patterns:
            for match in pattern.finditer(lowered):
                word, raw = (match.group(1), match.group(2)) if kind == "prefix" else (match.group(2), match.group(1))
                operator = {
                    "büyük": ">", "fazla": ">", "yüksek": ">", "üzeri": ">", "üstü": ">",
                    "küçük": "<", "az": "<", "düşük": "<", "altı": "<",
                }[word]
                if kind == "prefix":
                    operator = ">=" if word == "az" else "<="
                prefix = self._normalize(lowered[:match.start()])
                mentioned = [(prefix.rfind(self._normalize(c)), c) for c in numeric if self._normalize(c) in prefix]
                if not mentioned:
                    continue
                column = max(mentioned)[1]
                value = parse_numeric_text(pd.Series([raw]))[0].iloc[0]
                if pd.isna(value):
                    continue
                filters.append(QueryFilter(column=column, operator=operator, value=str(float(value))))
                consumed.append(match.span())
                # Karşılaştırmanın kullandığı sütun adı da tüketilir; böylece hedef aranırken sayılmaz.
                mention = lowered.rfind(column.replace("İ", "i").lower(), 0, match.start())
                if mention >= 0:
                    consumed.append((mention, mention + len(column)))

        # 3. Kategori değeri: "İstanbul için", "delivered siparişler". Değer, kelime sınırlarıyla aranır
        # ("teslimat" içinde "teslim" geçmez); uzun eşleşme önce gelir ve çakışan kısa eşleşmeler atlanır
        # ("teslim edilmedi" sorusu "teslim" kayıtlarını da almaz).
        letter = "0-9a-zçğıöşüâîû"
        candidates: list[tuple[int, int, int, str, str]] = []
        for column in categorical:
            series = frame[column]
            if is_identifier_column(series, column) or series.nunique() > _MAX_FILTER_CATEGORIES:
                continue
            for value in series.dropna().unique():
                text = str(value)
                tokens = [token for token in re.split(r"[\s_\-]+", text.replace("İ", "i").lower()) if token]
                if not tokens or len("".join(tokens)) < 4:
                    continue
                pattern = rf"(?<![{letter}])" + r"[\s_\-]+".join(map(re.escape, tokens)) + rf"(?![{letter}])"
                for match in re.finditer(pattern, lowered):
                    candidates.append((match.end() - match.start(), match.start(), match.end(), column, text))
        taken = list(consumed)
        matched: dict[str, list[str]] = {}
        for _, start, end, column, text in sorted(candidates, key=lambda item: (-item[0], item[1])):
            if any(start < other_end and other_start < end for other_start, other_end in taken):
                continue
            taken.append((start, end))
            consumed.append((start, end))
            if text not in matched.setdefault(column, []):
                matched[column].append(text)
        for column in categorical:
            values = matched.get(column)
            if not values:
                continue
            if len(values) == 1:
                filters.append(QueryFilter(column=column, operator="==", value=values[0]))
            else:
                filters.append(QueryFilter(column=column, operator="in", values=sorted(values)))
        return filters, consumed

    @staticmethod
    def _validate_columns(query: StructuredDataQuery, frame: pd.DataFrame) -> StructuredDataQuery:
        columns = {str(column) for column in frame.columns}
        names = [query.group_col, query.target_col, query.time_col, *[item.column for item in query.filters]]
        for name in names:
            if name is not None and name not in columns:
                raise ValueError(f"Plan sütun içermiyor veya geçersiz sütun seçti: {name}")
        if query.operation == "time_trend" and (
            query.time_col is None or not pd.api.types.is_datetime64_any_dtype(frame[query.time_col])
        ):
            raise ValueError("Zaman kırılımı için geçerli bir tarih sütunu gerekir.")
        return query

    # --- Filtreler ---------------------------------------------------------------------------

    @staticmethod
    def _typed_value(series: pd.Series, raw: str, column: str):
        if pd.api.types.is_datetime64_any_dtype(series):
            try:
                return pd.Timestamp(raw)
            except ValueError as exc:
                raise ValueError(f"“{column}” için geçersiz tarih: {raw}") from exc
        if pd.api.types.is_numeric_dtype(series):
            value = parse_numeric_text(pd.Series([raw]))[0].iloc[0]
            if pd.isna(value):
                raise ValueError(f"“{column}” sayısal; “{raw}” sayı değil.")
            return float(value)
        return raw

    def apply_filters(self, frame: pd.DataFrame, query: StructuredDataQuery) -> tuple[pd.DataFrame, list[str]]:
        """Filtreleri tipleri doğrulanmış maskelerle uygular; girdiyi değiştirmez."""
        self._validate_columns(query, frame)
        mask = pd.Series(True, index=frame.index)
        descriptions: list[str] = []
        for item in query.filters:
            series = frame[item.column]
            if item.operator == "in":
                if not item.values:
                    raise ValueError(f"“{item.column}” için liste filtresi boş.")
                values = [self._typed_value(series, value, item.column) for value in item.values]
                mask &= series.isin(values)
                descriptions.append(f"{item.column} ∈ {{{', '.join(item.values)}}}")
                continue
            value = self._typed_value(series, item.value, item.column)
            if item.operator not in ("==", "!=") and is_text_series(series):
                raise ValueError(f"“{item.column}” metin sütunu; “{item.operator}” karşılaştırması yapılamaz.")
            comparisons = {
                "==": series == value, "!=": series != value, ">": series > value,
                ">=": series >= value, "<": series < value, "<=": series <= value,
            }
            mask &= comparisons[item.operator].fillna(False)
            shown = value.strftime("%d.%m.%Y") if isinstance(value, pd.Timestamp) else (
                decimal_tr(value, 2).rstrip("0").rstrip(",") if isinstance(value, float) else value
            )
            descriptions.append(f"{item.column} {item.operator} {shown}")
        filtered = frame[mask]
        if query.filters and filtered.empty:
            raise ValueError(f"Filtrelere uyan kayıt yok: {'; '.join(descriptions)}.")
        return filtered, descriptions

    # --- Çalıştırma --------------------------------------------------------------------------

    @staticmethod
    def _period_labels(periods: pd.Series, grain: str) -> pd.Series:
        if grain == "month":
            return periods.map(lambda stamp: f"{_MONTH_LABELS[stamp.month - 1]} {stamp.year}")
        if grain == "year":
            return periods.map(lambda stamp: str(stamp.year))
        if grain == "week":
            return periods.map(lambda stamp: f"{stamp:%d.%m.%Y} haftası")
        return periods.map(lambda stamp: f"{stamp:%d.%m.%Y}")

    def _execute(self, frame: pd.DataFrame, query: StructuredDataQuery) -> tuple[pd.DataFrame, str, object]:
        group = query.group_col
        target = query.target_col
        if query.operation == "time_trend":
            grain = query.time_grain or "month"
            stamps = frame[query.time_col].dropna()
            period = {"day": "D", "week": "W-SUN", "month": "M", "year": "Y"}[grain]
            keys = stamps.dt.to_period(period).dt.start_time
            if query.agg_func == "count" or target is None:
                values = keys.value_counts().sort_index()
                label = "Kayıt Sayısı"
            else:
                if not pd.api.types.is_numeric_dtype(frame[target]):
                    raise ValueError("Zaman kırılımında toplanacak sütun sayısal olmalı.")
                values = frame.loc[stamps.index, target].groupby(keys).agg(query.agg_func).sort_index()
                label = "Değer"
            result = pd.DataFrame({"Dönem": self._period_labels(pd.Series(values.index), grain).to_numpy(),
                                   label: values.to_numpy()})
            figure = px.line(x=values.index, y=values.to_numpy(), markers=True, title=query.explanation,
                             labels={"x": "Dönem", "y": label})
            return result, query.explanation, figure
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
        if query.operation == "groupby_agg" and group is None:
            if target is None or not pd.api.types.is_numeric_dtype(frame[target]):
                raise ValueError("Hesaplama için sayısal hedef sütun gerekir.")
            value = frame[target].agg(query.agg_func)
            result = pd.DataFrame({"Ölçü": [f"{_AGG_TR[query.agg_func]} {target}"], "Değer": [value]})
            return result, query.explanation, px.bar(result, x="Ölçü", y="Değer")
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

    # --- Şeffaflık ---------------------------------------------------------------------------

    def explain(self, frame: pd.DataFrame, query: StructuredDataQuery) -> QueryCode:
        """Cevabın eşdeğer pandas ve SQL kodu. Yalnızca gösterilir; hiçbir zaman çalıştırılmaz."""
        filtered, descriptions = self.apply_filters(frame, query)

        def sql_name(name: str) -> str:
            return '"' + name.replace('"', '""') + '"'

        def sql_value(column: str, raw: str) -> str:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                return repr(self._typed_value(series, raw, column))
            if pd.api.types.is_datetime64_any_dtype(series):
                return f"TIMESTAMP '{pd.Timestamp(raw):%Y-%m-%d %H:%M:%S}'"
            return "'" + raw.replace("'", "''") + "'"

        def py_value(column: str, raw: str) -> str:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                return repr(self._typed_value(series, raw, column))
            if pd.api.types.is_datetime64_any_dtype(series):
                return f"pd.Timestamp({pd.Timestamp(raw).date().isoformat()!r})"
            return repr(raw)

        py_conditions, sql_conditions = [], []
        for item in query.filters:
            if item.operator == "in":
                py_conditions.append(f"df[{item.column!r}].isin([{', '.join(py_value(item.column, v) for v in item.values)}])")
                sql_conditions.append(f"{sql_name(item.column)} IN ({', '.join(sql_value(item.column, v) for v in item.values)})")
            else:
                py_conditions.append(f"(df[{item.column!r}] {item.operator} {py_value(item.column, item.value)})")
                sql_operator = "=" if item.operator == "==" else ("<>" if item.operator == "!=" else item.operator)
                sql_conditions.append(f"{sql_name(item.column)} {sql_operator} {sql_value(item.column, item.value)}")
        lines = ["df = veri"]
        if py_conditions:
            lines.append(f"df = df[{' & '.join(py_conditions)}]")
        where = f"\nWHERE {' AND '.join(sql_conditions)}" if sql_conditions else ""
        group, target, agg, limit = query.group_col, query.target_col, query.agg_func, query.limit

        def sql_agg(column: str) -> str:
            if agg == "median":
                return f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {sql_name(column)})"
            return f"{_SQL_AGG[agg]}({sql_name(column)})"

        operation = query.operation
        if operation == "time_trend":
            grain = query.time_grain or "month"
            period = {"day": "D", "week": "W-SUN", "month": "M", "year": "Y"}[grain]
            key = f"df[{query.time_col!r}].dt.to_period({period!r}).dt.start_time"
            if agg == "count" or target is None:
                lines.append(f"sonuc = {key}.value_counts().sort_index()")
                measure = "COUNT(*)"
            else:
                lines.append(f"sonuc = df.groupby({key})[{target!r}].agg({agg!r}).sort_index()")
                measure = sql_agg(target)
            truncated = f"DATE_TRUNC('{grain}', {sql_name(query.time_col)})"
            sql = (f"SELECT {truncated} AS donem, {measure} AS deger\nFROM veri{where}"
                   f"{' AND ' if where else chr(10) + 'WHERE '}{sql_name(query.time_col)} IS NOT NULL"
                   f"\nGROUP BY donem\nORDER BY donem;")
        elif operation == "groupby_agg" and group is None:
            lines.append(f"sonuc = df[{target!r}].agg({agg!r})")
            sql = f"SELECT {sql_agg(target)} AS deger\nFROM veri{where};"
        elif operation == "groupby_agg":
            lines.append(
                f"sonuc = df.groupby({group!r}, dropna=False)[{target!r}].agg({agg!r})"
                f".sort_values(ascending=False).head({limit})"
            )
            sql = (f"SELECT {sql_name(group)}, {sql_agg(target)} AS deger\nFROM veri{where}"
                   f"\nGROUP BY {sql_name(group)}\nORDER BY deger DESC\nLIMIT {limit};")
        elif operation == "value_counts":
            lines.append(f"sonuc = df[{group!r}].value_counts(dropna=False).head({limit})")
            sql = (f"SELECT {sql_name(group)}, COUNT(*) AS sayi\nFROM veri{where}"
                   f"\nGROUP BY {sql_name(group)}\nORDER BY sayi DESC\nLIMIT {limit};")
        elif operation == "top_n":
            lines.append(f"sonuc = df[[{group!r}, {target!r}]].dropna().sort_values({target!r}, ascending=False).head({limit})")
            not_null = f"{sql_name(target)} IS NOT NULL AND {sql_name(group)} IS NOT NULL"
            sql = (f"SELECT {sql_name(group)}, {sql_name(target)}\nFROM veri"
                   f"{where + ' AND ' if where else chr(10) + 'WHERE '}{not_null}"
                   f"\nORDER BY {sql_name(target)} DESC\nLIMIT {limit};")
        elif operation == "histogram":
            column = target or group
            lines.append(f"sonuc = pd.cut(df[{column!r}].dropna(), bins=10, include_lowest=True).value_counts(sort=False)")
            sql = (f"-- Yaklaşık karşılık: eşit genişlikte 10 aralık\n"
                   f"SELECT WIDTH_BUCKET({sql_name(column)}, (SELECT MIN({sql_name(column)}) FROM veri), "
                   f"(SELECT MAX({sql_name(column)}) FROM veri), 10) AS aralik, COUNT(*) AS sayi\n"
                   f"FROM veri{where}\nGROUP BY aralik\nORDER BY aralik;")
        elif operation == "correlation":
            lines.append("sonuc = df.select_dtypes(include='number').corr()")
            numeric = [str(c) for c in frame.select_dtypes(include="number").columns][:3]
            pairs = [f"CORR({sql_name(a)}, {sql_name(b)})" for i, a in enumerate(numeric) for b in numeric[i + 1:]]
            sql = ("-- Her sayısal sütun çifti için CORR(a, b)\n"
                   f"SELECT {', '.join(pairs) or 'NULL'}\nFROM veri{where};")
        else:
            lines.append("sonuc = df.describe(include='all')")
            sql = f"-- Sütun başına COUNT, AVG, MIN, MAX\nSELECT COUNT(*) AS kayit_sayisi\nFROM veri{where};"
        return QueryCode(filters=descriptions, rows_used=len(filtered), pandas="\n".join(lines), sql=sql)
