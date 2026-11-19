"""Müşteri analizi: RFM segmentasyonu, kohort tutundurma ve tarihsel müşteri değeri (CLV).

Tanımlar (arayüzde ve raporda aynen yazılır):

- İşlem: müşteri, tarih ve tutarı dolu, tutarı sıfırdan büyük kayıt. Sipariş numarası sütunu verilirse
  aynı müşterinin aynı siparişteki satırları tek işlem sayılır (tutar toplanır); aynı numara farklı
  müşterilerde görünürse gelir karıştırılmaz ve durum not edilir. Tutarı boş/sıfır/negatif kayıtlar
  (iade, iptal ya da hatalı giriş olabilir) hesaba girmez ve sayısı not edilir.
- Analiz tarihi: verideki son işlem gününden bir gün sonrası. Yakınlık bu tarihe göre ölçülür.
- RFM: Yakınlık (son işlemden bu yana gün), Sıklık (işlem sayısı), Parasal (toplam tutar), seçilen
  pencere içindeki işlemlerle. Skorlar 1–5 arası yüzdelik dilimlerdir; eşit değerli müşteriler aynı
  skoru alır (bu yüzden çoğu müşterinin tek işlemi olan veride sıklık skorunda bazı değerler boş kalır).
  Segmentler yakınlık ve sıklık skorundan klasik RFM tablosuyla atanır.
- Kohort: müşterinin tüm verideki ilk işlem ayı. Tutundurma (k. ay) = kohorttaki müşterilerden
  ilk işleminden k ay sonra (takvim ayı) en az bir işlem yapanların oranı. Henüz gözlenemeyen hücreler
  boş bırakılır, 0 sayılmaz.
- Müşteri değeri (CLV, tarihsel): edinimden sonraki ilk N ayda müşteri başına ortalama gelir. Yalnızca
  N ayı tamamen gözlenmiş müşteriler kullanılır. Bu bir tahmin modeli değildir; gelir bazlıdır, kâr değildir.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from core.columns import _PERSON_NAME_TOKENS, is_identifier_name, is_text_series, name_tokens

_CUSTOMER_TOKENS = frozenset({"müşteri", "musteri", "customer", "client", "kullanıcı", "kullanici", "user",
                              "üye", "uye", "member", "cari"})
_ORDER_TOKENS = frozenset({"sipariş", "siparis", "order", "fatura", "invoice", "işlem", "islem", "fiş", "fis"})
_DATE_TOKENS = frozenset({"sipariş", "siparis", "order", "purchase", "satış", "satis", "işlem", "islem",
                          "fatura", "invoice", "alışveriş", "alisveris"})
_AMOUNT_PRIORITY = (
    frozenset({"tutar", "tutarı", "tutari", "amount", "revenue", "gelir", "ciro", "total", "toplam"}),
    frozenset({"payment", "ödeme", "odeme", "harcama", "spend", "sales", "satış", "satis"}),
    frozenset({"price", "fiyat", "fiyatı", "fiyati", "value", "değer"}),
)
WINDOW_OPTIONS = {"Tüm veri": None, "Son 12 ay": 12, "Son 6 ay": 6}
CLV_HORIZONS = (3, 6, 12)
MIN_GROUP_CUSTOMERS = 30

SEGMENT_RULES: list[tuple[str, str, str]] = [  # (R-F deseni, segment, önerilen aksiyon)
    (r"5[4-5]", "Şampiyonlar", "Ödüllendirin; yeni ürünlerde ilk erişim ve tavsiye programı."),
    (r"[3-4][4-5]", "Sadık müşteriler", "Sadakat programı ve çapraz satış önerileri."),
    (r"[4-5][2-3]", "Potansiyel sadıklar", "Üyelik/sadakat teklifiyle ikinci-üçüncü alışverişe yönlendirin."),
    (r"51", "Yeni müşteriler", "Karşılama akışı ve ilk 30 günde ikinci sipariş teşviki."),
    (r"41", "Umut vadeden", "Marka bilinirliği ve düşük eşikli ikinci sipariş kampanyası."),
    (r"33", "İlgi bekliyor", "Sınırlı süreli, kişiselleştirilmiş teklif."),
    (r"3[1-2]", "Uyumak üzere", "Geçmiş alışverişe dayalı öneriyle yeniden etkileşim."),
    (r"[1-2][3-4]", "Risk altında", "Kişisel geri kazanım iletişimi; neden ayrıldıklarını öğrenin."),
    (r"[1-2]5", "Kaybedilmemeli", "Öncelikli geri kazanım: yüksek değerli ama uzun süredir yok."),
    (r"[1-2][1-2]", "Kış uykusunda", "Düşük maliyetli hatırlatma; yanıt yoksa iletişimi azaltın."),
]
SEGMENT_ORDER = [name for _, name, _ in SEGMENT_RULES]
SEGMENT_ACTIONS = {name: action for _, name, action in SEGMENT_RULES}

DEFINITIONS = {
    "işlem": "Müşteri, tarih ve tutarı dolu, tutarı sıfırdan büyük kayıt; sipariş numarası verilirse müşteri ve "
             "sipariş başına tek işlem (numarası boş kayıt tek başına işlemdir). Boş/sıfır/negatif tutarlı kayıtlar "
             "hesaba girmez.",
    "rfm": "Yakınlık = analiz tarihine (son işlem günü + 1) kalan gün; Sıklık = işlem sayısı; Parasal = toplam tutar. "
           "Skorlar 1–5 yüzdelik dilim; eşit değerler aynı skoru alır. Segment, yakınlık ve sıklık skorlarından atanır.",
    "kohort": "Kohort = ilk işlem ayı (tüm veri). k. ay tutundurma = kohorttan ilk işleminden k takvim ayı sonra en az "
              "bir işlem yapanların oranı. Gözlenemeyen hücreler boştur, 0 sayılmaz.",
    "clv": "Tarihsel müşteri değeri = edinimden sonraki ilk N ayda müşteri başına ortalama gelir; yalnızca N ayı tamamen "
           "gözlenmiş müşteriler. Tahmin modeli değildir; gelir bazlıdır, kâr marjı içermez.",
}


class CustomerColumns(BaseModel):
    customer: str
    date: str
    amount: str
    order: str | None = None


class SegmentSummary(BaseModel):
    segment: str
    customers: int
    customer_share: float
    revenue_share: float
    recency_days: float
    frequency: float
    monetary: float
    action: str


class CLVRow(BaseModel):
    horizon_months: int
    customers: int
    revenue_per_customer: float | None
    orders_per_customer: float | None


class CustomerSummary(BaseModel):
    """Rapora giden özet: yalnızca toplu sayılar; müşteri kimliği ya da ham satır içermez."""

    columns: CustomerColumns
    window_label: str
    analysis_date: str
    window_start: str
    transactions: int
    customers: int
    repeat_customer_share: float
    excluded_rows: int
    segments: list[SegmentSummary]
    retention: dict[int, float] = Field(default_factory=dict)  # ay → kohort büyüklüğüyle ağırlıklı tutundurma
    clv: list[CLVRow] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    definitions: dict[str, str] = Field(default_factory=lambda: dict(DEFINITIONS))


class CustomerAnalysis(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    summary: CustomerSummary
    rfm_table: pd.DataFrame
    cohort_matrix: pd.DataFrame
    cohort_sizes: pd.Series
    clv_by_group: pd.DataFrame | None = None
    segment_figure: go.Figure
    cohort_figure: go.Figure
    clv_figure: go.Figure


def _has(tokens: list[str], vocabulary: frozenset) -> bool:
    return bool(set(tokens) & vocabulary)


def detect_customer_columns(frame: pd.DataFrame) -> CustomerColumns | None:
    """Müşteri, tarih ve tutar sütunlarını ad ve içerikten bulur; biri yoksa None (arayüz elle seçtirir)."""
    customers, orders = [], []
    for column in frame.columns:
        series, tokens = frame[column], name_tokens(column)
        if not is_text_series(series) or not is_identifier_name(column) or _has(tokens, _PERSON_NAME_TOKENS):
            continue
        present = series.dropna()
        if present.empty:
            continue
        repeat_ratio = len(present) / present.nunique()
        if _has(tokens, _CUSTOMER_TOKENS) and repeat_ratio >= 1.05 and present.nunique() >= 20:
            customers.append(str(column))
        elif _has(tokens, _ORDER_TOKENS):
            orders.append(str(column))
    dates = [str(column) for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])]
    dates.sort(key=lambda column: (not _has(name_tokens(column), _DATE_TOKENS), frame[column].isna().mean()))
    amounts: list[str] = []
    for vocabulary in _AMOUNT_PRIORITY:
        amounts += [
            str(column) for column in frame.columns
            if pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
            and _has(name_tokens(column), vocabulary) and str(column) not in amounts
        ]
    if not (customers and dates and amounts):
        return None
    return CustomerColumns(customer=customers[0], date=dates[0], amount=amounts[0], order=orders[0] if orders else None)


def transactions_from(frame: pd.DataFrame, columns: CustomerColumns) -> tuple[pd.DataFrame, int]:
    """(işlemler: müşteri, tarih, tutar; dışarıda kalan kayıt sayısı)."""
    for column in (columns.customer, columns.date, columns.amount, columns.order):
        if column is not None and column not in frame.columns:
            raise ValueError(f"“{column}” sütunu veride yok.")
    if not pd.api.types.is_datetime64_any_dtype(frame[columns.date]):
        raise ValueError(f"“{columns.date}” tarih sütunu değil.")
    if not pd.api.types.is_numeric_dtype(frame[columns.amount]):
        raise ValueError(f"“{columns.amount}” sayısal değil.")
    selected = [columns.customer, columns.date, columns.amount] + ([columns.order] if columns.order else [])
    data = frame[selected].copy()
    valid = data[columns.customer].notna() & data[columns.date].notna() & (data[columns.amount] > 0)
    excluded = int((~valid).sum())
    data = data[valid].rename(columns={columns.customer: "customer", columns.date: "date", columns.amount: "amount"})
    if columns.order:
        # Sipariş, müşteri ile birlikte gruplanır: aynı numara iki müşteride görünürse (hatalı veri) gelir ilk
        # müşteriye yazılmaz, her müşterinin kendi payı ayrı işlem olur. Numarası boş kayıt tek başına işlemdir.
        has_order = data[columns.order].notna()
        merged = data[has_order].groupby(["customer", columns.order], sort=False).agg(
            date=("date", "min"), amount=("amount", "sum"),
        ).reset_index()
        data = pd.concat([merged, data[~has_order]], ignore_index=True)[["customer", "date", "amount"]]
        data = data.sort_values("date", kind="stable")
    data["customer"] = data["customer"].astype(str)
    if data.empty:
        raise ValueError("Müşteri, tarih ve pozitif tutarı birlikte dolu kayıt yok.")
    return data.reset_index(drop=True), excluded


def _score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """1–5 yüzdelik skoru; eşit değerler aynı skoru alır (en düşük sıra yöntemi)."""
    ranked = (values if higher_is_better else -values).rank(method="min")
    return (1 + np.floor(5 * (ranked - 1) / len(values))).clip(1, 5).astype(int)


def _segment(code: str) -> str:
    return next(name for pattern, name, _ in SEGMENT_RULES if re.fullmatch(pattern, code))


def rfm_table(transactions: pd.DataFrame, window_months: int | None = None) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """(müşteri başına RFM tablosu, analiz tarihi, pencere başlangıcı)."""
    analysis_date = transactions["date"].max().normalize() + pd.Timedelta(days=1)
    start = transactions["date"].min().normalize() if window_months is None \
        else analysis_date - pd.DateOffset(months=window_months)
    window = transactions[transactions["date"] >= start]
    table = window.groupby("customer").agg(
        son_işlem=("date", "max"), sıklık=("date", "size"), parasal=("amount", "sum"),
    )
    table.insert(1, "yakınlık_gün", (analysis_date - table["son_işlem"]).dt.days)
    table["R"] = _score(table["yakınlık_gün"], higher_is_better=False)
    table["F"] = _score(table["sıklık"])
    table["M"] = _score(table["parasal"])
    table["RFM"] = table["R"].astype(str) + table["F"].astype(str) + table["M"].astype(str)
    table["segment"] = [_segment(f"{r}{f}") for r, f in zip(table["R"], table["F"])]
    return table.reset_index().rename(columns={"customer": "müşteri"}), analysis_date, start


def segment_summaries(table: pd.DataFrame) -> list[SegmentSummary]:
    total_customers, total_revenue = len(table), float(table["parasal"].sum())
    grouped = table.groupby("segment").agg(
        customers=("müşteri", "size"), revenue=("parasal", "sum"), recency=("yakınlık_gün", "mean"),
        frequency=("sıklık", "mean"), monetary=("parasal", "mean"),
    )
    return [
        SegmentSummary(
            segment=name, customers=int(row.customers), customer_share=row.customers / total_customers,
            revenue_share=row.revenue / total_revenue if total_revenue else 0.0, recency_days=float(row.recency),
            frequency=float(row.frequency), monetary=float(row.monetary), action=SEGMENT_ACTIONS[name],
        )
        for name in SEGMENT_ORDER if name in grouped.index
        for row in [grouped.loc[name]]
    ]


def _month_index(dates: pd.Series) -> pd.Series:
    return dates.dt.year * 12 + dates.dt.month - 1


def cohort_retention(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[int, float]]:
    """(kohort × ay tutundurma matrisi, kohort büyüklükleri, ağırlıklı ortalama eğri)."""
    month = _month_index(transactions["date"])
    first = month.groupby(transactions["customer"]).transform("min")
    offsets = pd.DataFrame({"customer": transactions["customer"], "cohort": first, "offset": month - first})
    sizes = offsets.groupby("cohort")["customer"].nunique()
    active = offsets.drop_duplicates(["customer", "offset"]).groupby(["cohort", "offset"]).size().unstack(fill_value=0)
    last_month = int(month.max())
    matrix = active.reindex(columns=range(0, last_month - int(sizes.index.min()) + 1), fill_value=0)
    matrix = matrix.div(sizes, axis=0).astype(float)
    for cohort in matrix.index:  # henüz gözlenemeyen aylar boş kalır
        matrix.loc[cohort, matrix.columns > last_month - cohort] = np.nan
    curve: dict[int, float] = {}
    for offset in matrix.columns[1:]:
        observed = matrix[offset].notna()
        if observed.any():
            curve[int(offset)] = float((matrix.loc[observed, offset] * sizes[observed]).sum() / sizes[observed].sum())
    labels = [f"{cohort // 12}-{cohort % 12 + 1:02d}" for cohort in matrix.index]
    matrix.index, sizes.index = labels, labels
    return matrix, sizes, curve


def _months_after(dates: pd.Series, months: int) -> pd.Series:
    return dates + pd.DateOffset(months=months)


def historical_value(
    transactions: pd.DataFrame, horizons: tuple[int, ...] = CLV_HORIZONS, groups: pd.Series | None = None,
) -> tuple[list[CLVRow], pd.DataFrame | None, list[float]]:
    """(ufuk başına tarihsel değer, isteğe bağlı kırılım tablosu, aylık kümülatif gelir eğrisi)."""
    first = transactions.groupby("customer")["date"].min()
    last_date = transactions["date"].max()
    acquisition = transactions["customer"].map(first)
    rows: list[CLVRow] = []
    group_rows: list[dict] = []
    for horizon in horizons:
        eligible = first[_months_after(first, horizon) <= last_date].index
        inside = transactions["customer"].isin(eligible) & (transactions["date"] < _months_after(acquisition, horizon))
        revenue = transactions[inside].groupby("customer")["amount"].sum().reindex(eligible, fill_value=0)
        orders = transactions[inside].groupby("customer").size().reindex(eligible, fill_value=0)
        rows.append(CLVRow(
            horizon_months=horizon, customers=len(eligible),
            revenue_per_customer=float(revenue.mean()) if len(eligible) else None,
            orders_per_customer=float(orders.mean()) if len(eligible) else None,
        ))
        if groups is not None and len(eligible):
            labels = groups.reindex(eligible)
            for label, values in revenue.groupby(labels):
                group_rows.append({"Grup": label, "Ufuk (ay)": horizon, "Müşteri": len(values),
                                   "Müşteri başına gelir": float(values.mean())})
    by_group = None
    if group_rows:
        by_group = pd.DataFrame(group_rows)
        by_group = by_group[by_group["Müşteri"] >= MIN_GROUP_CUSTOMERS]
    curve: list[float] = []
    offset_months = ((transactions["date"].dt.year - acquisition.dt.year) * 12
                     + transactions["date"].dt.month - acquisition.dt.month)
    observed_months = (last_date.year - first.dt.year) * 12 + last_date.month - first.dt.month
    for month in range(0, 25):
        eligible = observed_months[observed_months >= month].index
        if len(eligible) < MIN_GROUP_CUSTOMERS:
            break
        inside = transactions["customer"].isin(eligible) & (offset_months <= month)
        curve.append(float(transactions.loc[inside, "amount"].sum() / len(eligible)))
    return rows, by_group, curve


def _segment_figure(summaries: list[SegmentSummary]) -> go.Figure:
    names = [item.segment for item in summaries]
    figure = go.Figure([
        go.Bar(name="Müşteri payı", x=names, y=[item.customer_share * 100 for item in summaries],
               marker_color="#1d5fd1"),
        go.Bar(name="Gelir payı", x=names, y=[item.revenue_share * 100 for item in summaries], marker_color="#d97706"),
    ])
    figure.update_layout(barmode="group", title="RFM segmentleri: müşteri ve gelir payı (%)", yaxis_title="%",
                         separators=",.", legend={"orientation": "h", "y": -0.25})
    return figure


def _cohort_figure(matrix: pd.DataFrame, sizes: pd.Series) -> go.Figure:
    shown = matrix.iloc[:, : min(matrix.shape[1], 13)]
    text = [["" if pd.isna(value) else f"%{value * 100:.0f}" for value in row] for row in shown.to_numpy()]
    figure = go.Figure(go.Heatmap(
        z=shown.to_numpy() * 100, x=[str(column) for column in shown.columns],
        y=[f"{label} ({sizes[label]})" for label in shown.index], text=text, texttemplate="%{text}",
        colorscale="Blues", zmin=0, zmax=max(10.0, float(np.nanmax(shown.iloc[:, 1:].to_numpy() * 100))
                                             if shown.shape[1] > 1 and np.isfinite(shown.iloc[:, 1:].to_numpy()).any() else 100),
        colorbar={"title": "%"}, hovertemplate="Kohort %{y}<br>%{x}. ay: %{text}<extra></extra>",
    ))
    figure.update_layout(title="Kohort tutundurma (ilk işlemden sonraki ay; parantezde kohort büyüklüğü)",
                         xaxis_title="İlk işlemden sonraki ay", yaxis={"autorange": "reversed"},
                         height=max(380, 22 * len(shown) + 120), separators=",.")
    return figure


def _clv_figure(curve: list[float]) -> go.Figure:
    figure = go.Figure(go.Scatter(x=list(range(len(curve))), y=curve, mode="lines+markers", line_color="#1d5fd1",
                                  name="Müşteri başına kümülatif gelir"))
    figure.update_layout(title="Edinimden sonra müşteri başına kümülatif gelir (tarihsel)",
                         xaxis_title="İlk işlemden sonraki ay", yaxis_title="Gelir", separators=",.")
    return figure


class CustomerAnalyzer:
    def analyze(
        self, frame: pd.DataFrame, columns: CustomerColumns, window_label: str = "Tüm veri",
        group_column: str | None = None,
    ) -> CustomerAnalysis:
        if window_label not in WINDOW_OPTIONS:
            raise ValueError(f"Bilinmeyen pencere: {window_label}")
        transactions, excluded = transactions_from(frame, columns)
        table, analysis_date, start = rfm_table(transactions, WINDOW_OPTIONS[window_label])
        summaries = segment_summaries(table)
        matrix, sizes, curve = cohort_retention(transactions)
        groups = None
        if group_column is not None:
            if group_column not in frame.columns:
                raise ValueError(f"“{group_column}” sütunu veride yok.")
            valid = frame[columns.customer].notna() & frame[columns.date].notna()
            ordered = frame.loc[valid].sort_values(columns.date, kind="stable")
            groups = ordered.groupby(ordered[columns.customer].astype(str))[group_column].first().astype(str)
        clv_rows, clv_by_group, value_curve = historical_value(transactions, groups=groups)

        notes: list[str] = []
        if excluded:
            notes.append(f"{excluded} kayıt müşteri/tarih eksik ya da tutarı sıfır veya negatif olduğu için hesaba girmedi.")
        if columns.order:
            orders = frame.loc[frame[columns.order].notna() & frame[columns.customer].notna()]
            shared = int((orders.groupby(columns.order)[columns.customer].nunique() > 1).sum())
            if shared:
                notes.append(f"{shared} sipariş numarası birden fazla müşteride görünüyor; her müşterinin payı ayrı "
                             "işlem sayıldı.")
        last = transactions["date"].max()
        if last.normalize() < (last + pd.offsets.MonthEnd(0)).normalize():
            notes.append(f"Son ay ({last:%m.%Y}) {last.day}. güne kadar gözlendi; son sütunlardaki tutundurma eksik ayı içerir.")
        for row in clv_rows:
            if row.customers == 0:
                notes.append(f"{row.horizon_months} aylık değer hesaplanamadı: o kadar süre gözlenmiş müşteri yok.")
        per_customer = transactions.groupby("customer").size()
        summary = CustomerSummary(
            columns=columns, window_label=window_label, analysis_date=f"{analysis_date:%d.%m.%Y}",
            window_start=f"{start:%d.%m.%Y}", transactions=len(transactions), customers=int(per_customer.size),
            repeat_customer_share=float((per_customer > 1).mean()), excluded_rows=excluded, segments=summaries,
            retention={month: value for month, value in curve.items() if month in (1, 3, 6, 12)},
            clv=clv_rows, notes=notes,
        )
        return CustomerAnalysis(
            summary=summary, rfm_table=table, cohort_matrix=matrix, cohort_sizes=sizes, clv_by_group=clv_by_group,
            segment_figure=_segment_figure(summaries), cohort_figure=_cohort_figure(matrix, sizes),
            clv_figure=_clv_figure(value_curve),
        )
