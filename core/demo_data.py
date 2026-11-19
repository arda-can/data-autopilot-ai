"""Sentetik Türkçe e-ticaret demo veri seti.

Gerçek kişi ya da şirket verisi değildir; isimler, e-postalar ve telefonlar rastgele üretilir. Veri,
uygulamanın her bölümünü dürüstçe gösterebilmek için bilinen örüntülerle üretilir ve bu örüntüler
`DEMO_STORY` içinde açıkça yazılıdır:

- Tekrar eden müşteriler (RFM, kohort, CLV): her müşterinin ilk siparişinden sonra tekrar olasılığı farklıdır.
- Öğrenilebilir sinyal: iade olasılığı gecikme, kategori, yüksek indirim ve kanala bağlıdır;
  değerlendirme puanı gecikmeyle düşer.
- Kayma: 2025'in ikinci yarısında mobil kanal payı artar, kapıda ödeme azalır; Ekim 2025'ten itibaren
  Ankara'da teslimat gecikir. Fiyatlar ay başına %2 artar (enflasyon).
- A/B deneyi: 1 Temmuz 2025'ten itibaren web ve mobil müşteriler müşteri bazında A/B gruplarına
  ayrılır; B grubunda (yeni öneri kutusu) ek ürün ekleme oranı daha yüksektir.
- Bilinçli kural ihlalleri ve hijyen sorunları: `DemoDataset.injected` içinde sayılarıyla kayıtlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_FILENAME = "demo_eticaret_siparisleri.csv"
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
EXPERIMENT_START = pd.Timestamp("2025-07-01")
ANKARA_DELAY_START = pd.Timestamp("2025-10-01")
SHIFT_START = pd.Timestamp("2025-07-01")

DEMO_STORY = (
    "Sentetik veri: 2024–2025 arası Türk e-ticaret siparişleri. Tekrar eden müşteriler, gecikmeye bağlı "
    "iade ve puan, 2025'in ikinci yarısında mobil kanala kayma, Ekim 2025'ten itibaren Ankara'da "
    "teslimat gecikmesi ve Temmuz 2025'te başlayan bir A/B deneyi (öneri kutusu) içerir. Bazı kayıtlar "
    "kural ihlali ve hijyen sorunu göstermek için bilerek bozulmuştur."
)

_FIRST_NAMES = np.array([
    "Ayşe", "Mehmet", "Fatma", "Ali", "Zeynep", "Mustafa", "Elif", "Ahmet", "Emine", "Hüseyin",
    "Merve", "Can", "Selin", "Burak", "Deniz", "Ece", "Emre", "Gizem", "Kerem", "Sena",
])
_LAST_NAMES = np.array([
    "Yılmaz", "Kaya", "Demir", "Çelik", "Şahin", "Yıldız", "Aydın", "Öztürk", "Arslan", "Doğan",
    "Kılıç", "Aslan", "Koç", "Kurt", "Özdemir",
])
_ASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
_CITIES = {  # şehir: (müşteri payı, olağan teslim günü)
    "İstanbul": (0.35, 2), "Ankara": (0.15, 3), "İzmir": (0.12, 3), "Bursa": (0.08, 3),
    "Antalya": (0.08, 4), "Adana": (0.07, 4), "Konya": (0.07, 4), "Trabzon": (0.08, 5),
}
_CATEGORIES = {  # kategori: (pay, ortalama birim fiyat TL, Ocak 2024)
    "Elektronik": (0.18, 4500.0), "Giyim": (0.30, 650.0), "Ev & Yaşam": (0.20, 900.0),
    "Kozmetik": (0.22, 350.0), "Kitap": (0.10, 180.0),
}
_MONTH_FACTOR = {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.0, 6: 0.95, 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.05,
                 11: 1.6, 12: 1.3}
_WEEKDAY_FACTOR = np.array([0.85, 0.85, 0.9, 0.9, 1.0, 1.2, 1.3])  # Pazartesi … Pazar


@dataclass
class DemoDataset:
    frame: pd.DataFrame
    injected: dict[str, int] = field(default_factory=dict)
    story: str = DEMO_STORY


def _seasonal_weight(days: pd.DatetimeIndex) -> np.ndarray:
    months = np.array([_MONTH_FACTOR[month] for month in days.month])
    return months * _WEEKDAY_FACTOR[days.dayofweek]


def _order_dates(rng: np.random.Generator, customers: int) -> tuple[np.ndarray, np.ndarray]:
    """(müşteri indeksi, sipariş günü). İlk siparişler büyüyen ve mevsimsel bir talepten çekilir."""
    days = pd.date_range(START, END, freq="D")
    growth = 1 + 1.2 * np.arange(len(days)) / len(days)
    weights = growth * _seasonal_weight(days)
    first = rng.choice(len(days), size=customers, p=weights / weights.sum())
    loyalty = rng.beta(2.2, 2.5, customers)  # sipariş sonrası tekrar olasılığı
    seasonal = _seasonal_weight(days) / _seasonal_weight(days).max()
    owner, day_index = [], []
    for customer in range(customers):
        current = int(first[customer])
        repeat = loyalty[customer]
        while current < len(days):
            owner.append(customer)
            day_index.append(current)
            if rng.random() > repeat:
                break
            repeat *= 0.93  # her siparişten sonra tekrar olasılığı biraz azalır
            current += 3 + int(rng.exponential(50))
            while current < len(days) and rng.random() > seasonal[current]:
                current += 1
    return np.array(owner), days[np.array(day_index)].to_numpy()


def _choice(rng: np.random.Generator, options: list[str], probabilities: np.ndarray) -> np.ndarray:
    """Satır başına farklı olasılıklarla seçim (olasılık matrisi: satır × seçenek)."""
    cumulative = probabilities.cumsum(axis=1)
    draws = rng.random(len(probabilities))[:, None]
    return np.array(options)[(draws > cumulative).sum(axis=1).clip(max=len(options) - 1)]


def generate_demo_data(seed: int = 7, customers: int = 4000) -> DemoDataset:
    rng = np.random.default_rng(seed)
    owner, order_day = _order_dates(rng, customers)
    rows = len(owner)
    order_time = pd.to_datetime(order_day) + pd.to_timedelta(rng.integers(8 * 60, 24 * 60, rows), unit="min")
    after_shift = order_time >= SHIFT_START
    months_since_start = (order_time.year - START.year) * 12 + order_time.month - 1

    first_names = _FIRST_NAMES[rng.integers(0, len(_FIRST_NAMES), customers)]
    last_names = _LAST_NAMES[rng.integers(0, len(_LAST_NAMES), customers)]
    names = np.char.add(np.char.add(first_names, " "), last_names)
    emails = np.array([
        f"{first.lower().translate(_ASCII)}.{last.lower().translate(_ASCII)}{index}@ornek.com.tr"
        for index, (first, last) in enumerate(zip(first_names, last_names))
    ])
    phones = np.array([
        f"05{rng.integers(30, 60)} {rng.integers(100, 1000)} {rng.integers(10, 100)} {rng.integers(10, 100)}"
        for _ in range(customers)
    ])
    city_names = list(_CITIES)
    city_shares = np.array([share for share, _ in _CITIES.values()])
    customer_city = rng.choice(city_names, customers, p=city_shares / city_shares.sum())

    city = customer_city[owner]
    november = (order_time.month == 11)
    category_names = list(_CATEGORIES)
    category_probs = np.tile([share for share, _ in _CATEGORIES.values()], (rows, 1))
    category_probs[november, 0] *= 2.0  # Kasım kampanyası elektroniği öne çıkarır
    category_probs /= category_probs.sum(axis=1, keepdims=True)
    category = _choice(rng, category_names, category_probs)

    channel_probs = np.where(after_shift[:, None], [0.33, 0.52, 0.15], [0.45, 0.35, 0.20])
    channel = _choice(rng, ["Web", "Mobil Uygulama", "Mağaza"], channel_probs)
    payment_probs = np.where(after_shift[:, None], [0.78, 0.12, 0.10], [0.68, 0.10, 0.22])
    payment = _choice(rng, ["Kredi Kartı", "Havale/EFT", "Kapıda Ödeme"], payment_probs)

    base_price = np.array([_CATEGORIES[name][1] for name in category])
    unit_price = np.round(base_price * (1.02 ** months_since_start) * rng.lognormal(0, 0.35, rows), 2)
    quantity = 1 + rng.poisson(np.where(np.isin(category, ["Kozmetik", "Kitap"]), 0.8, 0.3))
    discount_options = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40])
    discount_probs = np.where(november[:, None], [0.25, 0.10, 0.20, 0.20, 0.15, 0.10],
                              [0.55, 0.15, 0.15, 0.10, 0.04, 0.01])
    discount = discount_options[(rng.random(rows)[:, None] > discount_probs.cumsum(axis=1)).sum(axis=1).clip(max=5)]
    amount = np.round(quantity * unit_price * (1 - discount), 2)

    last_days = order_time > END - pd.Timedelta(days=10)
    status = np.where(rng.random(rows) < 0.04, "İptal", "Teslim edildi")
    status = np.where(last_days & (status != "İptal") & (rng.random(rows) < 0.7), "Kargoda", status)
    usual_days = np.array([_CITIES[name][1] for name in city])
    ankara_problem = (city == "Ankara") & (order_time >= ANKARA_DELAY_START)
    delivery_days = usual_days + rng.poisson(1.0, rows) + np.where(ankara_problem, 4 + rng.poisson(1.5, rows), 0)
    delivery_days = np.where(channel == "Mağaza", 0, delivery_days)
    promised_days = np.where(channel == "Mağaza", 0, usual_days + 2)
    delay = delivery_days - promised_days
    delivered = status == "Teslim edildi"
    order_date = order_time.normalize()
    promised = order_date + pd.to_timedelta(promised_days, unit="D")
    delivered_at = pd.Series(order_date + pd.to_timedelta(delivery_days, unit="D")).where(delivered)

    late = np.clip(delay, 0, None)
    rating = np.clip(np.round(
        4.3 - 0.4 * late - 0.6 * (category == "Elektronik") + 0.4 * (category == "Kitap")
        + 0.5 * (channel == "Mağaza") + 0.3 * (discount >= 0.2) + rng.normal(0, 0.45, rows)
    ), 1, 5)
    rating = pd.Series(rating).where(delivered & (rng.random(rows) < 0.8))
    return_logit = (-3.6 + 0.9 * late + 2.2 * (category == "Giyim") + 2.0 * (discount >= 0.3)
                    - 1.5 * (channel == "Mağaza"))
    returned = delivered & (rng.random(rows) < 1 / (1 + np.exp(-return_logit)))

    in_experiment = (order_time >= EXPERIMENT_START) & (channel != "Mağaza")
    customer_group = np.where(rng.random(customers) < 0.5, "A", "B")
    group = np.where(in_experiment, customer_group[owner], None)
    upsell_rate = np.where(group == "B", 0.165, 0.12)
    upsell = rng.random(rows) < upsell_rate

    frame = pd.DataFrame({
        "sipariş_no": "",
        "müşteri_no": [f"MUS-{index + 1:05d}" for index in owner],
        "müşteri_adı": names[owner],
        "eposta": emails[owner],
        "telefon": phones[owner],
        "sipariş_tarihi": order_time,
        "şehir": city,
        "kanal": channel,
        "ödeme_tipi": payment,
        "kategori": category,
        "adet": quantity,
        "birim_fiyat": unit_price,
        "indirim_oranı": discount,
        "tutar": amount,
        "para_birimi": "TRY",
        "sipariş_durumu": status,
        "tahmini_teslim_tarihi": promised,
        "teslim_tarihi": delivered_at.to_numpy(),
        "değerlendirme_puanı": rating.to_numpy(),
        "iade_edildi": np.where(returned, "Evet", "Hayır"),
        "deney_grubu": group,
        "ek_ürün_eklendi": np.where(upsell, "Evet", "Hayır"),
    }).sort_values("sipariş_tarihi", kind="stable").reset_index(drop=True)
    frame["sipariş_no"] = [f"SP-{index + 1:06d}" for index in range(len(frame))]
    frame.loc[rng.random(len(frame)) < 0.03, "eposta"] = None
    return _inject_problems(frame, rng)


def _inject_problems(frame: pd.DataFrame, rng: np.random.Generator) -> DemoDataset:
    """Kural ihlallerini ve hijyen sorunlarını bilinen sayılarla ekler; değerler `injected` içinde kayıtlıdır."""
    frame = frame.copy()
    injected: dict[str, int] = {}

    def pick(count: int, mask: pd.Series | None = None) -> pd.Index:
        pool = frame.index if mask is None else frame.index[mask]
        return pd.Index(rng.choice(pool, size=count, replace=False))

    delivered = frame["sipariş_durumu"] == "Teslim edildi"
    rows = pick(4)
    frame.loc[rows, "tutar"] = -frame.loc[rows, "tutar"]
    injected["negatif_tutar"] = 4
    rows = pick(3, frame["indirim_oranı"] > 0)
    frame.loc[rows, "indirim_oranı"] = frame.loc[rows, "indirim_oranı"] * 100  # yüzde yerine oran yazılmış
    injected["indirim_oranı_1_üstü"] = 3
    rows = pick(5, delivered & (frame["kanal"] != "Mağaza"))
    frame.loc[rows, "teslim_tarihi"] = frame.loc[rows, "sipariş_tarihi"].dt.normalize() - pd.Timedelta(days=2)
    injected["teslim_siparişten_önce"] = 5
    rows = pick(3, frame["değerlendirme_puanı"].notna())
    frame.loc[rows, "değerlendirme_puanı"] = [0, 6, 6]
    injected["puan_1_5_dışı"] = 3
    rows = pick(6, frame["eposta"].notna())
    frame.loc[rows, "eposta"] = frame.loc[rows, "eposta"].str.replace("@", " at ", regex=False)
    injected["geçersiz_eposta"] = 6
    rows = pick(8, frame["şehir"] == "İstanbul")
    frame.loc[rows, "şehir"] = "istanbul"
    rows = pick(5, frame["şehir"] == "Ankara")
    frame.loc[rows, "şehir"] = "ANKARA"
    injected["şehir_yazım_farkı"] = 13
    rows = pick(2)
    neighbours = [index - 1 if index > 0 else index + 1 for index in rows]
    frame.loc[rows, "sipariş_no"] = frame.loc[neighbours, "sipariş_no"].to_numpy()
    injected["tekrarlanan_sipariş_no"] = 2
    duplicates = frame.loc[pick(12)]
    frame = pd.concat([frame, duplicates]).sort_index(kind="stable").reset_index(drop=True)
    injected["birebir_mükerrer_satır"] = 12
    return DemoDataset(frame=frame, injected=injected)


def write_demo_data(output_path: Path | str | None = None, seed: int = 7) -> Path:
    from core.ingestion import data_directory

    output = Path(output_path) if output_path is not None else data_directory() / DEMO_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_demo_data(seed).frame.copy()
    frame["sipariş_tarihi"] = frame["sipariş_tarihi"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for column in ("tahmini_teslim_tarihi", "teslim_tarihi"):
        frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
    frame.to_csv(output, index=False, encoding="utf-8-sig", lineterminator="\n")
    return output
