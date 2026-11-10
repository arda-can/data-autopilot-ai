"""Sütun tipi çıkarımı: metin, sayı, tarih ve kimlik tespitinin tek doğruluk kaynağı."""

from __future__ import annotations

import re

import pandas as pd

_PYTHON_STRING = pd.StringDtype(storage="python")
_PROBE_SIZE = 2000

# --- Metin sütunları -------------------------------------------------------------------------


def is_text_series(series: pd.Series) -> bool:
    """pandas 2 (`object`) ve pandas 3 (`str`) metin sütunlarını birlikte tanır."""
    return bool(pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))


def is_text_dtype_name(dtype: str) -> bool:
    return dtype in {"object", "str"} or dtype.startswith("string")


def text_columns(frame: pd.DataFrame) -> list:
    return [column for column in frame.columns if is_text_series(frame[column])]


def _as_text(series: pd.Series) -> pd.Series:
    """Regex işlemleri için Python `re` semantiğiyle çalışan, kırpılmış metin serisi."""
    return series.astype(_PYTHON_STRING).str.strip()


def _probe(series: pd.Series) -> pd.Series:
    present = series.dropna()
    return present.sample(_PROBE_SIZE, random_state=0) if len(present) > _PROBE_SIZE else present


# --- Sütun adı sezgileri ---------------------------------------------------------------------

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-zçğıöşü0-9])(?=[A-ZÇĞİÖŞÜ])")
_TOKEN_SPLIT = re.compile(r"[^0-9a-zçğıöşü]+")
_ID_TOKENS = frozenset({
    "id", "ids", "uuid", "guid", "kimlik", "tckn", "tc", "zip", "posta", "postcode", "postal",
    "code", "kod", "kodu", "no", "numara", "numarası", "numarasi",
})
_PERSON_NAME_TOKENS = frozenset({
    "adı", "adi", "soyad", "soyadı", "soyadi", "adsoyad", "isim", "ismi",
    "firstname", "lastname", "fullname", "surname",
})
_PERSON_QUALIFIERS = frozenset({
    "first", "last", "full", "customer", "user", "client", "person", "contact",
    "müşteri", "musteri", "kullanıcı", "kullanici",
})
_GEO_TOKENS = frozenset({"lat", "lng", "lon", "long", "latitude", "longitude", "enlem", "boylam"})


def name_tokens(name: object) -> list[str]:
    """Sütun adını kelimelere böler: `customerId`, `order_id`, `Müşteri Adı` → kelime listesi."""
    spaced = _CAMEL_BOUNDARY.sub("_", str(name)).replace("İ", "i").lower()
    return [token for token in (part.rstrip("0123456789") for part in _TOKEN_SPLIT.split(spaced)) if token]


def is_identifier_name(name: object) -> bool:
    """Ad, kimlik/kod/kişi adı sütununa işaret ediyor mu? Alt dize değil, tam kelime eşleşir."""
    tokens = name_tokens(name)
    token_set = set(tokens)
    if token_set & _ID_TOKENS or token_set & _PERSON_NAME_TOKENS:
        return True
    if tokens == ["ad"]:
        return True
    return "name" in token_set and (len(tokens) == 1 or bool(token_set & _PERSON_QUALIFIERS))


def is_identifier_column(series: pd.Series, name: object) -> bool:
    """Ada göre kimlik ya da neredeyse her satırı benzersiz metin sütunu."""
    if is_identifier_name(name):
        return True
    return bool(
        not pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_datetime64_any_dtype(series)
        and series.nunique(dropna=True) >= max(50, int(len(series) * 0.95))
    )


def is_geo_coordinate_name(name: object) -> bool:
    return bool(set(name_tokens(name)) & _GEO_TOKENS)


# --- Sayısal metin ---------------------------------------------------------------------------

_CURRENCY = r"(?:₺|\$|€|£|tl|try|usd|eur)"
_LEADING_UNIT = rf"^(?:{_CURRENCY}|%)\s*"
_TRAILING_UNIT = rf"\s*(?:{_CURRENCY}|%)$"
_TR_CURRENCY = r"₺|(?:^|[^a-z])(?:tl|try)(?:$|[^a-z])"
_TR_NUMBER = r"[+-]?(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?"
_US_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CODE_LIKE = r"[+-]?0\d.*|\d{16,}"


def parse_numeric_text(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Para/sayı biçimli metni sayıya çevirir.

    Bir değer yalnızca *tamamı* sayı biçimindeyse geçerlidir (birim ve para simgesi hariç);
    içinde rakam geçen metinler, hash'ler, tarihler ve baştaki sıfırı anlamlı kodlar sayılmaz.
    Türkçe (`1.250,50`) ve ABD (`1,250.50`) biçimi sütun genelindeki kanıta göre seçilir.
    Dönüş: (float64 sayılar, geçerlilik maskesi).
    """
    text = _as_text(series)
    body = (
        text.str.replace(_LEADING_UNIT, "", regex=True, flags=re.IGNORECASE)
        .str.replace(_TRAILING_UNIT, "", regex=True, flags=re.IGNORECASE)
        .str.strip()
    )
    code_like = body.str.fullmatch(_CODE_LIKE, na=False).astype(bool)
    tr_ok = body.str.fullmatch(_TR_NUMBER, na=False).astype(bool) & ~code_like
    us_ok = body.str.fullmatch(_US_NUMBER, na=False).astype(bool) & ~code_like
    tr_votes = int((tr_ok & ~us_ok).sum())
    us_votes = int((us_ok & ~tr_ok).sum())
    if tr_votes != us_votes:
        turkish = tr_votes > us_votes
    else:
        turkish = bool(text.str.contains(_TR_CURRENCY, regex=True, flags=re.IGNORECASE, na=False).any())
    if turkish:
        valid = tr_ok
        normalized = body.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        valid = us_ok
        normalized = body.str.replace(",", "", regex=False)
    numbers = pd.to_numeric(normalized.where(valid), errors="coerce").astype("float64")
    return numbers, valid


def numeric_text_ratio(series: pd.Series) -> float:
    """Boş olmayan değerlerin ne kadarı sayı biçiminde? Büyük sütunlarda örneklemle tahmin eder."""
    probe = _probe(series)
    if probe.empty:
        return 0.0
    return float(parse_numeric_text(probe)[1].mean())


# --- Tarih metni -----------------------------------------------------------------------------

_TIME_SUFFIX = r"(?:[ T]\d{1,2}:\d{2}.*)?"
_YEAR_FIRST = rf"\d{{4}}[-/.]\d{{1,2}}[-/.]\d{{1,2}}{_TIME_SUFFIX}"
_DAY_FIRST = rf"\d{{1,2}}[-/.]\d{{1,2}}[-/.]\d{{2,4}}{_TIME_SUFFIX}"
_DAY_FIRST_FORMATS = tuple(
    f"%d{separator}%m{separator}%Y{time}"
    for separator in (".", "/", "-")
    for time in ("", " %H:%M:%S", " %H:%M")
)


def parse_datetime_text(series: pd.Series, min_ratio: float = 0.80) -> pd.Series | None:
    """Metin sütunu tarihse ayrıştırır, değilse `None` döner.

    Yıl-önce (ISO) biçim asla gün-önce varsayımıyla okunmaz; `2017-10-02` her zaman 2 Ekim'dir.
    Gün-önce biçimler (`02.10.2017`) Türkiye'deki kullanıma göre yorumlanır.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if not is_text_series(series):
        return None
    text = _as_text(series)
    present = int(text.notna().sum())
    probe = _probe(text)
    if probe.empty:
        return None
    year_first = float(probe.str.fullmatch(_YEAR_FIRST, na=False).mean())
    day_first = float(probe.str.fullmatch(_DAY_FIRST, na=False).mean())
    if max(year_first, day_first) < min_ratio:
        return None

    def ratio(parsed: pd.Series) -> float:
        return float(parsed.notna().sum() / present)

    if year_first >= day_first:
        best = pd.to_datetime(text, errors="coerce", format="ISO8601")
        if ratio(best) < min_ratio:
            best = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
    else:
        best = max(
            (pd.to_datetime(text, errors="coerce", format=fmt) for fmt in _DAY_FIRST_FORMATS),
            key=ratio,
        )
        if ratio(best) < min_ratio:
            best = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
    return best if ratio(best) >= min_ratio else None
