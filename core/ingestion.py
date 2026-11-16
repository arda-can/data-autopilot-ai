"""Güvenli ve deterministik veri alma yardımcıları."""

from __future__ import annotations

import csv
import os
import zipfile
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import chardet
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENCODINGS = ("utf-8", "cp1254", "iso-8859-9")
# Eşitlikte sıradaki önce gelir: Türkçe verilerde ondalık virgül yaygın olduğu için ";" virgülden önce.
DELIMITERS = ("\t", ";", ",")
_DELIMITER_SAMPLE_LINES = 50


def data_directory() -> Path:
    """Yerel veri klasörü; `AUTOPILOT_DATA_DIR` ortam değişkeniyle değiştirilebilir (testler bunu kullanır)."""
    return Path(os.environ.get("AUTOPILOT_DATA_DIR") or DATA_DIR)


def list_local_csv_files(data_dir: Path | str | None = None) -> list[Path]:
    """Yerel data klasöründeki CSV dosyalarını alfabetik listeler."""
    directory = Path(data_dir) if data_dir is not None else data_directory()
    return sorted(directory.glob("*.csv"), key=lambda path: path.name.lower())


def _read_bytes(source: Any) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        return source.read()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    raise TypeError("Veri kaynağı bir dosya yolu, yüklenen dosya veya byte olmalıdır.")


_TURKISH_LETTERS = set("çğıöşüÇĞİÖŞÜ")


def detect_encoding(raw: bytes) -> str:
    """Önce UTF-8; değilse adaylar arasından en çok Türkçe harf üreten 8 bitlik kodlama.

    chardet küçük cp1254 dosyalarda başka bir kodlama tahmin edebiliyor ("şehir" → "˛ehir"); çözülebilmek
    tek başına doğruluk kanıtı olmadığından Türkçe harf sayısı belirleyicidir, eşitlikte chardet tahmini.
    """
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass
    detected = chardet.detect(raw).get("encoding")
    best, best_score = "utf-8", -1
    for encoding in dict.fromkeys(item for item in (detected, "cp1254", "iso-8859-9") if item):
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        score = sum(character in _TURKISH_LETTERS for character in text)
        if score > best_score:
            best, best_score = encoding, score
    return best


def detect_delimiter(text: str) -> str:
    """Tırnak içindeki ayraçları saymaz.

    Her aday için ilk satırlar `csv` modülüyle (tırnak kurallarıyla) ayrıştırılır; satırların en tutarlı
    alan sayısını verdiği ayraç seçilir, eşitlikte daha çok sütun veren. Böylece `;` ayrımlı bir dosyada
    açıklama metnindeki virgüller ayraç sanılmaz.
    """
    sample = "\n".join(text.splitlines()[:_DELIMITER_SAMPLE_LINES])
    best, best_score = ",", (0.0, 1)
    for delimiter in DELIMITERS:
        try:
            rows = [row for row in csv.reader(StringIO(sample), delimiter=delimiter) if row]
        except csv.Error:
            continue
        if len(rows) > 2:
            rows = rows[:-1]  # örneklem sınırında kesilmiş olabilecek son satır
        if not rows:
            continue
        width, frequency = Counter(len(row) for row in rows).most_common(1)[0]
        if width < 2:
            continue
        score = (frequency / len(rows), width)
        if score > best_score:
            best, best_score = delimiter, score
    return best


_LEADING_ZERO = r"^[+-]?0\d"


def _leading_zero_columns(raw: bytes, encoding: str, delimiter: str) -> dict[str, type]:
    """Değerlerinde baştaki sıfır bulunan sütunlar (00123, 01037): sayı sanılırsa sıfırlar kaybolur."""
    probe = pd.read_csv(BytesIO(raw), encoding=encoding, sep=delimiter, dtype=str)
    return {
        column: str for column in probe.columns
        if probe[column].dropna().str.match(_LEADING_ZERO).any()
    }


def read_tabular_file(source: Any, filename: str | None = None) -> pd.DataFrame:
    """CSV veya Excel kaynağını otomatik encoding/ayraç ile okur; hataları Türkçe `ValueError` olarak verir."""
    name = filename or getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    if suffix == ".xls":
        raise ValueError(
            f"“{name}” eski Excel (.xls) biçiminde; bu biçim desteklenmiyor. Dosyayı .xlsx veya .csv olarak kaydedin."
        )
    raw = _read_bytes(source)
    if not raw.strip():
        raise ValueError(f"“{name}” boş bir dosya.")
    try:
        if suffix == ".xlsx":
            try:
                return pd.read_excel(BytesIO(raw))
            except ValueError as exc:
                raise ValueError(f"“{name}” Excel dosyası olarak açılamadı: {exc}") from exc
        encoding = detect_encoding(raw)
        delimiter = detect_delimiter(raw.decode(encoding))
        return pd.read_csv(
            BytesIO(raw), encoding=encoding, sep=delimiter,
            dtype=_leading_zero_columns(raw, encoding, delimiter) or None,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"“{name}” okunamadı: dosyada veri yok.") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise ValueError(
            f"“{name}” okunamadı: satırlardaki sütun sayısı tutarsız (ayraç veya tırnak hatası olabilir). "
            f"Ayrıntı: {exc}"
        ) from exc
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        raise ValueError(f"“{name}” Excel dosyası olarak açılamadı: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"“{name}” karakter kodlaması çözülemedi: {exc}") from exc


def read_local_csv(path: Path | str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() != ".csv":
        raise ValueError("Yerel dosya CSV olmalıdır.")
    return read_tabular_file(file_path, file_path.name)


def generate_sample_dirty_data(output_path: Path | str | None = None) -> Path:
    """Temizleme akışını sınamak için deterministik 200 satırlık kirli veri üretir."""
    rng = np.random.default_rng(42)
    rows = 200
    names = np.array([" Ayşe Yılmaz ", "Mehmet Kaya", "Fatma Demir ", "Ali Çelik", "Zeynep Şahin"])
    cities = np.array(["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"])
    amounts = np.array(["1.250,50 TL", "250,00 TL", "99,90 TL", "2.000,00 TL", "75 TL"])
    frame = pd.DataFrame(
        {
            "Müşteri Adı": names[rng.integers(0, len(names), rows)],
            "Şehir": cities[rng.integers(0, len(cities), rows)],
            "Yaş": rng.integers(18, 76, rows),
            "Harcama": amounts[rng.integers(0, len(amounts), rows)],
            "Sabit Sütun": "tek_değer",
            "Neredeyse Boş": [None] * 190 + ["var"] * 10,
            "Notlar": [None] * rows,
        }
    )
    frame.loc[[3, 17], "Yaş"] = [-5, 999]
    frame.loc[[8, 9, 10], "Harcama"] = ["1.250,50 TL", "1.250,50 TL", "1.250,50 TL"]
    frame = pd.concat([frame, frame.iloc[[0, 1, 2]]], ignore_index=True).iloc[:rows]
    output = Path(output_path) if output_path is not None else data_directory() / "kirli_musteri_verisi.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output
