"""Güvenli ve deterministik veri alma yardımcıları."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import chardet
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENCODINGS = ("utf-8", "cp1254", "iso-8859-9")
DELIMITERS = (",", ";", "\t")


def list_local_csv_files(data_dir: Path | str = DATA_DIR) -> list[Path]:
    """Yerel data klasöründeki CSV dosyalarını alfabetik listeler."""
    directory = Path(data_dir)
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


def detect_encoding(raw: bytes) -> str:
    detected = chardet.detect(raw).get("encoding")
    candidates = [detected, *ENCODINGS]
    for encoding in candidates:
        if encoding:
            try:
                raw.decode(encoding)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue
    return "utf-8"


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    scores = {delimiter: sample.count(delimiter) for delimiter in DELIMITERS}
    return max(scores, key=scores.get) if max(scores.values()) else ","


def read_tabular_file(source: Any, filename: str | None = None) -> pd.DataFrame:
    """CSV veya Excel kaynağını otomatik encoding/delimiter ile okur."""
    name = filename or getattr(source, "name", str(source))
    suffix = Path(name).suffix.lower()
    raw = _read_bytes(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(raw))

    encoding = detect_encoding(raw)
    text = raw.decode(encoding)
    delimiter = detect_delimiter(text)
    return pd.read_csv(BytesIO(raw), encoding=encoding, sep=delimiter)


def read_local_csv(path: Path | str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() != ".csv":
        raise ValueError("Yerel dosya CSV olmalıdır.")
    return read_tabular_file(file_path, file_path.name)


def generate_sample_dirty_data(output_path: Path | str = DATA_DIR / "kirli_musteri_verisi.csv") -> Path:
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
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    return output
