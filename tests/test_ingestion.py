"""Dosya okuma: ayraç tespiti, baştaki sıfırlar, desteklenmeyen biçimler ve okunaksız dosyalar."""

from __future__ import annotations

import pandas as pd
import pytest

from core.ingestion import detect_delimiter, list_local_csv_files, read_tabular_file


def test_semicolon_file_with_commas_inside_quotes() -> None:
    raw = (
        'kod;ad;açıklama\n'
        '1;Kalem;"Mavi, siyah, kırmızı renkler"\n'
        '2;Defter;"Kareli, 80 yaprak"\n'
        '3;Silgi;"Beyaz"\n'
    ).encode("utf-8")
    frame = read_tabular_file(raw, "urunler.csv")
    assert list(frame.columns) == ["kod", "ad", "açıklama"]
    assert frame["açıklama"].tolist() == ["Mavi, siyah, kırmızı renkler", "Kareli, 80 yaprak", "Beyaz"]


def test_comma_file_with_semicolons_inside_quotes() -> None:
    raw = 'id,not,tutar\n1,"a; b; c",10\n2,"d; e",20\n3,"f",30\n'.encode("utf-8")
    frame = read_tabular_file(raw, "x.csv")
    assert list(frame.columns) == ["id", "not", "tutar"] and frame["tutar"].tolist() == [10, 20, 30]


def test_turkish_decimal_comma_prefers_semicolon() -> None:
    raw = "şehir;tutar;oran\nİzmir;1,5;2,5\nBursa;3,5;4,5\n".encode("cp1254")
    frame = read_tabular_file(raw, "x.csv")
    assert list(frame.columns) == ["şehir", "tutar", "oran"]


@pytest.mark.parametrize(("text", "expected"), [
    ("a,b,c\n1,2,3\n", ","), ("a;b;c\n1;2;3\n", ";"), ("a\tb\n1\t2\n", "\t"),
    ('a;b\n"x, y, z";1\n"q";2\n', ";"),
])
def test_detect_delimiter(text: str, expected: str) -> None:
    assert detect_delimiter(text) == expected


def test_leading_zeros_are_kept_but_numbers_stay_numeric() -> None:
    raw = "posta_kodu;miktar;fiyat\n00123;5;1.5\n04567;7;2.5\n12345;9;3.0\n".encode("utf-8")
    frame = read_tabular_file(raw, "x.csv")
    assert frame["posta_kodu"].tolist() == ["00123", "04567", "12345"]
    assert pd.api.types.is_integer_dtype(frame["miktar"]) and pd.api.types.is_float_dtype(frame["fiyat"])


def test_unsupported_and_broken_files_raise_turkish_errors() -> None:
    with pytest.raises(ValueError, match=r"\.xls\) biçiminde; bu biçim desteklenmiyor"):
        read_tabular_file(b"anything", "eski.xls")
    with pytest.raises(ValueError, match="boş bir dosya"):
        read_tabular_file(b"   \n", "bos.csv")
    with pytest.raises(ValueError, match="Excel dosyası olarak açılamadı"):
        read_tabular_file(b"bu bir excel degil", "sahte.xlsx")
    with pytest.raises(ValueError, match="sütun sayısı tutarsız"):
        read_tabular_file('a,b\n1,2\n3,4,5,6\n"kapanmayan,7\n'.encode("utf-8"), "bozuk.csv")


def test_data_directory_can_be_overridden(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "b.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("x\n1\n", encoding="utf-8")
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    assert [path.name for path in list_local_csv_files()] == ["a.csv", "b.csv"]


def test_app_shows_turkish_error_for_unreadable_local_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    from tests.conftest import ROOT

    (tmp_path / "bozuk.csv").write_text('a,b\n1,2\n3,4,5,6\n"kapanmayan,7\n', encoding="utf-8")
    monkeypatch.setenv("AUTOPILOT_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    app.run()
    app.sidebar.selectbox[0].select("bozuk.csv").run()
    assert not app.exception
    assert any(error.value.startswith("Dosya yüklenemedi: “bozuk.csv” okunamadı") for error in app.error)
