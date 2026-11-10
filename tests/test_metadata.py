from __future__ import annotations

import pandas as pd
import pytest

from core.metadata import MetadataGenerator

MASK = "[PII_MASKED]"


@pytest.mark.parametrize("value", [
    "ali@example.com",
    "0532 123 45 67",
    "+90 532 123 45 67",
    "05321234567",
    "(532) 123 45 67",
    "12345678901",
    "TR33 0006 1005 1978 6457 8413 26",
    "TR330006100519786457841326",
    "4543 1234 5678 9012",
    "4543-1234-5678-9012",
])
def test_pii_patterns_are_masked(value: str) -> None:
    masked = MetadataGenerator.mask_pii(f"İletişim: {value} lütfen")
    assert MASK in masked
    assert value not in masked
    assert not any(character.isdigit() for character in masked)


def test_ordinary_text_is_untouched() -> None:
    assert MetadataGenerator.mask_pii("İstanbul 2017 yılı 3 adet") == "İstanbul 2017 yılı 3 adet"


def test_numeric_id_column_does_not_leak_min_max() -> None:
    frame = pd.DataFrame({"tc_no": [12345678901, 98765432109, 55555555555], "yaş": [30, 41, 52]})
    card = MetadataGenerator().generate(frame)
    by_name = {column.name: column for column in card.columns}
    assert by_name["tc_no"].min is None and by_name["tc_no"].max is None and by_name["tc_no"].median is None
    assert set(by_name["tc_no"].sample_values) == {MASK}
    assert by_name["yaş"].min == 30 and by_name["yaş"].max == 52
    assert "12345678901" not in card.model_dump_json()


def test_long_integers_are_hidden_even_without_a_telling_name() -> None:
    frame = pd.DataFrame({"deger": [5321234567, 5329876543, 5551112233], "enlem": [-23.54562128115268, 41.0, 39.9]})
    card = MetadataGenerator().generate(frame)
    by_name = {column.name: column for column in card.columns}
    assert by_name["deger"].max is None and set(by_name["deger"].sample_values) == {MASK}
    assert by_name["enlem"].sample_values[0] == "-23.5456" and by_name["enlem"].max == 41.0


def test_sensitive_columns_are_fully_masked() -> None:
    frame = pd.DataFrame({
        "Ad Soyad": ["Ayşe Yılmaz", "Ali Çelik", "Can Öz"],
        "Adres": ["Atatürk Cad. No:5 Kadıköy", "Barış Sok. 12 Çankaya", "Lale Sk. 3 Konak"],
        "Şehir": ["İstanbul", "Ankara", "İzmir"],
    })
    card = MetadataGenerator().generate(frame)
    by_name = {column.name: column for column in card.columns}
    assert set(by_name["Ad Soyad"].sample_values) == {MASK}
    assert set(by_name["Adres"].sample_values) == {MASK}
    assert by_name["Şehir"].sample_values == ["İstanbul", "Ankara", "İzmir"]


def test_sample_dataset_card_contains_no_customer_names(dirty: pd.DataFrame) -> None:
    payload = MetadataGenerator().generate(dirty).model_dump_json()
    for name in ("Ayşe", "Yılmaz", "Mehmet", "Kaya", "Fatma", "Demir", "Çelik", "Zeynep", "Şahin"):
        assert name not in payload


def test_high_cardinality_text_is_shape_masked(orders: pd.DataFrame) -> None:
    card = MetadataGenerator().generate(orders)
    payload = card.model_dump_json()
    assert orders["order_id"].iloc[0] not in payload
    assert orders["customer_id"].iloc[0] not in payload


def test_pii_in_column_name_is_masked() -> None:
    card = MetadataGenerator().generate(pd.DataFrame({"ali@example.com notu": [1, 2, 3]}))
    assert "ali@example.com" not in card.model_dump_json()
