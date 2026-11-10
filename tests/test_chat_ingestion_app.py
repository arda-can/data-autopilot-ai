from __future__ import annotations

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.autopilot import AutopilotResult
from core.chat import DataChatEngine, StructuredDataQuery
from core.ingestion import list_local_csv_files, read_tabular_file
from core.metadata import MetadataGenerator
from tests.conftest import ROOT


def test_chat_fallback_group_average(dirty_result: AutopilotResult) -> None:
    table, summary, _figure, query = DataChatEngine(None).answer(
        dirty_result.cleaned_frame, "Şehirlere göre ortalama harcama nedir?", dirty_result.metadata
    )
    assert (query.operation, query.group_col, query.target_col, query.agg_func) == (
        "groupby_agg", "Şehir", "Harcama", "mean",
    )
    assert len(table) == 5 and summary


def test_chat_rejects_unknown_columns(dirty: pd.DataFrame) -> None:
    query = StructuredDataQuery(operation="value_counts", group_col="yok_böyle_sütun", explanation="x")
    with pytest.raises(ValueError):
        DataChatEngine._validate_columns(query, dirty)


def test_chat_prompt_masks_pii_in_question(dirty: pd.DataFrame) -> None:
    card = MetadataGenerator().generate(dirty)
    prompt = DataChatEngine(None)._build_prompt("ali@example.com ve 0532 123 45 67 için harcama nedir?", card)
    assert "ali@example.com" not in prompt and "0532 123 45 67" not in prompt
    assert "harcama nedir?" in prompt


def test_semicolon_cp1254_file_is_read() -> None:
    raw = "Şehir;Tutar;Müşteri\nİstanbul;1,5;Ayşe Güneş\nÇanakkale;2,5;Ömer Öztürk\n".encode("cp1254")
    frame = read_tabular_file(raw, "ornek.csv")
    assert list(frame.columns) == ["Şehir", "Tutar", "Müşteri"]
    assert frame["Şehir"].tolist() == ["İstanbul", "Çanakkale"]


def test_sample_dataset_is_listed() -> None:
    assert "kirli_musteri_verisi.csv" in [path.name for path in list_local_csv_files()]


def test_app_runs_end_to_end() -> None:
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    app.sidebar.selectbox[0].select("kirli_musteri_verisi.csv").run()
    assert not app.exception and not app.error
    assert any("Yönetici Özeti" in header.value for header in app.header)
    app.text_input(key="autopilot_chat_question").set_value("Şehirlere göre ortalama harcama nedir?")
    app.button(key="autopilot_chat_button").click().run()
    assert not app.exception and not app.error
