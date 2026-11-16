"""Gemini yolu: sahte SDK yanıtıyla JSON doğrulaması ve hata görünürlüğü (ağ erişimi yok)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from core.autopilot import AutopilotEngine
from core.chat import DataChatEngine
from core.planner import AnalysisPlan, GeminiPlanner

PLAN_JSON = json.dumps({
    "domain": "Müşteri", "problem_type": "Regresyon",
    "actions": [{"column": "Harcama", "action": "parse_currency", "reason": "Para biçiminde."}],
}, ensure_ascii=False)


class FakeClient:
    """`genai.Client` yerine: `parsed` alanı boş, yalnızca JSON metni dönen yanıt (SDK'nın bazı sürümleri)."""

    response_text = PLAN_JSON
    error: Exception | None = None
    prompts: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.models = self

    def generate_content(self, model: str, contents: str, config) -> SimpleNamespace:
        FakeClient.prompts.append(contents)
        if FakeClient.error is not None:
            raise FakeClient.error
        return SimpleNamespace(parsed=None, text=FakeClient.response_text)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch):
    import google.genai

    FakeClient.response_text, FakeClient.error, FakeClient.prompts = PLAN_JSON, None, []
    monkeypatch.setattr(google.genai, "Client", FakeClient)
    return FakeClient


def test_plan_json_text_is_validated_not_discarded(fake_sdk, dirty: pd.DataFrame) -> None:
    planner = GeminiPlanner("sahte-anahtar")
    plan = planner.create_plan(dirty)
    assert isinstance(plan, AnalysisPlan) and plan.domain == "Müşteri"
    assert [(item.column, item.action) for item in plan.actions] == [("Harcama", "parse_currency")]
    assert planner.origin.source == "Gemini" and not planner.origin.error
    assert "Ayşe" not in fake_sdk.prompts[0]  # LLM'e giden metin maskeli


def test_gemini_failure_is_visible(fake_sdk, dirty: pd.DataFrame) -> None:
    fake_sdk.error = RuntimeError("kota aşıldı")
    planner = GeminiPlanner("sahte-anahtar")
    plan = planner.create_plan(dirty)
    assert plan.domain == "Genel yapılandırılmış veri"  # yerel plan
    assert planner.origin.source == "yerel kurallar" and planner.origin.error
    assert "RuntimeError: kota aşıldı" in planner.origin.reason


def test_invalid_json_is_visible(fake_sdk, dirty: pd.DataFrame) -> None:
    fake_sdk.response_text = '{"domain": "eksik alanlar"}'
    planner = GeminiPlanner("sahte-anahtar")
    planner.create_plan(dirty)
    assert planner.origin.error and "ValidationError" in planner.origin.reason


def test_no_key_is_explained(dirty: pd.DataFrame) -> None:
    planner = GeminiPlanner(None)
    planner.create_plan(dirty)
    assert planner.origin.source == "yerel kurallar" and not planner.origin.error
    assert planner.origin.reason == "Gemini API anahtarı girilmedi."


def test_autopilot_carries_plan_origin(fake_sdk, dirty: pd.DataFrame) -> None:
    result = AutopilotEngine().run_full_autopilot(dirty, api_key="sahte-anahtar")
    assert result.plan_origin is not None and result.plan_origin.source == "Gemini"


def test_chat_uses_gemini_query_and_reports_failures(fake_sdk, dirty_result) -> None:
    frame = dirty_result.cleaned_frame
    fake_sdk.response_text = json.dumps({
        "operation": "groupby_agg", "group_col": "Şehir", "target_col": "Harcama", "agg_func": "max",
        "explanation": "Şehirlere göre en yüksek harcama.",
    }, ensure_ascii=False)
    engine = DataChatEngine("sahte-anahtar")
    _, _, _, query = engine.answer(frame, "şehirlere göre en büyük harcama")
    assert query.agg_func == "max" and engine.origin.source == "Gemini"

    fake_sdk.response_text = json.dumps({"operation": "value_counts", "group_col": "yok", "explanation": "x"})
    _, _, _, query = engine.answer(frame, "şehirlere göre ortalama harcama")
    assert engine.origin.error and "geçersiz sütun" in engine.origin.reason
    assert query.group_col == "Şehir"  # yerel ayrıştırıcı devraldı
