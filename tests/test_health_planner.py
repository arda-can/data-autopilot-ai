from __future__ import annotations

import pandas as pd

from core.health import DataHealthAnalyzer
from core.planner import GeminiPlanner


def test_health_report_basics(dirty: pd.DataFrame) -> None:
    report = DataHealthAnalyzer().analyze(dirty)
    assert report.satır_sayısı == 200
    assert set(report.sabit_sütunlar) == {"Sabit Sütun", "Notlar"}
    assert set(report.yüksek_eksik_sütunlar) == {"Neredeyse Boş", "Notlar"}
    assert report.mükerrer_satır_sayısı == 6
    assert 0 <= report.kalite_skoru < 100


def test_dirty_numeric_text_is_detected(dirty: pd.DataFrame) -> None:
    report = DataHealthAnalyzer().analyze(dirty)
    assert report.kirli_sayısal_sütunlar == ["Harcama"]


def test_fallback_plan_recognises_text_columns(dirty: pd.DataFrame) -> None:
    plan = GeminiPlanner(None).create_plan(dirty)
    actions = {item.column: item.action for item in plan.actions}
    assert actions["Harcama"] == "parse_currency"
    assert actions["Müşteri Adı"] == "strip_whitespace"
    assert actions["Notlar"] == "drop_column"


def test_fallback_plan_never_proposes_statistical_actions() -> None:
    frame = pd.DataFrame({"ölçüm": [1.0, None, 3.0] * 10, "şehir": ["Ankara", None, "İzmir"] * 10})
    plan = GeminiPlanner(None).create_plan(frame)
    statistical = {"impute_median", "impute_mode", "cap_outliers_iqr"}
    assert not statistical & {item.action for item in plan.actions}
