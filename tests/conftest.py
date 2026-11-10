"""Ortak test verileri."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.autopilot import AutopilotEngine, AutopilotResult
from core.ingestion import read_local_csv

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ISO_FORMAT = "%Y-%m-%d %H:%M:%S"

# ANALIZ.md'deki henüz düzeltilmemiş hataların testleri. Her düzeltmede ilgili satır silinir.
KNOWN_FAILURES: dict[str, list[str]] = {
    "Y3": [
        "test_measurement_columns_are_kept_as_features", "test_column_containing_id_substring_can_be_target",
        "test_regression_learns_known_signal",
    ],
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    pending = {name: tag for tag, names in KNOWN_FAILURES.items() for name in names}
    for item in items:
        tag = pending.get(getattr(item, "originalname", None) or item.name)
        if tag:
            item.add_marker(pytest.mark.xfail(reason=f"ANALIZ.md {tag}: henüz düzeltilmedi", strict=False))


@pytest.fixture(scope="session")
def orders_raw() -> pd.DataFrame:
    """Olist benzeri sentetik sipariş tablosu (hash kimlikler, ISO tarihler, eksikler)."""
    return read_local_csv(FIXTURES / "olist_benzeri_siparisler.csv")


@pytest.fixture
def orders(orders_raw: pd.DataFrame) -> pd.DataFrame:
    return orders_raw.copy()


@pytest.fixture(scope="session")
def dirty_raw() -> pd.DataFrame:
    return read_local_csv(ROOT / "data" / "kirli_musteri_verisi.csv")


@pytest.fixture
def dirty(dirty_raw: pd.DataFrame) -> pd.DataFrame:
    return dirty_raw.copy()


@pytest.fixture(scope="session")
def orders_result(orders_raw: pd.DataFrame) -> AutopilotResult:
    return AutopilotEngine().run_full_autopilot(orders_raw.copy())


@pytest.fixture(scope="session")
def dirty_result(dirty_raw: pd.DataFrame) -> AutopilotResult:
    return AutopilotEngine().run_full_autopilot(dirty_raw.copy())
