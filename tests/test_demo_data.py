"""Sentetik demo veri seti: bilinen örüntüler ve bilinçli ihlaller gerçekten veride olmalı."""

from __future__ import annotations

import pandas as pd
import pytest

from core.autopilot import AutopilotEngine, AutopilotResult
from core.automl import AutoMLEngine
from core.demo_data import (
    ANKARA_DELAY_START, DEMO_FILENAME, EXPERIMENT_START, DemoDataset, generate_demo_data, write_demo_data,
)
from core.ingestion import read_local_csv
from core.metadata import PII_MASK, MetadataGenerator
from tests.conftest import ROOT


@pytest.fixture(scope="module")
def demo() -> DemoDataset:
    return generate_demo_data()


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> AutopilotResult:
    path = write_demo_data(tmp_path_factory.mktemp("demo") / DEMO_FILENAME)
    return AutopilotEngine().run_full_autopilot(read_local_csv(path))


def test_generation_is_deterministic(demo: DemoDataset) -> None:
    pd.testing.assert_frame_equal(generate_demo_data().frame, demo.frame)


def test_committed_csv_matches_generator(tmp_path) -> None:
    """data/ altındaki dosya üreticiyle aynı kalmalı; üretici değişirse dosya da yeniden yazılmalı."""
    fresh = read_local_csv(write_demo_data(tmp_path / DEMO_FILENAME))
    pd.testing.assert_frame_equal(fresh, read_local_csv(ROOT / "data" / DEMO_FILENAME), check_exact=False)


def test_injected_problems_are_present(demo: DemoDataset) -> None:
    frame, injected = demo.frame, demo.injected
    assert (frame["tutar"] < 0).sum() == injected["negatif_tutar"] == 4
    assert (frame["indirim_oranı"] > 1).sum() == injected["indirim_oranı_1_üstü"] == 3
    delivered = pd.to_datetime(frame["teslim_tarihi"])
    assert (delivered < frame["sipariş_tarihi"].dt.normalize()).sum() == injected["teslim_siparişten_önce"]
    assert (~frame["değerlendirme_puanı"].dropna().between(1, 5)).sum() == injected["puan_1_5_dışı"]
    assert (frame["eposta"].str.contains(" at ", na=False)).sum() == injected["geçersiz_eposta"]
    assert frame.duplicated().sum() == injected["birebir_mükerrer_satır"] == 12
    distinct = frame.drop_duplicates()
    assert distinct["sipariş_no"].duplicated().sum() == injected["tekrarlanan_sipariş_no"]
    assert {"istanbul", "ANKARA"} <= set(frame["şehir"])


def test_customers_repeat_and_cohorts_exist(demo: DemoDataset) -> None:
    orders_per_customer = demo.frame.drop_duplicates().groupby("müşteri_no").size()
    assert (orders_per_customer > 1).mean() > 0.3
    cohorts = demo.frame.groupby("müşteri_no")["sipariş_tarihi"].min().dt.to_period("M").nunique()
    assert cohorts == 24


def test_experiment_window_and_effect(demo: DemoDataset) -> None:
    frame = demo.frame.drop_duplicates()
    grouped = frame[frame["deney_grubu"].notna()]
    assert grouped["sipariş_tarihi"].min() >= EXPERIMENT_START
    assert set(grouped["deney_grubu"]) == {"A", "B"} and "Mağaza" not in set(grouped["kanal"])
    assert grouped.groupby("müşteri_no")["deney_grubu"].nunique().max() == 1  # müşteri bazında atama
    rates = grouped.groupby("deney_grubu")["ek_ürün_eklendi"].apply(lambda values: (values == "Evet").mean())
    assert rates["B"] > rates["A"] + 0.02


def test_shift_and_ankara_delay(demo: DemoDataset) -> None:
    frame = demo.frame.drop_duplicates()
    late = frame["sipariş_tarihi"] >= pd.Timestamp("2025-07-01")
    mobile = frame["kanal"] == "Mobil Uygulama"
    assert mobile[late].mean() > mobile[~late].mean() + 0.1
    days = (pd.to_datetime(frame["teslim_tarihi"]) - frame["sipariş_tarihi"].dt.normalize()).dt.days
    ankara = (frame["şehir"] == "Ankara") & (frame["kanal"] != "Mağaza")
    after = frame["sipariş_tarihi"] >= ANKARA_DELAY_START
    assert days[ankara & after].mean() > days[ankara & ~after].mean() + 3


def test_pii_is_masked_in_metadata(demo: DemoDataset) -> None:
    card = MetadataGenerator().generate(demo.frame)
    by_name = {column.name: column for column in card.columns}
    for name in ("müşteri_adı", "eposta", "telefon"):
        assert set(by_name[name].sample_values) == {PII_MASK}
    text = card.model_dump_json()
    assert "@ornek.com.tr" not in text and "Yılmaz" not in text


def test_autopilot_reports_injected_problems(demo_result: AutopilotResult) -> None:
    after = demo_result.after_report
    assert [(item.column, item.count) for item in after.domain_findings] == [("tutar", 4)]
    assert [item.column for item in after.category_variants] == ["şehir"]
    assert len(demo_result.cleaned_frame) == len(generate_demo_data().frame) - 12
    assert "para_birimi" not in demo_result.cleaned_frame.columns
    assert demo_result.timeseries is not None


def test_return_target_is_learnable_without_leakage(demo_result: AutopilotResult) -> None:
    result = AutoMLEngine().run(demo_result.cleaned_frame, "iade_edildi")
    assert result.beats_baseline and not result.leakage_warnings
