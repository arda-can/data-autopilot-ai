"""A/B test analizi: istatistikler bağımsız uygulamalarla (statsmodels) çapraz doğrulanır."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from core.autopilot import AutopilotEngine, AutopilotResult
from core.demo_data import DEMO_FILENAME, write_demo_data
from core.experiment import ExperimentAnalyzer, ExperimentSpec, detect_experiment
from core.ingestion import read_local_csv
from core.report import build_report, render_html, render_pdf


def _binary(control: tuple[int, int], treatment: tuple[int, int]) -> pd.DataFrame:
    """(başarı, toplam) çiftlerinden satır düzeyinde deney verisi."""
    rows = []
    for group, (success, total) in (("A", control), ("B", treatment)):
        rows += [{"grup": group, "dönüşüm": "Evet"}] * success + [{"grup": group, "dönüşüm": "Hayır"}] * (total - success)
    return pd.DataFrame(rows)


def _spec(**changes) -> ExperimentSpec:
    base = dict(group_column="grup", control="A", treatment="B", metric_column="dönüşüm", metric_kind="oran",
                success_value="Evet")
    return ExperimentSpec(**(base | changes))


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> AutopilotResult:
    path = write_demo_data(tmp_path_factory.mktemp("deney") / DEMO_FILENAME)
    return AutopilotEngine().run_full_autopilot(read_local_csv(path))


def test_proportion_test_matches_statsmodels() -> None:
    report = ExperimentAnalyzer().analyze(_binary((120, 1000), (140, 1000)), _spec()).report
    _, expected_p = proportions_ztest([140, 120], [1000, 1000])
    assert report.test_name == "İki oranlı z-testi"
    assert report.p_value == pytest.approx(expected_p, rel=1e-9)
    assert report.difference == pytest.approx(0.02)
    se = np.sqrt(0.12 * 0.88 / 1000 + 0.14 * 0.86 / 1000)
    assert (report.ci_low, report.ci_high) == pytest.approx((0.02 - 1.959964 * se, 0.02 + 1.959964 * se), rel=1e-5)
    assert report.relative_lift == pytest.approx(1 / 6)
    assert report.significant is False and "anlamlı fark yok" in report.verdict
    assert "grup başına" in report.verdict  # anlamsız sonuçta gereken örneklem söylenir
    significant = ExperimentAnalyzer().analyze(_binary((120, 1000), (150, 1000)), _spec()).report
    assert significant.p_value == pytest.approx(proportions_ztest([150, 120], [1000, 1000])[1], rel=1e-9)
    assert significant.significant and "anlamlı biçimde yüksek" in significant.verdict


def test_welch_interval_matches_statsmodels() -> None:
    rng = np.random.default_rng(3)
    control, treatment = rng.normal(100, 20, 300), rng.normal(106, 30, 250)
    frame = pd.DataFrame({"grup": ["A"] * 300 + ["B"] * 250, "tutar": np.concatenate([control, treatment])})
    report = ExperimentAnalyzer().analyze(frame, _spec(metric_column="tutar", metric_kind="ortalama",
                                                       success_value=None)).report
    compare = CompareMeans(DescrStatsW(treatment), DescrStatsW(control))
    low, high = compare.tconfint_diff(usevar="unequal")
    _, p_value, _ = compare.ttest_ind(usevar="unequal")
    assert (report.ci_low, report.ci_high) == pytest.approx((low, high), rel=1e-9)
    assert report.p_value == pytest.approx(p_value, rel=1e-9) and report.significant


def test_sample_size_matches_normal_power_for_equal_variances() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(50, 10, 4000)
    frame = pd.DataFrame({"grup": ["A", "B"] * 2000, "tutar": values})
    report = ExperimentAnalyzer().analyze(frame, _spec(metric_column="tutar", metric_kind="ortalama",
                                                       success_value=None, mde_relative=0.05)).report
    pooled_sd = np.sqrt((report.control.sd ** 2 + report.treatment.sd ** 2) / 2)
    effect = 0.05 * report.control.mean / pooled_sd
    expected = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1)
    assert report.power.required_per_group == int(np.ceil(expected))


def test_srm_alarm_blocks_interpretation() -> None:
    report = ExperimentAnalyzer().analyze(_binary((500, 5000), (700, 4500)), _spec()).report
    assert report.srm.alarm and report.srm.p_value < 0.001
    assert report.verdict.startswith("Örneklem oranı uyumsuz (SRM)")
    balanced = ExperimentAnalyzer().analyze(_binary((500, 5000), (560, 5000)), _spec()).report
    assert not balanced.srm.alarm and balanced.srm.p_value == pytest.approx(1.0)


def test_expected_split_is_respected() -> None:
    report = ExperimentAnalyzer().analyze(_binary((90, 900), (10, 100)), _spec(expected_control_share=0.9)).report
    assert not report.srm.alarm and report.srm.observed_control_share == pytest.approx(0.9)


def test_unit_level_analysis_and_mixed_units() -> None:
    frame = pd.DataFrame({
        "grup": ["A", "A", "A", "B", "B", "B", "A", "B", "A", "B"],
        "müşteri": ["m1", "m1", "m2", "m3", "m3", "m4", "m5", "m5", "m6", "m7"],
        "dönüşüm": ["Evet", "Hayır", "Hayır", "Evet", "Evet", "Hayır", "Evet", "Evet", "Hayır", "Evet"],
    })
    report = ExperimentAnalyzer().analyze(frame, _spec(unit_column="müşteri")).report
    assert report.excluded_units == 1 and "her iki grupta" in report.notes[0]
    assert (report.control.units, report.treatment.units) == (3, 3)  # m1, m2, m6 / m3, m4, m7
    assert report.control.mean == pytest.approx((0.5 + 0 + 0) / 3)
    assert report.test_name.startswith("Welch t-testi (müşteri düzeyinde)")


@pytest.mark.parametrize(("changes", "message"), [
    ({"treatment": "A"}, "farklı olmalı"),
    ({"success_value": None}, "başarı değeri"),
    ({"metric_column": "grup"}, "grup ya da birim"),
    ({"alpha": 0.7}, "less than"),
])
def test_invalid_specs_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _spec(**changes)


def test_missing_group_and_non_numeric_mean_raise_turkish_errors() -> None:
    frame = _binary((5, 10), (6, 10))
    with pytest.raises(ValueError, match="“C” grubunda kayıt yok"):
        ExperimentAnalyzer().analyze(frame, _spec(treatment="C"))
    with pytest.raises(ValueError, match="sayısal değil"):
        ExperimentAnalyzer().analyze(frame, _spec(metric_kind="ortalama", success_value=None))


def test_demo_experiment_is_detected_and_significant(demo_result: AutopilotResult) -> None:
    spec = detect_experiment(demo_result.cleaned_frame)
    assert (spec.group_column, spec.control, spec.treatment) == ("deney_grubu", "A", "B")
    assert (spec.metric_column, spec.success_value, spec.unit_column) == ("ek_ürün_eklendi", "Evet", "müşteri_no")
    report = ExperimentAnalyzer().analyze(demo_result.cleaned_frame, spec).report
    assert report.significant and not report.srm.alarm and report.relative_lift > 0.2
    assert report.analysis_unit == "müşteri_no" and report.control.units < report.control.rows
    assert detect_experiment(demo_result.cleaned_frame.drop(columns=["deney_grubu"])) is None


def test_report_section(demo_result: AutopilotResult) -> None:
    test = ExperimentAnalyzer().analyze(demo_result.cleaned_frame, detect_experiment(demo_result.cleaned_frame)).report
    report = build_report(demo_result, "demo.csv", experiment=test)
    page = render_html(report, embed_plotly=False)
    assert "<h2>A/B Testi</h2>" in page and "Yakalanabilir en küçük fark" in page
    assert '<p class="verdict good">' in page.split("<h2>A/B Testi</h2>")[1].split("</section>")[0]
    assert any(item.startswith("A/B testi (") for item in report.headline)
    assert render_pdf(report).startswith(b"%PDF")
