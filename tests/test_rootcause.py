"""“Neden değişti?” ayrıştırması: katkılar tam, karma etkisi ayrı, sıralama büyük gruplara kanmıyor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.autopilot import AutopilotEngine, AutopilotResult
from core.demo_data import DEMO_FILENAME, write_demo_data
from core.drift import split_by_ranges, split_by_years
from core.ingestion import read_local_csv
from core.report import build_report, render_html, render_pdf
from core.rootcause import MetricSpec, RootCauseAnalyzer, candidate_dimensions

COUNT = MetricSpec(kind="sayım")


def _rows(spec: dict[tuple, int], **extra) -> pd.DataFrame:
    """{(şehir, kanal): kayıt sayısı} → satırlar."""
    rows = [{"şehir": city, "kanal": channel, **extra} for (city, channel), count in spec.items() for _ in range(count)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def demo_result(tmp_path_factory: pytest.TempPathFactory) -> AutopilotResult:
    path = write_demo_data(tmp_path_factory.mktemp("kok") / DEMO_FILENAME)
    return AutopilotEngine().run_full_autopilot(read_local_csv(path))


def test_change_is_split_exactly_and_concentrated_dimension_wins() -> None:
    reference = _rows({("İstanbul", "Web"): 400, ("İstanbul", "Mağaza"): 400, ("Ankara", "Web"): 100,
                       ("Ankara", "Mağaza"): 100})
    current = _rows({("İstanbul", "Web"): 400, ("İstanbul", "Mağaza"): 400, ("Ankara", "Web"): 100,
                     ("Ankara", "Mağaza"): 20})
    report = RootCauseAnalyzer().analyze(reference, current, "Önce", "Sonra", COUNT).report
    assert report.change == -80
    for dimension in report.dimensions:
        assert sum(item.contribution for item in dimension.groups) == pytest.approx(report.change)
    city = next(item for item in report.dimensions if item.dimension == "şehir")
    ankara = next(item for item in city.groups if item.group == "Ankara")
    assert ankara.share_of_change == pytest.approx(1.0) and ankara.expected == pytest.approx(-80 * 0.2)
    assert report.dimensions[0].dimension == "şehir"  # İstanbul büyük ama değişmedi
    assert "(%-8,0)" in report.headline and "en büyük katkı “Ankara”" in report.headline


def test_uniform_change_has_no_surprise() -> None:
    reference = _rows({("İstanbul", "Web"): 600, ("Ankara", "Web"): 200, ("İzmir", "Mağaza"): 200})
    current = _rows({("İstanbul", "Web"): 660, ("Ankara", "Web"): 220, ("İzmir", "Mağaza"): 220})
    report = RootCauseAnalyzer().analyze(reference, current, "Önce", "Sonra", COUNT).report
    assert all(item.score == pytest.approx(0.0) for item in report.dimensions)


def test_mean_change_from_mix_only_is_attributed_to_mix() -> None:
    """Grup ortalamaları sabit; yalnızca yüksek puanlı grubun payı azalınca genel ortalama düşer."""
    reference = pd.concat([_rows({("A", "x"): 50}, puan=5.0), _rows({("B", "x"): 50}, puan=3.0)])
    current = pd.concat([_rows({("A", "x"): 20}, puan=5.0), _rows({("B", "x"): 80}, puan=3.0)])
    report = RootCauseAnalyzer().analyze(
        reference, current, "Önce", "Sonra", MetricSpec(kind="ortalama", column="puan"), dimensions=["şehir"],
    ).report
    assert report.change == pytest.approx(3.4 - 4.0)
    groups = report.dimensions[0].groups
    assert sum(item.contribution for item in groups) == pytest.approx(report.change)
    assert all(item.rate_effect == pytest.approx(0.0) for item in groups)
    assert "payının değişmesinden" in report.dimensions[0].summary


def test_rate_metric_and_group_present_in_one_period_stay_exact() -> None:
    reference = pd.concat([_rows({("A", "x"): 100}, iade="Evet"), _rows({("A", "x"): 300}, iade="Hayır")])
    current = pd.concat([_rows({("A", "x"): 100}, iade="Evet"), _rows({("A", "x"): 100}, iade="Hayır"),
                         _rows({("Yeni", "x"): 40}, iade="Evet")])
    report = RootCauseAnalyzer().analyze(
        reference, current, "Önce", "Sonra", MetricSpec(kind="oran", column="iade", value="Evet"),
        dimensions=["şehir"],
    ).report
    assert report.change == pytest.approx(140 / 240 - 0.25)
    groups = {item.group: item for item in report.dimensions[0].groups}
    assert sum(item.contribution for item in groups.values()) == pytest.approx(report.change)
    assert groups["Yeni"].only_in == "güncel" and groups["Yeni"].rate_effect == pytest.approx(0.0)


def test_dimensions_exclude_identifiers_pii_metric_and_coverage_changes() -> None:
    frame = pd.DataFrame({
        "müşteri_no": [f"M{i}" for i in range(60)], "eposta": [f"k{i}@ornek.com" for i in range(60)],
        "şehir": ["A", "B"] * 30, "serbest": [f"metin {i}" for i in range(60)], "iade": ["Evet", "Hayır"] * 30,
        "yeni_alan": [None] * 30 + ["x", "y"] * 15,
    })
    dimensions, skipped = candidate_dimensions(frame, exclude=("iade",))
    assert dimensions == ["şehir", "yeni_alan"]
    assert any("müşteri_no" in note for note in skipped) and any("eposta" in note for note in skipped)
    report = RootCauseAnalyzer().analyze(frame.iloc[:30], frame.iloc[30:], "Önce", "Sonra", COUNT).report
    assert [item.dimension for item in report.dimensions] == ["şehir", "iade"]
    assert any("“yeni_alan” dönemler arasında doluluğu 100 puan" in note for note in report.skipped)


@pytest.mark.parametrize(("metric", "message"), [
    (MetricSpec(kind="toplam", column="şehir"), "sayısal değil"),
    (MetricSpec(kind="ortalama", column="yok"), "veride yok"),
    (MetricSpec(kind="oran", column="şehir"), "bir değer seçilmeli"),
])
def test_invalid_metrics_raise_turkish_errors(metric: MetricSpec, message: str) -> None:
    frame = _rows({("A", "x"): 5})
    with pytest.raises(ValueError, match=message):
        RootCauseAnalyzer().analyze(frame, frame, "Önce", "Sonra", metric)


def test_empty_period_is_refused() -> None:
    frame = _rows({("A", "x"): 5})
    with pytest.raises(ValueError, match="hiç kayıt yok"):
        RootCauseAnalyzer().analyze(frame, frame.iloc[0:0], "Önce", "Sonra", COUNT)


def test_split_by_ranges_includes_end_day() -> None:
    frame = pd.DataFrame({"tarih": pd.to_datetime(["2025-03-31 23:00", "2025-04-01 00:30", "2025-06-30 18:00"])})
    reference, current, ref_label, cur_label = split_by_ranges(
        frame, "tarih", ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    )
    assert (len(reference), len(current)) == (1, 2)
    assert (ref_label, cur_label) == ("01.01.2025 – 31.03.2025", "01.04.2025 – 30.06.2025")
    with pytest.raises(ValueError, match="başlangıcı bitişinden sonra"):
        split_by_ranges(frame, "tarih", ("2025-02-01", "2025-01-01"), ("2025-04-01", "2025-06-30"))


def test_demo_rating_drop_is_traced_to_ankara(demo_result: AutopilotResult) -> None:
    parts = split_by_ranges(demo_result.cleaned_frame, "sipariş_tarihi", ("2025-07-01", "2025-09-30"),
                            ("2025-10-01", "2025-12-31"))
    report = RootCauseAnalyzer().analyze(*parts, MetricSpec(kind="ortalama", column="değerlendirme_puanı")).report
    assert report.change < 0 and report.dimensions[0].dimension == "şehir"
    top = report.dimensions[0].groups[0]
    assert top.group == "Ankara" and top.share_of_change > 0.8 and top.rate_effect < 0
    assert top.expected / report.change < 0.25


def test_demo_revenue_growth_points_to_mobile_not_new_experiment_field(demo_result: AutopilotResult) -> None:
    parts = split_by_years(demo_result.cleaned_frame, "sipariş_tarihi", 2024, 2025)
    report = RootCauseAnalyzer().analyze(*parts, MetricSpec(kind="toplam", column="tutar")).report
    assert report.dimensions[0].dimension == "kanal" and report.dimensions[0].groups[0].group == "Mobil Uygulama"
    assert "deney_grubu" not in [item.dimension for item in report.dimensions]


def test_report_section(demo_result: AutopilotResult) -> None:
    parts = split_by_ranges(demo_result.cleaned_frame, "sipariş_tarihi", ("2025-07-01", "2025-09-30"),
                            ("2025-10-01", "2025-12-31"))
    cause = RootCauseAnalyzer().analyze(*parts, MetricSpec(kind="oran", column="iade_edildi", value="Evet")).report
    report = build_report(demo_result, "demo.csv", root_cause=cause)
    assert cause.headline in report.headline
    page = render_html(report, embed_plotly=False)
    assert "<h2>Neden Değişti?</h2>" in page and "“şehir” grupları" in page
    assert np.isfinite(cause.change) and render_pdf(report).startswith(b"%PDF")
