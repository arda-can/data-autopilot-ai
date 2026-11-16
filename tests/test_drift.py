from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from core.drift import (
    DriftEngine,
    DriftReport,
    prepare_for_comparison,
    split_by_date,
    split_by_years,
)


def _period(seed: int, rows: int = 600, price_shift: float = 0.0, city_weights=(0.5, 0.3, 0.2),
            missing_note: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "sipariş_id": [f"s{seed}-{index}" for index in range(rows)],
        "fiyat": rng.normal(100 + price_shift, 15, rows),
        "adet": rng.integers(1, 5, rows),
        "şehir": rng.choice(["İstanbul", "Ankara", "İzmir"], rows, p=city_weights),
        "kargo": rng.normal(20, 3, rows),
        "not": [f"serbest metin {seed}-{index}" for index in range(rows)],
    })
    frame.loc[frame.index[: int(rows * missing_note)], "not"] = None
    return frame


@pytest.fixture(scope="module")
def shifted() -> DriftReport:
    reference = _period(1)
    current = _period(2, price_shift=25, city_weights=(0.2, 0.3, 0.5), missing_note=0.3)
    return DriftEngine().compare(reference, current, "2017", "2018").report


def _by_column(report: DriftReport) -> dict:
    return {item.column: item for item in report.columns}


def test_stable_columns_are_green(shifted: DriftReport) -> None:
    columns = _by_column(shifted)
    assert columns["kargo"].status == "yeşil" and columns["kargo"].psi < 0.1
    assert columns["adet"].status == "yeşil" and columns["adet"].kind == "kategorik"  # az değerli sayısal


def test_numeric_shift_is_red_and_explained(shifted: DriftReport) -> None:
    price = _by_column(shifted)["fiyat"]
    assert price.status == "kırmızı" and price.psi > 0.25 and price.test == "Kolmogorov-Smirnov"
    assert price.p_value is not None and price.p_value < 0.001
    assert price.explanation.startswith("Medyan ") and "çeyrekler arası" in price.explanation


def test_category_share_shift_is_detected(shifted: DriftReport) -> None:
    city = _by_column(shifted)["şehir"]
    assert city.status == "kırmızı" and city.test == "Ki-kare"
    assert "“İstanbul” payı" in city.explanation or "“İzmir” payı" in city.explanation


def test_free_text_is_compared_only_by_missing_ratio(shifted: DriftReport) -> None:
    note = _by_column(shifted)["not"]
    assert note.test == "Yalnızca eksik oranı" and note.psi == 0
    assert "neredeyse her satırı benzersiz" in note.explanation
    assert note.status == "kırmızı" and note.missing_current == pytest.approx(0.3)


def test_identifiers_are_skipped_and_report_is_sorted(shifted: DriftReport) -> None:
    assert any(item.startswith("sipariş_id:") for item in shifted.skipped)
    severities = [{"kırmızı": 2, "sarı": 1, "yeşil": 0}[item.status] for item in shifted.columns]
    assert severities == sorted(severities, reverse=True)
    assert shifted.counts == {"kırmızı": 3, "sarı": 0, "yeşil": 2}
    assert shifted.summary.startswith("“2017” ile “2018” arasında 5 sütun karşılaştırıldı: 3 kırmızı")


def test_long_free_text_with_repeats_is_missing_only() -> None:
    rng = np.random.default_rng(8)
    words = [f"yorum {index}" for index in range(300)]
    reference = pd.DataFrame({"yorum": rng.choice(words, 900)})
    current = pd.DataFrame({"yorum": rng.choice(words, 900)})
    current.loc[current.index[:200], "yorum"] = None
    item = DriftEngine().compare(reference, current).report.columns[0]
    assert item.kind == "metin" and item.test == "Yalnızca eksik oranı" and item.status == "kırmızı"


def test_identical_periods_are_all_green() -> None:
    frame = _period(3)
    report = DriftEngine().compare(frame, frame.copy()).report
    assert report.counts["kırmızı"] == report.counts["sarı"] == 0
    assert "belirgin bir değişim yok" in report.summary


def test_schema_changes_are_reported() -> None:
    reference = _period(4)
    current = _period(5).drop(columns=["kargo"]).assign(indirim=1.0, adet=lambda frame: frame["adet"].astype(str))
    report = DriftEngine().compare(reference, current).report
    changes = {(item.column, item.change) for item in report.schema_changes}
    assert changes == {("indirim", "eklendi"), ("kargo", "kaldırıldı"), ("adet", "tip değişti")}
    assert "Şema farkı: 3 sütun" in report.summary


def test_too_few_rows_raise_turkish_error() -> None:
    with pytest.raises(ValueError, match="en az 30 satır"):
        DriftEngine().compare(_period(1), _period(2, rows=10), "A", "B")


def test_input_is_not_mutated_and_report_round_trips() -> None:
    reference, current = _period(6), _period(7, price_shift=10)
    snapshots = reference.copy(deep=True), current.copy(deep=True)
    result = DriftEngine().compare(reference, current)
    pd.testing.assert_frame_equal(reference, snapshots[0])
    pd.testing.assert_frame_equal(current, snapshots[1])
    assert DriftReport.model_validate_json(result.report.model_dump_json()) == result.report


def test_figures(shifted: DriftReport, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.chart_export import chart_png

    result = DriftEngine().compare(_period(1), _period(2, price_shift=25), "A", "B")
    assert result.figure is not None and set(result.detail_figures) == {"fiyat"}
    monkeypatch.setattr(go.Figure, "to_image", lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError()))
    assert chart_png(result.figure).startswith(b"\x89PNG")


def test_split_by_date_and_years() -> None:
    frame = pd.DataFrame({
        "tarih": pd.date_range("2017-01-01", "2018-12-31", freq="D"),
        "değer": np.arange(730, dtype=float),
    })
    reference, current, first, second = split_by_date(frame, "tarih", "2018-01-01")
    assert len(reference) == 365 and len(current) == 365
    assert first == "01.01.2017 – 31.12.2017" and second == "01.01.2018 – 31.12.2018"
    reference, current, first, second = split_by_date(frame, "tarih")
    assert abs(len(reference) - len(current)) <= 1
    reference, current, first, second = split_by_years(frame, "tarih", 2017, 2018)
    assert (first, second) == ("2017", "2018") and len(reference) == 365
    with pytest.raises(ValueError, match="tarih sütunu değil"):
        split_by_date(frame, "değer")


def test_prepare_for_comparison_parses_like_the_autopilot() -> None:
    raw = pd.DataFrame({"tutar": ["1.250,50 TL", "99,90 TL"] * 20, "tarih": ["2018-01-02", "2018-02-03"] * 20})
    prepared = prepare_for_comparison(raw)
    assert pd.api.types.is_numeric_dtype(prepared["tutar"])
    assert pd.api.types.is_datetime64_any_dtype(prepared["tarih"])
