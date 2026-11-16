from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.timeseries import (
    TimeSeriesEngine,
    TimeSeriesSummary,
    _active_segments,
    _pick_segment,
    date_columns,
    format_period,
    value_columns,
)

START = pd.Timestamp("2023-01-02")  # Pazartesi
WEEKDAY_EFFECT = np.array([25, 10, 5, 0, -5, -15, -20])  # Pazartesi … Pazar
SPIKE_DAY = 150


def _events(counts: np.ndarray, start: pd.Timestamp = START, amount: float | None = None) -> pd.DataFrame:
    """Her kayıt bir olay: gün başına `counts[i]` satır, saatleri gün içine yayılmış."""
    days = pd.date_range(start, periods=len(counts), freq="D")
    stamps = np.repeat(days.to_numpy(), counts)
    hours = np.concatenate([np.linspace(8, 20, count) for count in counts if count]) if counts.sum() else []
    frame = pd.DataFrame({"sipariş_tarihi": pd.to_datetime(stamps) + pd.to_timedelta(hours, unit="h")})
    frame["müşteri_id"] = [f"m{index}" for index in range(len(frame))]
    if amount is not None:
        frame["tutar"] = amount
    return frame


def _trend_weekly_counts(days: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    counts = 60 + 0.3 * t + WEEKDAY_EFFECT[t % 7] + rng.normal(0, 2, days)
    counts[SPIKE_DAY] *= 5
    return np.clip(np.round(counts), 0, None).astype(int)


@pytest.fixture(scope="module")
def daily_frame() -> pd.DataFrame:
    body = _trend_weekly_counts()
    lead_in = np.ones(20, dtype=int)                       # seyrek deneme dönemi
    tail = np.round(body[-3:] * 0.4).astype(int)            # dışa aktarım anında eksik kayıtlı son günler
    return _events(np.concatenate([lead_in, body, tail]), start=START - pd.Timedelta(days=20))


@pytest.fixture(scope="module")
def daily_summary(daily_frame: pd.DataFrame) -> TimeSeriesSummary:
    return TimeSeriesEngine().analyze(daily_frame).summary


def test_auto_frequency_picks_daily_with_weekly_seasonality(daily_summary: TimeSeriesSummary) -> None:
    assert daily_summary.frequency == "D" and daily_summary.seasonal_period == 7
    assert daily_summary.date_column == "sipariş_tarihi" and daily_summary.value_label == "Kayıt sayısı"
    assert daily_summary.seasonality_strength is not None and daily_summary.seasonality_strength >= 0.6
    assert "en yoğun gün Pazartesi" in daily_summary.seasonality_text
    assert "en sakin gün Pazar " in daily_summary.seasonality_text


def test_trend_is_measured(daily_summary: TimeSeriesSummary) -> None:
    assert daily_summary.level_change is not None and daily_summary.level_change > 0.5
    assert "arttı" in daily_summary.trend_text


def test_sparse_lead_in_and_censored_tail_are_excluded_and_reported(daily_summary: TimeSeriesSummary) -> None:
    assert daily_summary.start == START.date().isoformat()
    assert daily_summary.periods == 300
    assert "başta 20, sonda 0 gün" in daily_summary.notes[0]
    assert any(note.startswith("Son 3 gün") and "veri kesimi" in note for note in daily_summary.notes)


def test_spike_is_flagged_as_anomaly(daily_summary: TimeSeriesSummary) -> None:
    top = daily_summary.anomalies[0]
    assert top.date == (START + pd.Timedelta(days=SPIKE_DAY)).date().isoformat()
    assert top.direction == "yüksek" and top.observed > 3 * top.expected
    assert top.label == format_period(top.date, "D")


def test_backtest_against_seasonal_naive(daily_summary: TimeSeriesSummary) -> None:
    assert daily_summary.backtest_folds == 3 and daily_summary.horizon == 28
    assert "mevsimsel naif" in daily_summary.baseline_name
    assert daily_summary.beats_baseline and daily_summary.skill is not None and daily_summary.skill > 0.05
    assert daily_summary.model_mae < daily_summary.baseline_mae
    assert "geçti" in daily_summary.verdict


def test_forecast_has_ordered_interval(daily_summary: TimeSeriesSummary) -> None:
    forecast = daily_summary.forecast
    assert len(forecast) == 28
    dates = pd.to_datetime([point.date for point in forecast])
    assert dates[0] == pd.Timestamp(daily_summary.end) + pd.Timedelta(days=1)
    assert (np.diff(dates) == np.timedelta64(1, "D")).all()
    assert all(0 <= point.lower <= point.mean <= point.upper for point in forecast)
    assert daily_summary.forecast_total == pytest.approx(sum(point.mean for point in forecast), rel=1e-3)


def test_summary_round_trips_as_json(daily_summary: TimeSeriesSummary) -> None:
    assert TimeSeriesSummary.model_validate_json(daily_summary.model_dump_json()) == daily_summary


def test_figures_are_built(daily_frame: pd.DataFrame) -> None:
    result = TimeSeriesEngine().analyze(daily_frame)
    names = [trace.name for trace in result.forecast_figure.data]
    assert {"Gözlenen", "Eğilim", "Tahmin", "%80 tahmin aralığı", "Olağan dışı"} <= set(names)
    assert result.seasonal_figure is not None
    assert list(result.seasonal_figure.data[0].x)[0] == "Pazartesi"
    # İngilizce ay kısaltmaları yerine sayısal Türkçe tarih biçimi
    assert result.forecast_figure.layout.xaxis.tickformat == "%m.%Y"
    assert result.forecast_figure.layout.xaxis.hoverformat == "%d.%m.%Y"


def test_input_is_not_mutated(daily_frame: pd.DataFrame) -> None:
    snapshot = daily_frame.copy(deep=True)
    TimeSeriesEngine().analyze(daily_frame)
    pd.testing.assert_frame_equal(daily_frame, snapshot)


def test_short_monthly_series_is_analysed_without_seasonality(orders: pd.DataFrame) -> None:
    from core.executor import ActionExecutor

    frame = ActionExecutor().parse_datetime_columns(orders)
    summary = TimeSeriesEngine().analyze(frame).summary
    assert summary.frequency == "M" and summary.seasonal_period is None
    assert summary.date_column == "order_purchase_timestamp"
    assert "mevsimsellik incelenmedi" in summary.seasonality_text
    assert summary.baseline_name == "naif (son dönemin değeri)"
    assert summary.notes[0].startswith("Analiz Şubat 2017 – Temmuz 2018")


def test_value_column_is_summed() -> None:
    frame = _events(np.full(120, 10), amount=2.5)
    frame.loc[frame.index[:5], "tutar"] = np.nan
    summary = TimeSeriesEngine().analyze(frame, value_column="tutar").summary
    assert summary.value_label == "tutar toplamı" and summary.value_column == "tutar"
    assert any("boş olan 5 kayıt toplama katılmadı" in note for note in summary.notes)
    assert summary.forecast[0].mean == pytest.approx(25, abs=1)


def test_perfectly_periodic_series_does_not_claim_to_beat_naive() -> None:
    pattern = np.array([30, 28, 26, 24, 22, 12, 8])
    summary = TimeSeriesEngine().analyze(_events(np.tile(pattern, 30))).summary
    assert summary.baseline_mae == 0 and not summary.beats_baseline
    assert "geçemedi" in summary.verdict


def test_forced_weekly_frequency() -> None:
    summary = TimeSeriesEngine().analyze(_events(_trend_weekly_counts(days=210)), frequency="W").summary
    assert summary.frequency == "W" and summary.seasonal_period is None
    assert summary.anomalies == [] or summary.anomalies[0].label.endswith("haftası")


def test_sparse_daily_series_is_flagged_when_forced() -> None:
    rng = np.random.default_rng(4)
    summary = TimeSeriesEngine().analyze(_events(rng.poisson(0.6, 200)), frequency="D").summary
    assert any(note.startswith("Seri seyrek") for note in summary.notes)


def test_errors_are_turkish() -> None:
    with pytest.raises(ValueError, match="tarih sütunu bulunamadı"):
        TimeSeriesEngine().analyze(pd.DataFrame({"a": range(30)}))
    with pytest.raises(ValueError, match="en az 12 dönem"):
        TimeSeriesEngine().analyze(_events(np.full(5, 3)))
    with pytest.raises(ValueError, match="sayısal bir ölçü"):
        TimeSeriesEngine().analyze(_events(np.full(60, 3)), value_column="müşteri_id")


def test_column_helpers(daily_frame: pd.DataFrame) -> None:
    frame = daily_frame.assign(teslim=daily_frame["sipariş_tarihi"] + pd.Timedelta(days=3), adet=1.0, tutar=2.0)
    frame.loc[frame.index[::4], "teslim"] = pd.NaT
    assert date_columns(frame) == ["sipariş_tarihi", "teslim"]
    assert value_columns(frame.assign(adet=np.arange(len(frame)) % 5)) == ["adet"]


def test_active_segments_and_pick() -> None:
    active = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    assert _active_segments(active, min_dead_run=3) == [(0, 9), (12, 14)]   # 2 dönemlik boşluk bölmez
    assert _active_segments(active, min_dead_run=2) == [(0, 1), (3, 9), (12, 14)]
    assert _active_segments(active, min_dead_run=4) == [(0, 14)]
    assert _active_segments(np.zeros(4, dtype=bool), 2) == []
    assert _pick_segment([(0, 9), (12, 14)], min_length=2) == (12, 14)     # en güncel yeterli bölüm
    assert _pick_segment([(0, 9), (12, 14)], min_length=5) == (0, 9)       # güncel bölüm çok kısa
    assert _pick_segment([], min_length=5) == (0, 0)


def test_format_period() -> None:
    assert format_period("2017-11-24", "D") == "24.11.2017"
    assert format_period("2017-11-20", "W") == "20.11.2017 haftası"
    assert format_period("2017-11-01", "M") == "Kasım 2017"


def test_autopilot_runs_timeseries_and_drops_raw_area_chart(orders_result) -> None:
    assert orders_result.timeseries is not None and orders_result.timeseries_message == ""
    assert orders_result.timeseries.summary.date_column == "order_purchase_timestamp"
    assert "Zaman Serisi Alan Grafiği" not in [chart.title for chart in orders_result.charts]


def test_autopilot_without_dates_has_no_timeseries(dirty_result) -> None:
    assert dirty_result.timeseries is None and dirty_result.timeseries_message == ""


def test_autopilot_reports_why_timeseries_failed() -> None:
    from core.autopilot import AutopilotEngine

    frame = pd.DataFrame({
        "tarih": pd.date_range("2024-01-01", periods=40, freq="D").repeat(3)[:100],
        "değer": np.arange(100, dtype=float),
        "grup": ["a", "b"] * 50,
    })
    frame.loc[frame.index[20:], "tarih"] = pd.Timestamp("2024-01-08")
    result = AutopilotEngine().run_full_autopilot(frame)
    assert result.timeseries is None
    assert result.timeseries_message.startswith("Zaman serisi analizi yapılamadı: ")


# --- Gözden geçirmede bulunan kenar durumları ----------------------------------------------------


def test_mid_series_gap_keeps_recent_data() -> None:
    """100 dolu gün + 7 boş gün + 30 dolu gün: güncel 30 gün atılmamalı, boşluk 0 sayılmalı."""
    counts = np.concatenate([np.full(100, 50), np.zeros(7, dtype=int), np.full(30, 50)])
    summary = TimeSeriesEngine().analyze(_events(counts), frequency="D").summary
    last_day = START + pd.Timedelta(days=len(counts) - 1)
    assert summary.end == last_day.date().isoformat()
    assert summary.forecast[0].date == (last_day + pd.Timedelta(days=1)).date().isoformat()
    assert summary.periods == len(counts)
    assert any("Hiç kaydı olmayan 7 gün" in note for note in summary.notes)


def test_long_gap_starts_analysis_at_the_latest_regime() -> None:
    """Olist'teki gibi: kısa bir deneme dönemi, uzun sessizlik, sonra asıl dönem → asıl dönem analiz edilir."""
    counts = np.concatenate([np.full(20, 50), np.zeros(60, dtype=int), _trend_weekly_counts(days=200)])
    summary = TimeSeriesEngine().analyze(_events(counts)).summary
    assert summary.start == (START + pd.Timedelta(days=80)).date().isoformat()
    assert summary.end == (START + pd.Timedelta(days=len(counts) - 1)).date().isoformat()
    assert "başta 80" in summary.notes[0]


def test_regular_weekend_dip_is_not_treated_as_censoring() -> None:
    """Hafta içi 100, hafta sonu 20: seri pazar günü bittiğinde son cumartesi ve pazar silinmemeli."""
    week = np.array([100, 100, 100, 100, 100, 20, 20])
    counts = np.tile(week, 12)  # START pazartesi → seri pazar günü biter
    summary = TimeSeriesEngine().analyze(_events(counts), frequency="D").summary
    last_day = START + pd.Timedelta(days=len(counts) - 1)
    assert last_day.dayofweek == 6
    assert summary.end == last_day.date().isoformat()
    assert not any("veri kesimi" in note for note in summary.notes)
    assert pd.Timestamp(summary.forecast[0].date).dayofweek == 0


def test_genuine_tail_drop_is_still_censored_against_same_weekday() -> None:
    week = np.array([100, 100, 100, 100, 100, 40, 40])
    counts = np.tile(week, 12)
    counts[-2:] = [12, 11]  # seyrek eşiğinin (10) üstünde ama cumartesi/pazarın olağan seviyesinin (40) yarısının altında
    summary = TimeSeriesEngine().analyze(_events(counts), frequency="D").summary
    assert any(note.startswith("Son 2 gün") for note in summary.notes)


def test_complete_february_is_kept() -> None:
    """Son ayın kapsaması kendi uzunluğuyla ölçülür: 28 Şubat akşamına kadar dolu Şubat eksik sayılmaz."""
    from core.timeseries import _SPECS

    days = pd.date_range("2023-03-01", "2024-02-29", freq="D")  # 2024 artık yıl: Şubat 29 gün
    frame = pd.DataFrame({"tarih": np.repeat(days.to_numpy(), 3) + np.tile(pd.to_timedelta([9, 14, 20], unit="h"), len(days))})
    built = TimeSeriesEngine._build(frame.rename(columns={"tarih": "date"}), _SPECS["M"])
    assert built.series.index[0] == pd.Timestamp("2023-03-01")
    assert built.series.index[-1] == pd.Timestamp("2024-02-01")

    # Aralıkta (31 gün) başlayıp 28 Şubat akşamı biten seri: ilk ayın uzunluğuyla ölçülürse Şubat %90'ın altında kalırdı.
    days = pd.date_range("2022-12-01", "2023-02-28", freq="D")
    frame = pd.DataFrame({"date": days + pd.Timedelta(hours=20)})
    built = TimeSeriesEngine._build(frame, _SPECS["M"])
    assert list(built.series.index) == [pd.Timestamp("2022-12-01"), pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01")]


def test_partial_last_month_is_still_dropped() -> None:
    from core.timeseries import _SPECS

    days = pd.date_range("2023-01-01", "2023-03-10", freq="D")
    built = TimeSeriesEngine._build(pd.DataFrame({"date": days + pd.Timedelta(hours=12)}), _SPECS["M"])
    assert built.series.index[-1] == pd.Timestamp("2023-02-01") and built.trailing_removed == 1
