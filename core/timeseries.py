"""Zaman serisi otopilotu: tarih sütunundan trend, mevsimsellik, anomali ve güven aralıklı tahmin.

Akış:
1. Kayıtlar seçilen frekansta (günlük/haftalık/aylık) sayılır veya bir ölçü sütunu toplanır.
2. Tam kapsanmayan ve seyrek kenar dönemler analiz dışı bırakılır (ör. veri setinin başındaki
   deneme ayları); bu, okura not olarak bildirilir.
3. STL ile trend ve mevsimsellik ayrıştırılır; artıklardan olağan dışı dönemler bulunur.
4. AICc ile seçilen ETS modeli, naif (mevsimsel) tahmine karşı birden çok pencerede geriye dönük test edilir.
5. Tüm seriyle yeniden kurulan model, %80 tahmin aralığıyla ileriye tahmin üretir.

Modül yalnızca toplulaştırılmış dönem değerleriyle çalışır; ham satır döndürmez.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from core.columns import is_geo_coordinate_name, is_identifier_column
from core.formatting import decimal_tr, integer_tr, localize_figure

logger = logging.getLogger(__name__)

FrequencyCode = Literal["D", "W", "M"]

MIN_POINTS = 12
SPARSE_RATIO = 0.10
MAX_ZERO_SHARE = 0.20
EDGE_COVERAGE = 0.90
CENSOR_RATIO = 0.50
ANOMALY_Z = 4.0
MAX_ANOMALIES = 5
BACKTEST_FOLDS = 3
SKILL_THRESHOLD = 0.05
INTERVAL = 0.80
_MIN_TRAIN_NON_SEASONAL = 10

MONTHS_TR = ("Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
             "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık")
WEEKDAYS_TR = ("Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar")


@dataclass(frozen=True)
class _Spec:
    code: FrequencyCode
    pandas_freq: str
    season: int
    horizon: int
    level_window: int
    label: str        # "günlük"
    unit: str         # "gün"
    genitive: str     # "günün"
    locative: str     # "günde"
    min_dead_run: int
    max_censored: int  # sondan en fazla kaç dönem veri kesimi sayılabilir
    reference_window: int
    cycle: str        # "Haftalık"
    slot: str         # profil biriminin adı: "gün" / "ay"
    baseline: str


_SPECS: dict[str, _Spec] = {
    "M": _Spec("M", "MS", 12, 3, 3, "aylık", "ay", "ayın", "ayda", 2, 1, 6, "Yıllık", "ay",
               "mevsimsel naif (geçen yılın aynı ayı)"),
    "W": _Spec("W", "W-MON", 52, 8, 8, "haftalık", "hafta", "haftanın", "haftada", 3, 2, 8, "Yıllık", "ay",
               "mevsimsel naif (geçen yılın aynı haftası)"),
    "D": _Spec("D", "D", 7, 28, 28, "günlük", "gün", "günün", "günde", 7, 7, 28, "Haftalık", "gün",
               "mevsimsel naif (geçen haftanın aynı günü)"),
}
_AUTO_ORDER: tuple[FrequencyCode, ...] = ("M", "W", "D")
_NAIVE = "naif (son dönemin değeri)"


# --- Şemalar ---------------------------------------------------------------------------------


class TimeSeriesAnomaly(BaseModel):
    date: str
    label: str
    observed: float
    expected: float
    z: float
    direction: Literal["yüksek", "düşük"]


class ForecastPoint(BaseModel):
    date: str
    mean: float
    lower: float
    upper: float


class TimeSeriesSummary(BaseModel):
    """JSON'a çevrilebilir özet: rapor, arayüz ve ileride API aynı nesneyi kullanır."""

    date_column: str
    value_column: str | None = None
    value_label: str
    frequency: FrequencyCode
    frequency_label: str
    period_unit: str
    start: str
    end: str
    periods: int
    seasonal_period: int | None = None
    seasonality_strength: float | None = None
    level_change: float | None = None
    model_name: str
    baseline_name: str
    horizon: int
    backtest_folds: int
    model_mae: float | None = None
    baseline_mae: float | None = None
    skill: float | None = None
    beats_baseline: bool = False
    forecast_total: float
    trend_text: str
    seasonality_text: str
    verdict: str
    forecast_text: str
    notes: list[str] = Field(default_factory=list)
    anomalies: list[TimeSeriesAnomaly] = Field(default_factory=list)
    forecast: list[ForecastPoint] = Field(default_factory=list)


@dataclass
class TimeSeriesResult:
    summary: TimeSeriesSummary
    forecast_figure: Any
    seasonal_figure: Any | None = None


@dataclass
class _Built:
    series: pd.Series
    leading_removed: int
    trailing_removed: int
    censored: int
    gaps: int
    zero_share: float


# --- Yardımcılar -----------------------------------------------------------------------------


def format_period(value: pd.Timestamp | str, frequency: str) -> str:
    """Dönemi Türkçe yazar: günlük `24.11.2017`, haftalık `20.11.2017 haftası`, aylık `Kasım 2017`."""
    stamp = pd.Timestamp(value)
    if frequency == "M":
        return f"{MONTHS_TR[stamp.month - 1]} {stamp.year}"
    text = stamp.strftime("%d.%m.%Y")
    return f"{text} haftası" if frequency == "W" else text


def _signed(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{decimal_tr(abs(value))}"


def date_columns(frame: pd.DataFrame) -> list[str]:
    """Tarih sütunları, en uygun olan önce: en az eksik, eşitlikte en erken olay (sipariş < teslimat)."""
    columns = [
        str(column) for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column]) and frame[column].notna().sum() >= MIN_POINTS
    ]

    def key(column: str) -> tuple[float, pd.Timestamp]:
        series = frame[column]
        return float(series.isna().mean()), series.dropna().median()

    return sorted(columns, key=key)


def value_columns(frame: pd.DataFrame) -> list[str]:
    """Dönem toplamı alınabilecek sayısal ölçüler (kimlik, koordinat ve sabit sütunlar hariç)."""
    return [
        str(column) for column in frame.select_dtypes(include="number").columns
        if not is_identifier_column(frame[column], column)
        and not is_geo_coordinate_name(column)
        and frame[column].nunique(dropna=True) > 1
    ]


def _period_start(dates: pd.Series, spec: _Spec) -> pd.Series:
    if spec.code == "D":
        return dates.dt.floor("D")
    period = "W-SUN" if spec.code == "W" else "M"
    return dates.dt.to_period(period).dt.start_time


def _longest_active_run(active: np.ndarray, min_dead_run: int) -> tuple[int, int]:
    """Etkin dönemlerin, `min_dead_run` uzunlukta ölü aralıklarla bölünmemiş en uzun bölümü [başlangıç, bitiş)."""
    best = (0, 0)
    start: int | None = None
    last_active = -1
    dead = 0
    for index, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = index
            last_active = index
            dead = 0
            continue
        if start is None:
            continue
        dead += 1
        if dead >= min_dead_run:
            if last_active + 1 - start > best[1] - best[0]:
                best = (start, last_active + 1)
            start, dead = None, 0
    if start is not None and last_active + 1 - start > best[1] - best[0]:
        best = (start, last_active + 1)
    return best


# --- Motor -----------------------------------------------------------------------------------


class TimeSeriesEngine:
    """Deterministik zaman serisi analizi; LLM kullanmaz."""

    def analyze(
        self,
        frame: pd.DataFrame,
        date_column: str | None = None,
        value_column: str | None = None,
        frequency: FrequencyCode | Literal["auto"] = "auto",
    ) -> TimeSeriesResult:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("Zaman serisi analizi için boş olmayan bir pandas DataFrame gereklidir.")
        if date_column is None:
            candidates = date_columns(frame)
            if not candidates:
                raise ValueError("Zaman serisi analizi için tarih sütunu bulunamadı.")
            date_column = candidates[0]
        if date_column not in frame.columns or not pd.api.types.is_datetime64_any_dtype(frame[date_column]):
            raise ValueError(f"'{date_column}' bir tarih sütunu değil.")
        if value_column is not None and (
            value_column not in frame.columns or not pd.api.types.is_numeric_dtype(frame[value_column])
        ):
            raise ValueError(f"'{value_column}' sayısal bir ölçü sütunu değil.")
        if frequency != "auto" and frequency not in _SPECS:
            raise ValueError(f"Desteklenmeyen frekans: {frequency}")

        events = pd.DataFrame({"date": frame[date_column]})
        if value_column is not None:
            events["value"] = frame[value_column]
        events = events.dropna(subset=["date"])
        notes: list[str] = []
        if value_column is not None:
            missing = int(events["value"].isna().sum())
            if missing:
                notes.append(f"'{value_column}' değeri boş olan {integer_tr(missing)} kayıt toplama katılmadı.")

        codes = _AUTO_ORDER if frequency == "auto" else (frequency,)
        built = {code: self._build(events, _SPECS[code]) for code in codes}
        chosen, seasonal = self._choose(built)
        spec, data = _SPECS[chosen], built[chosen]
        if data.zero_share > MAX_ZERO_SHARE:
            notes.append(
                f"Seri seyrek: {spec.unit} dönemlerinin %{decimal_tr(data.zero_share * 100)} kadarında hiç kayıt yok; "
                "daha kaba bir frekans daha güvenilir sonuç verir."
            )
        value_label = "Kayıt sayısı" if value_column is None else f"{value_column} toplamı"
        return self._analyze_series(data, spec, seasonal, date_column, value_column, value_label, notes)

    # -- Seri kurulumu --

    @staticmethod
    def _build(events: pd.DataFrame, spec: _Spec) -> _Built:
        keys = _period_start(events["date"], spec)
        grouped = (
            keys.value_counts() if "value" not in events
            else events["value"].groupby(keys).sum(min_count=1).dropna()
        ).sort_index()
        if grouped.empty:
            raise ValueError("Zaman serisi için kullanılabilir tarih bulunamadı.")
        index = pd.date_range(grouped.index.min(), grouped.index.max(), freq=spec.pandas_freq)
        series = grouped.reindex(index, fill_value=0).astype(float)
        series.index.name = None

        start, end = 0, len(series)
        if spec.code != "D" and len(series) > 2:
            first, last = events["date"].min(), events["date"].max()
            step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(days=1)
            if (index[0] + step - first) / step < EDGE_COVERAGE:
                start += 1
            if (last - index[-1]) / step < EDGE_COVERAGE:
                end -= 1
        window = series.iloc[start:end]
        if window.min() >= 0 and (window > 0).any():
            threshold = SPARSE_RATIO * float(window[window > 0].median())
            run_start, run_end = _longest_active_run((window >= threshold).to_numpy(), spec.min_dead_run)
            start, end = start + run_start, start + run_end
        # Sağdan kesilme: veri dışa aktarımı dönemin ortasında yapıldıysa son dönemler eksik kayıtlıdır ve
        # sahte bir çöküş gibi görünür. Önceki dönemlerin medyanının yarısının altına düşen son birkaç dönem
        # (en fazla `max_censored`) veri kesimi sayılır; bu, okura ayrıca bildirilir.
        censored = 0
        if series.iloc[start:end].min() >= 0:
            while censored < spec.max_censored and end - start - 1 > spec.reference_window:
                reference = float(series.iloc[max(start, end - 1 - spec.reference_window):end - 1].median())
                if not (reference > 0 and series.iloc[end - 1] < CENSOR_RATIO * reference):
                    break
                end -= 1
                censored += 1
        final = series.iloc[start:end]
        gaps = int((~final.index.isin(grouped.index)).sum())
        return _Built(
            series=final,
            leading_removed=start,
            trailing_removed=len(series) - end - censored,
            censored=censored,
            gaps=gaps,
            zero_share=float((final == 0).mean()) if len(final) else 1.0,
        )

    @staticmethod
    def _choose(built: dict[str, _Built]) -> tuple[FrequencyCode, bool]:
        """En kaba frekanstan başlar: önce mevsimsellik ölçülebilen yoğun seri, sonra yoğun, sonra herhangi biri."""
        def dense(code: str) -> bool:
            return built[code].zero_share <= MAX_ZERO_SHARE

        for code, data in built.items():
            if dense(code) and len(data.series) - _SPECS[code].horizon >= 2 * _SPECS[code].season:
                return code, True  # type: ignore[return-value]
        for require_dense in (True, False):
            for code, data in built.items():
                if (dense(code) or not require_dense) and len(data.series) >= MIN_POINTS:
                    return code, False  # type: ignore[return-value]
        longest = max(len(data.series) for data in built.values())
        raise ValueError(
            f"Zaman serisi analizi için en az {MIN_POINTS} dönem gerekir; seçilen frekansta yalnızca {longest} dönem var."
        )

    # -- Analiz --

    def _analyze_series(
        self, data: _Built, spec: _Spec, seasonal: bool, date_column: str,
        value_column: str | None, value_label: str, notes: list[str],
    ) -> TimeSeriesResult:
        y = data.series
        n = len(y)
        period = spec.season if seasonal else None

        if data.leading_removed or data.trailing_removed:
            notes.insert(0, (
                f"Analiz {format_period(y.index[0], spec.code)} – {format_period(y.index[-1], spec.code)} aralığını kapsıyor; "
                f"başta {data.leading_removed}, sonda {data.trailing_removed} {spec.unit} tam kapsanmadığı veya seyrek "
                f"olduğu (tipik değerin %{decimal_tr(SPARSE_RATIO * 100, 0)}'undan az) için dışarıda bırakıldı."
            ))
        if data.censored:
            notes.append(
                f"Son {data.censored} {spec.unit}, önceki dönemlerin medyanının yarısının altına düştüğü için veri kesimi "
                "(dışa aktarım anında eksik kayıt) olarak değerlendirildi ve analiz dışı bırakıldı. Bu düşüş gerçekse "
                "tahmin iyimser kalır; kaynağı doğrulanmalı."
            )
        if data.gaps:
            notes.append(f"Hiç kaydı olmayan {data.gaps} {spec.unit} 0 olarak sayıldı.")

        trend, seasonal_component, residual, strength = self._decompose(y, period)
        anomalies = self._anomalies(y, residual, spec)

        horizon = spec.horizon
        min_train = 2 * period if period else _MIN_TRAIN_NON_SEASONAL
        folds = max(0, min(BACKTEST_FOLDS, (n - min_train) // horizon))
        model_errors: list[np.ndarray] = []
        baseline_errors: list[np.ndarray] = []
        model_name = ""
        for fold in range(folds):
            cut = n - (folds - fold) * horizon
            train, test = y.iloc[:cut].to_numpy(), y.iloc[cut:cut + horizon].to_numpy()
            fitted, model_name = self._fit(train, period)
            model_errors.append(np.abs(np.asarray(fitted.forecast(horizon), dtype=float) - test))
            baseline_errors.append(np.abs(self._baseline(train, period, horizon) - test))

        final_fit, model_name = self._fit(y.to_numpy(), period)
        prediction = final_fit.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=1 - INTERVAL)
        forecast_index = pd.date_range(y.index[-1], periods=horizon + 1, freq=spec.pandas_freq)[1:]
        mean = prediction["mean"].to_numpy(dtype=float)
        lower = prediction["pi_lower"].to_numpy(dtype=float)
        upper = prediction["pi_upper"].to_numpy(dtype=float)
        if y.min() >= 0 and ((mean < 0).any() or (lower < 0).any()):
            mean, lower, upper = np.clip(mean, 0, None), np.clip(lower, 0, None), np.clip(upper, 0, None)
            notes.append("Negatif olamayan seride tahmin ve alt aralık sıfırda kesildi.")

        model_mae = float(np.concatenate(model_errors).mean()) if model_errors else None
        baseline_mae = float(np.concatenate(baseline_errors).mean()) if baseline_errors else None
        skill = None
        if model_mae is not None and baseline_mae is not None:
            skill = 1 - model_mae / baseline_mae if baseline_mae > 0 else (0.0 if model_mae == 0 else -1.0)
        beats = skill is not None and skill >= SKILL_THRESHOLD
        baseline_name = spec.baseline if period else _NAIVE

        window = min(spec.level_window, max(1, n // 4))
        first_level, last_level = float(y.iloc[:window].mean()), float(y.iloc[-window:].mean())
        level_change = (last_level - first_level) / first_level if first_level > 0 else None

        summary = TimeSeriesSummary(
            date_column=date_column,
            value_column=value_column,
            value_label=value_label,
            frequency=spec.code,
            frequency_label=spec.label,
            period_unit=spec.unit,
            start=y.index[0].date().isoformat(),
            end=y.index[-1].date().isoformat(),
            periods=n,
            seasonal_period=period,
            seasonality_strength=None if strength is None else round(strength, 4),
            level_change=None if level_change is None else round(level_change, 4),
            model_name=model_name,
            baseline_name=baseline_name,
            horizon=horizon,
            backtest_folds=folds,
            model_mae=model_mae,
            baseline_mae=baseline_mae,
            skill=None if skill is None else round(skill, 4),
            beats_baseline=beats,
            forecast_total=float(mean.sum()),
            trend_text=self._trend_text(value_label, spec, window, first_level, last_level, level_change),
            seasonality_text=self._seasonality_text(spec, n, period, strength, seasonal_component),
            verdict=self._verdict(spec, folds, horizon, baseline_name, model_mae, baseline_mae, skill, beats),
            forecast_text=(
                f"Önümüzdeki {horizon} {spec.unit} için beklenen toplam {decimal_tr(float(mean.sum()))} "
                f"({spec.locative} ortalama {decimal_tr(float(mean.mean()))}); her {spec.unit} için "
                f"%{decimal_tr(INTERVAL * 100, 0)} tahmin aralığı grafikte gölgeli alanla gösterildi."
            ),
            notes=notes,
            anomalies=anomalies,
            forecast=[
                ForecastPoint(date=stamp.date().isoformat(), mean=round(m, 4), lower=round(lo, 4), upper=round(hi, 4))
                for stamp, m, lo, hi in zip(forecast_index, mean, lower, upper)
            ],
        )
        figure = self._forecast_figure(y, trend, forecast_index, mean, lower, upper, anomalies, spec, value_label)
        seasonal_figure = (
            self._seasonal_figure(seasonal_component, spec) if seasonal_component is not None else None
        )
        return TimeSeriesResult(summary=summary, forecast_figure=figure, seasonal_figure=seasonal_figure)

    @staticmethod
    def _decompose(
        y: pd.Series, period: int | None
    ) -> tuple[pd.Series, pd.Series | None, pd.Series, float | None]:
        if period:
            from statsmodels.tsa.seasonal import STL

            result = STL(y.to_numpy(), period=period, robust=True).fit()
            trend = pd.Series(result.trend, index=y.index)
            seasonal = pd.Series(result.seasonal, index=y.index)
            residual = pd.Series(result.resid, index=y.index)
            # Güç, olağan dışı dönemler hariç ölçülür: tek bir kampanya günü artık varyansını şişirip
            # gerçek haftalık/yıllık döngüyü görünmez kılmasın.
            resid = result.resid
            center = float(np.median(resid))
            scale = float(np.median(np.abs(resid - center))) * 1.4826
            typical = np.abs(resid - center) < ANOMALY_Z * scale if scale > 0 else np.ones(len(resid), dtype=bool)
            denominator = float(np.var(result.seasonal[typical] + resid[typical]))
            strength = max(0.0, 1 - float(np.var(resid[typical])) / denominator) if denominator > 0 else 0.0
            return trend, seasonal, residual, strength
        window = 5 if len(y) >= 10 else 3
        trend = y.rolling(window, center=True, min_periods=1).mean()
        residual = y - y.rolling(window, center=True, min_periods=1).median()
        return trend, None, residual, None

    @staticmethod
    def _anomalies(y: pd.Series, residual: pd.Series, spec: _Spec) -> list[TimeSeriesAnomaly]:
        center = float(residual.median())
        scale = float((residual - center).abs().median()) * 1.4826
        if not scale > 0:
            return []
        z = (residual - center) / scale
        flagged = z[z.abs() >= ANOMALY_Z].abs().sort_values(ascending=False).head(MAX_ANOMALIES)
        return [
            TimeSeriesAnomaly(
                date=stamp.date().isoformat(),
                label=format_period(stamp, spec.code),
                observed=float(y[stamp]),
                expected=float(y[stamp] - residual[stamp]),
                z=round(float(z[stamp]), 2),
                direction="yüksek" if z[stamp] > 0 else "düşük",
            )
            for stamp in flagged.index
        ]

    @staticmethod
    def _fit(values: np.ndarray, period: int | None) -> tuple[Any, str]:
        """Aday ETS modellerinden AICc'si en düşük olanı seçer (yalnızca verilen eğitim verisiyle).

        Adaylar: sönümlü trend, doğrusal trend, trendsiz. Kurulamayan aday loglanır ve atlanır;
        seçimin geriye dönük test penceresine bakmaması, değerlendirmeyi iyimser yapmaz.
        """
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        season_text = f", {period} dönemlik mevsimsellik" if period else ""
        candidates = (
            ({"trend": "add", "damped_trend": True}, f"ETS (toplamsal, sönümlü trend{season_text})"),
            ({"trend": "add", "damped_trend": False}, f"ETS (toplamsal, doğrusal trend{season_text})"),
            ({"trend": None}, f"ETS (toplamsal, trendsiz{season_text})"),
        )
        best: tuple[float, Any, str] | None = None
        last_error: Exception | None = None
        for options, name in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # statsmodels 0.15: numpy girdide get_prediction çöküyor; tamsayı indeksli seri veriyoruz.
                    fitted = ETSModel(
                        pd.Series(values.astype(float)), error="add", seasonal="add" if period else None,
                        seasonal_periods=period, **options,
                    ).fit(disp=False)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                logger.warning("%s kurulamadı (%s: %s); bu aday atlandı.", name, type(exc).__name__, exc)
                last_error = exc
                continue
            score = float(fitted.aicc) if np.isfinite(fitted.aicc) else float(fitted.aic)
            if best is None or score < best[0]:
                best = (score, fitted, name)
        if best is None:
            raise ValueError(f"Zaman serisi modeli kurulamadı: {last_error}")
        return best[1], best[2]

    @staticmethod
    def _baseline(train: np.ndarray, period: int | None, horizon: int) -> np.ndarray:
        if period:
            return np.resize(train[-period:], horizon)
        return np.full(horizon, train[-1], dtype=float)

    # -- Metinler --

    @staticmethod
    def _trend_text(
        label: str, spec: _Spec, window: int, first: float, last: float, change: float | None
    ) -> str:
        base = (
            f"{label}, ilk {window} {spec.genitive} ortalamasında {spec.locative} {decimal_tr(first)} iken "
            f"son {window} {spec.genitive} ortalamasında {decimal_tr(last)}"
        )
        if change is None:
            return base + "."
        if abs(change) < 0.05:
            return f"{base}; seviye yatay seyretti (%{decimal_tr(change * 100)} değişim)."
        return f"{base}; %{decimal_tr(abs(change) * 100)} {'arttı' if change > 0 else 'azaldı'}."

    @staticmethod
    def _seasonality_text(
        spec: _Spec, n: int, period: int | None, strength: float | None, seasonal: pd.Series | None
    ) -> str:
        if not period or seasonal is None or strength is None:
            needed = 2 * spec.season + spec.horizon
            return (
                f"{spec.cycle} döngüyü ölçmek için en az {needed} {spec.unit} gerekir; seride {n} {spec.unit} var, "
                "mevsimsellik incelenmedi."
            )
        if strength < 0.3:
            return f"Belirgin bir {spec.cycle.lower()} döngü yok (mevsimsellik gücü {decimal_tr(strength, 2)})."
        profile = TimeSeriesEngine._profile(seasonal, spec)
        word = "güçlü" if strength >= 0.6 else "orta düzeyde"
        return (
            f"{spec.cycle} döngü {word} (mevsimsellik gücü {decimal_tr(strength, 2)}): en yoğun {spec.slot} "
            f"{profile.idxmax()} (ortalamadan {_signed(float(profile.max()))}), en sakin {spec.slot} "
            f"{profile.idxmin()} ({_signed(float(profile.min()))})."
        )

    @staticmethod
    def _verdict(
        spec: _Spec, folds: int, horizon: int, baseline: str,
        model_mae: float | None, baseline_mae: float | None, skill: float | None, beats: bool,
    ) -> str:
        if not folds or model_mae is None or baseline_mae is None or skill is None:
            return "Geriye dönük test için dönem sayısı yetersiz; tahmin yalnızca kaba bir yön göstergesidir."
        test = f"{folds} × {horizon} {spec.label} geriye dönük testte"
        errors = f"ortalama mutlak hata: {baseline} {decimal_tr(baseline_mae)}, model {decimal_tr(model_mae)}"
        if beats:
            return f"Model, {test} {baseline} tahmini %{decimal_tr(skill * 100)} geçti ({errors})."
        return (
            f"Model, {test} {baseline} tahmini anlamlı biçimde geçemedi ({errors}). "
            "Tahminler planlamada yalnızca kaba bir yön göstergesi olarak kullanılmalı."
        )

    # -- Grafikler --

    @staticmethod
    def _profile(seasonal: pd.Series, spec: _Spec) -> pd.Series:
        if spec.code == "D":
            keys = seasonal.index.dayofweek
            names = WEEKDAYS_TR
        else:
            keys = seasonal.index.month - 1
            names = MONTHS_TR
        profile = seasonal.groupby(keys).mean()
        profile.index = [names[int(key)] for key in profile.index]
        return profile

    @staticmethod
    def _forecast_figure(
        y: pd.Series, trend: pd.Series, forecast_index: pd.DatetimeIndex,
        mean: np.ndarray, lower: np.ndarray, upper: np.ndarray,
        anomalies: list[TimeSeriesAnomaly], spec: _Spec, value_label: str,
    ) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=y.index, y=y.to_numpy(), mode="lines", name="Gözlenen",
                                    line={"color": "#94a3b8", "width": 1}))
        figure.add_trace(go.Scatter(x=trend.index, y=trend.to_numpy(), mode="lines", name="Eğilim",
                                    line={"color": "#1d5fd1", "width": 2.5}))
        figure.add_trace(go.Scatter(x=forecast_index, y=lower, mode="lines", name="Alt sınır",
                                    line={"color": "#d97706", "width": 0}, showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=forecast_index, y=upper, mode="lines",
                                    name=f"%{decimal_tr(INTERVAL * 100, 0)} tahmin aralığı",
                                    line={"color": "#d97706", "width": 0}, fill="tonexty",
                                    fillcolor="rgba(217,119,6,0.22)"))
        figure.add_trace(go.Scatter(x=forecast_index, y=mean, mode="lines", name="Tahmin",
                                    line={"color": "#d97706", "width": 2.5, "dash": "dash"}))
        if anomalies:
            figure.add_trace(go.Scatter(
                x=[pd.Timestamp(item.date) for item in anomalies], y=[item.observed for item in anomalies],
                mode="markers", name="Olağan dışı", marker={"color": "#dc2626", "size": 9},
            ))
        figure.update_layout(
            title=f"{value_label}: {spec.label} seri ve {spec.horizon} {spec.unit} tahmin",
            xaxis_title="Tarih", yaxis_title=value_label, hovermode="x unified",
            legend={"orientation": "h", "y": -0.2},
        )
        # Plotly'nin varsayılan tarih etiketleri İngilizce ay adı basar ("Apr 2017"); sayısal Türkçe biçim kullanılır.
        figure.update_xaxes(tickformat="%m.%Y", hoverformat="%d.%m.%Y")
        return localize_figure(figure)

    @staticmethod
    def _seasonal_figure(seasonal: pd.Series, spec: _Spec) -> go.Figure:
        profile = TimeSeriesEngine._profile(seasonal, spec)
        title = "Haftanın günlerine göre ortalama etki" if spec.code == "D" else "Aylara göre ortalama etki"
        figure = go.Figure(go.Bar(
            x=list(profile.index), y=profile.to_numpy(), name="Mevsimsel etki",
            marker={"color": ["#1d5fd1" if value >= 0 else "#d97706" for value in profile.to_numpy()]},
        ))
        figure.update_layout(title=title, xaxis_title=spec.slot.capitalize(), yaxis_title="Ortalamadan sapma")
        return localize_figure(figure)
