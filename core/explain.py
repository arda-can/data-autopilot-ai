"""Model açıklanabilirliği: kısmi bağımlılık (PDP) ve what-if tahmini.

Kısmi bağımlılık, bir özellik değiştirilirken diğerleri olduğu gibi bırakıldığında modelin ortalama
tahmininin nasıl değiştiğini gösterir. Modelin öğrendiği ilişkidir, nedensellik değildir. Hesap, test
bölmesinden alınan en fazla 500 satırlık arka plan üzerinde yapılır; eğitim verisi kullanılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
from pydantic import BaseModel, Field

from core.automl import AutoMLResult
from core.formatting import decimal_tr, localize_figure

_GRID_POINTS = 20
_MAX_CATEGORIES = 12
_WHAT_IF_FEATURES = 5
PDP_NOTE = (
    "Kısmi bağımlılık: yalnızca bu özellik değiştirilirken diğer özellikler olduğu gibi bırakıldığında modelin "
    "ortalama tahmini. Modelin öğrendiği ilişkidir, nedensellik değildir."
)


class PartialDependence(BaseModel):
    feature: str
    kind: Literal["sayısal", "kategorik"]
    grid: list[str]
    numeric_grid: list[float] = Field(default_factory=list)
    values: list[float]
    output_label: str
    interpretation: str


class WhatIfFeature(BaseModel):
    name: str
    kind: Literal["sayısal", "kategorik"]
    default: float | str
    minimum: float | None = None
    maximum: float | None = None
    options: list[str] = Field(default_factory=list)


class WhatIfPrediction(BaseModel):
    value: float | None = None
    predicted_class: str | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    text: str


@dataclass
class ModelExplanation:
    curves: list[PartialDependence]
    figures: list[Any] = field(default_factory=list)
    what_if_features: list[WhatIfFeature] = field(default_factory=list)
    output_label: str = ""
    skipped: list[str] = field(default_factory=list)


def _num(value: float) -> str:
    return decimal_tr(float(value), 2)


def _is_suspect(column: str, suspects: list[str]) -> bool:
    return any(column == name or column.startswith(f"{name}_") or f" {name} " in f" {column} " for name in suspects)


def _ranked_features(automl: AutoMLResult) -> list[str]:
    importance = automl.feature_importance.sort_values("Importance", ascending=False)
    return [str(name) for name, value in zip(importance["Feature"], importance["Importance"]) if value > 0]


class ModelExplainer:
    """En iyi pipeline'ı kara kutu olarak kullanır: yalnızca `predict` / `predict_proba` çağrılır."""

    def explain(self, automl: AutoMLResult, top: int = 3) -> ModelExplanation:
        if automl.model is None or automl.background is None or automl.background.empty:
            raise ValueError("Açıklama için eğitilmiş model ve arka plan verisi gerekir.")
        background = automl.background
        # Sızıntı şüphesi olan özellikler gizlenmez: modelin ondan ne öğrendiğini görmek teşhise yardım eder.
        # Yorumlarına uyarı eklenir; öneri metinleri ise bu özellikleri zaten "izlenmeli" diye önermez.
        chosen = [name for name in _ranked_features(automl) if name in background.columns][:top]
        skipped: list[str] = []
        class_index, output_label = self._output(automl, background, chosen)
        curves, figures = [], []
        for name in chosen:
            curve = self._curve(automl, background, name, class_index, output_label)
            if curve is None:
                skipped.append(f"“{name}” arka planda tek değer aldığı için çizilmedi.")
                continue
            if _is_suspect(name, automl.leakage_suspects):
                curve.interpretation = f"Uyarı: olası hedef sızıntısı şüphesi olan özellik. {curve.interpretation}"
            curves.append(curve)
            figures.append(self._figure(curve))
        return ModelExplanation(
            curves=curves, figures=figures, output_label=output_label, skipped=skipped,
            what_if_features=self.what_if_features(automl),
        )

    # -- PDP --

    @staticmethod
    def _predict(automl: AutoMLResult, frame: pd.DataFrame, class_index: int | None) -> np.ndarray:
        if class_index is None:
            return np.asarray(automl.model.predict(frame), dtype=float)
        return np.asarray(automl.model.predict_proba(frame), dtype=float)[:, class_index]

    def _output(self, automl: AutoMLResult, background: pd.DataFrame, chosen: list[str]) -> tuple[int | None, str]:
        """Regresyonda tahmin; sınıflandırmada gösterilecek sınıfın olasılığı.

        İki sınıfta azınlık sınıf (genellikle ilgilenilen olay); çok sınıfta en önemli özellik değiştikçe
        olasılığı en çok oynayan sınıf seçilir.
        """
        if automl.problem_type != "Classification":
            return None, f"Tahmin edilen {automl.target}" if automl.target else "Tahmin"
        classes = [str(item) for item in automl.model.classes_]
        mean_probability = np.asarray(automl.model.predict_proba(background), dtype=float).mean(axis=0)
        if len(classes) == 2 or not chosen:
            best = int(np.argmin(mean_probability))
        else:
            rows = [
                np.asarray(automl.model.predict_proba(background.assign(**{chosen[0]: value})), dtype=float).mean(axis=0)
                for value in self._grid(background[chosen[0]])[0]
            ]
            matrix = np.vstack(rows)
            best = int(np.argmax(matrix.max(axis=0) - matrix.min(axis=0)))
        return best, f"“{classes[best]}” olasılığı"

    @staticmethod
    def _grid(series: pd.Series) -> tuple[list[Any], Literal["sayısal", "kategorik"]]:
        values = series.dropna()
        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            unique = np.sort(values.unique())
            if len(unique) <= _GRID_POINTS:
                return [float(item) for item in unique], "sayısal"
            return [float(item) for item in np.unique(np.quantile(values, np.linspace(0.05, 0.95, _GRID_POINTS)))], "sayısal"
        return [str(item) for item in values.astype(str).value_counts().index[:_MAX_CATEGORIES]], "kategorik"

    def _curve(
        self, automl: AutoMLResult, background: pd.DataFrame, name: str, class_index: int | None, output_label: str,
    ) -> PartialDependence | None:
        grid, kind = self._grid(background[name])
        if len(grid) < 2:
            return None
        means = []
        for value in grid:
            probe = background.copy()
            probe[name] = value
            means.append(float(self._predict(automl, probe, class_index).mean()))
        labels = [_num(value) for value in grid] if kind == "sayısal" else [str(value) for value in grid]
        return PartialDependence(
            feature=name, kind=kind, grid=labels, values=[round(value, 6) for value in means],
            numeric_grid=[float(value) for value in grid] if kind == "sayısal" else [],
            output_label=output_label, interpretation=self._interpret(name, kind, labels, means, output_label),
        )

    @staticmethod
    def _interpret(name: str, kind: str, labels: list[str], means: list[float], output: str) -> str:
        values = np.asarray(means)
        high, low = int(values.argmax()), int(values.argmin())
        if kind == "kategorik":
            return (
                f"“{name}”: {output} en yüksek “{labels[high]}” ({_num(values[high])}), "
                f"en düşük “{labels[low]}” ({_num(values[low])})."
            )
        span = float(values.max() - values.min())
        if span <= 1e-9:
            return f"“{name}” değiştikçe {output} neredeyse sabit kalıyor; model bu özelliğe duyarsız."
        steps = np.diff(values)
        tolerance = span * 0.02
        if (steps >= -tolerance).all():
            trend = "artıyor"
        elif (steps <= tolerance).all():
            trend = "azalıyor"
        else:
            return (
                f"“{name}” ile {output} arasında doğrusal olmayan bir ilişki var: en yüksek tahmin "
                f"{labels[high]} civarında ({_num(values[high])}), en düşük {labels[low]} civarında ({_num(values[low])})."
            )
        return (
            f"“{name}” {labels[0]} → {labels[-1]} aralığında arttıkça {output} ortalama "
            f"{_num(values[0])} → {_num(values[-1])} {trend}."
        )

    @staticmethod
    def _figure(curve: PartialDependence) -> Any:
        frame = pd.DataFrame({curve.feature: curve.grid, curve.output_label: curve.values})
        if curve.kind == "sayısal":
            frame[curve.feature] = curve.numeric_grid
            figure = px.line(frame, x=curve.feature, y=curve.output_label, markers=True)
        else:
            figure = px.bar(frame, x=curve.feature, y=curve.output_label)
        figure.update_layout(title=f"Kısmi bağımlılık: {curve.feature}")
        return localize_figure(figure)

    # -- What-if --

    @staticmethod
    def what_if_features(automl: AutoMLResult, count: int = _WHAT_IF_FEATURES) -> list[WhatIfFeature]:
        background = automl.background
        ranked = [name for name in _ranked_features(automl) if name in background.columns]
        features = []
        for name in ranked[:count]:
            values = background[name].dropna()
            if values.empty:
                continue
            if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                low, high = (float(item) for item in values.quantile([0.01, 0.99]))
                if high <= low:
                    continue
                features.append(WhatIfFeature(name=name, kind="sayısal", default=float(values.median()),
                                              minimum=low, maximum=high))
            else:
                options = [str(item) for item in values.astype(str).value_counts().index[:_MAX_CATEGORIES * 2]]
                features.append(WhatIfFeature(name=name, kind="kategorik", default=options[0], options=options))
        return features

    @staticmethod
    def typical_row(automl: AutoMLResult) -> pd.DataFrame:
        """Arka planın tipik satırı: sayısalda medyan, diğerlerinde en sık değer."""
        background = automl.background
        row = {}
        for column in background.columns:
            values = background[column].dropna()
            if values.empty:
                row[column] = np.nan
            elif pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
                row[column] = float(values.median())
            else:
                row[column] = values.mode().iloc[0]
        return pd.DataFrame([row], columns=background.columns).astype(background.dtypes.to_dict(), errors="ignore")

    def what_if(self, automl: AutoMLResult, overrides: dict[str, Any]) -> WhatIfPrediction:
        if automl.model is None or automl.background is None:
            raise ValueError("What-if için eğitilmiş model gerekir.")
        row = self.typical_row(automl)
        for name, value in overrides.items():
            if name not in row.columns:
                raise ValueError(f"Model “{name}” özelliğini kullanmıyor.")
            row[name] = value
        if automl.problem_type == "Classification":
            probabilities = automl.model.predict_proba(row)[0]
            classes = [str(item) for item in automl.model.classes_]
            ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
            best, chance = ranked[0]
            return WhatIfPrediction(
                predicted_class=best, probabilities={label: round(float(p), 4) for label, p in ranked},
                text=f"Tahmin: “{best}” (olasılık %{decimal_tr(float(chance) * 100)}).",
            )
        value = float(automl.model.predict(row)[0])
        return WhatIfPrediction(value=value, text=f"Tahmin edilen {automl.target or 'değer'}: {_num(value)}.")
