"""Veri sızıntısını önleyen deterministik AutoML benchmark motoru."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from core.columns import is_identifier_name, parse_datetime_text


@dataclass
class AutoMLResult:
    problem_type: str
    leaderboard: pd.DataFrame
    best_model: str
    feature_importance: pd.DataFrame
    figure: Any
    metric: str = ""
    baseline_model: str = ""
    baseline_score: float = 0.0
    best_score: float = 0.0
    beats_baseline: bool = False
    verdict: str = ""


class AutoMLEngine:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def run(self, frame: pd.DataFrame, target_col: str) -> AutoMLResult:
        if target_col not in frame.columns:
            raise ValueError(f"Hedef sütun bulunamadı: {target_col}")
        if self._is_identifier_name(str(target_col)):
            raise ValueError("Kimlik, isim, posta kodu veya referans sütunu hedef olamaz.")
        data = frame.dropna(subset=[target_col]).copy()
        if data.empty:
            raise ValueError("Hedef sütunda kullanılabilir veri bulunmuyor.")
        y = data.pop(target_col)
        features = self._feature_columns(data, target_col)
        if not features:
            raise ValueError("Model için uygun özellik sütunu bulunamadı.")
        X = self._prepare_features(data[features])
        if X.empty:
            raise ValueError("Model için uygun özellik sütunu bulunamadı.")
        problem_type = self._problem_type(y, target_col)
        if problem_type == "Classification" and pd.api.types.is_numeric_dtype(y):
            y = y.astype(str)
        stratify = y if problem_type == "Classification" and y.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=stratify
            )
        except ValueError as exc:
            raise ValueError(f"Veri eğitim/test ayrımı için yetersiz: {exc}") from exc
        numeric = X_train.select_dtypes(include="number").columns.tolist()
        categorical = [column for column in X_train.columns if column not in numeric]
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]), categorical),
            ],
            remainder="drop",
        )
        baseline_name, baseline = self._baseline(problem_type)
        models = {**self._models(problem_type), baseline_name: baseline}
        scores: list[dict[str, float | str]] = []
        fitted: dict[str, Pipeline] = {}
        for name, model in models.items():
            pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            if problem_type == "Classification":
                scores.append({
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "F1-Score (Weighted)": f1_score(y_test, predictions, average="weighted", zero_division=0),
                })
            else:
                scores.append({
                    "Model": name,
                    "R2 Score": r2_score(y_test, predictions),
                    "MAE": mean_absolute_error(y_test, predictions),
                    "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                })
            fitted[name] = pipeline
        sort_column = "F1-Score (Weighted)" if problem_type == "Classification" else "R2 Score"
        leaderboard = pd.DataFrame(scores).sort_values(sort_column, ascending=False).reset_index(drop=True)
        # Taban model hiçbir özelliğe bakmaz; gerçek modeller ancak onu geçerse bir şey öğrenmiş sayılır.
        is_baseline = leaderboard["Model"] == baseline_name
        best_model = str(leaderboard.loc[~is_baseline, "Model"].iloc[0])
        best_score = float(leaderboard.loc[~is_baseline, sort_column].iloc[0])
        baseline_score = float(leaderboard.loc[is_baseline, sort_column].iloc[0])
        beats_baseline = best_score - baseline_score > (0.02 if problem_type == "Classification" else 0.05)
        base_text, best_text = (f"{score:.3f}".replace(".", ",") for score in (baseline_score, best_score))
        if beats_baseline:
            verdict = (
                f"{best_model}, hiçbir özelliğe bakmayan taban modeli geçti "
                f"({sort_column}: taban {base_text} → model {best_text})."
            )
        else:
            verdict = (
                f"Modeller taban modeli anlamlı biçimde geçemedi ({sort_column}: taban {base_text}, "
                f"en iyi model {best_text}). Bu hedef mevcut sütunlarla güvenilir biçimde tahmin edilemiyor."
            )
        importance = self._importance(fitted[best_model], X_test, y_test)
        figure = px.bar(
            importance.head(30).sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title=f"{best_model} - Özellik Önem Düzeyleri",
        )
        return AutoMLResult(
            problem_type, leaderboard, best_model, importance, figure,
            metric=sort_column, baseline_model=baseline_name, baseline_score=baseline_score,
            best_score=best_score, beats_baseline=beats_baseline, verdict=verdict,
        )

    @staticmethod
    def _problem_type(target: pd.Series, target_col: str = "") -> str:
        unique = target.nunique(dropna=True)
        threshold = max(10, min(20, int(len(target) * 0.05)))
        metric_tokens = (
            "harcama", "price", "value", "freight", "weight", "consumption",
            "voltage", "current", "power", "temperature", "humidity", "score",
            "tutar", "fiyat", "lat", "lng",
        )
        is_measurement = any(token in target_col.lower() for token in metric_tokens)
        integer_score = (
            "score" in target_col.lower()
            and pd.api.types.is_integer_dtype(target)
            and unique <= 10
        )
        if (
            pd.api.types.is_numeric_dtype(target)
            and not integer_score
            and (
                (pd.api.types.is_float_dtype(target) and unique > 2)
                or is_measurement
            )
        ):
            return "Regression"
        return "Classification" if (
            not pd.api.types.is_numeric_dtype(target) or unique <= threshold
        ) else "Regression"

    @staticmethod
    def _feature_columns(frame: pd.DataFrame, target_col: str) -> list[str]:
        features: list[str] = []
        for column in frame.columns:
            series = frame[column]
            lowered = str(column).lower()
            if column == target_col or series.isna().all() or series.nunique(dropna=True) <= 1:
                continue
            if pd.api.types.is_datetime64_any_dtype(series) or AutoMLEngine._is_date_like(series):
                features.append(str(column))
                continue
            if AutoMLEngine._is_identifier_name(lowered):
                continue
            if (
                not pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_datetime64_any_dtype(series)
                and series.nunique(dropna=True) > 30
                and not AutoMLEngine._is_date_like(series)
            ):
                continue
            if series.nunique(dropna=True) >= max(50, int(len(frame) * 0.95)) and not pd.api.types.is_numeric_dtype(series):
                continue
            features.append(str(column))
        return features

    @staticmethod
    def _is_date_like(series: pd.Series) -> bool:
        return parse_datetime_text(series) is not None

    @staticmethod
    def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            series = frame[column]
            name = str(column)
            parsed = parse_datetime_text(series)
            if parsed is not None:
                prepared[f"{name}_month"] = parsed.dt.month
                prepared[f"{name}_dayofweek"] = parsed.dt.dayofweek
            else:
                prepared[name] = series
        return prepared

    @staticmethod
    def _is_identifier_name(column: str) -> bool:
        return is_identifier_name(column)

    @staticmethod
    def _baseline(problem_type: str) -> tuple[str, Any]:
        if problem_type == "Classification":
            return "Taban (en sık sınıf)", DummyClassifier(strategy="most_frequent")
        return "Taban (ortalama)", DummyRegressor(strategy="mean")

    def _models(self, problem_type: str) -> dict[str, Any]:
        if problem_type == "Classification":
            return {
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "RandomForestClassifier": RandomForestClassifier(n_estimators=150, random_state=self.random_state, n_jobs=-1),
                "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=self.random_state),
            }
        return {
            "Ridge": Ridge(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=150, random_state=self.random_state, n_jobs=-1),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=self.random_state),
        }

    def _importance(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        names = [str(column) for column in X_test.columns]
        try:
            permutation = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=3,
                random_state=self.random_state,
                n_jobs=-1,
            )
            values = permutation.importances_mean
        except Exception:
            transformed_names = list(preprocessor.get_feature_names_out())
            if hasattr(model, "feature_importances_"):
                raw_values = model.feature_importances_
            elif hasattr(model, "coef_"):
                coefficients = model.coef_
                raw_values = (
                    abs(coefficients).mean(axis=0)
                    if getattr(coefficients, "ndim", 1) > 1
                    else abs(coefficients)
                )
            else:
                raw_values = []
            values = pd.Series(0.0, index=names)
            for transformed_name, value in zip(transformed_names, raw_values):
                base_name = transformed_name.split("__", 1)[-1].rsplit("_", 1)[0]
                matching = next((name for name in names if base_name == name), None)
                if matching is not None:
                    values[matching] += float(value)
            values = values.to_numpy()
        return pd.DataFrame({"Feature": names, "Importance": values}).sort_values(
            "Importance", ascending=False
        ).reset_index(drop=True)
