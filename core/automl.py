"""Veri sızıntısını önleyen deterministik AutoML benchmark motoru."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from core.columns import is_identifier_name, parse_datetime_text
from core.formatting import localize_figure


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
    leakage_suspects: list[str] = field(default_factory=list)
    leakage_warnings: list[str] = field(default_factory=list)
    target: str = ""
    model: Any = None  # en iyi, eğitilmiş pipeline (açıklama ve what-if için)
    background: pd.DataFrame | None = None  # test bölmesinden en fazla 500 satır (hazırlanmış özellikler)
    train_rows: int = 0
    test_rows: int = 0
    cv_folds: int = 0
    split_method: str = ""
    cv_score: float = float("nan")
    cv_std: float = float("nan")
    time_column: str | None = None
    excluded_columns: list[str] = field(default_factory=list)
    acknowledged_columns: list[str] = field(default_factory=list)


# Tek bir özellik, modelin taban üzerindeki kazancının bu kadarını tek başına sağlıyorsa sızıntı şüphesi.
LEAKAGE_SHARE = 0.90
_CV_LARGE_ROWS = 5_000
_MAX_DATE_COLUMNS = 5
# Yalnızca boş/dolu bilgisi kazancın bu kadarını açıklıyorsa (sonradan doldurulan alan izi).
LEAKAGE_MISSING_SHARE = 0.50
_LEAKAGE_CANDIDATES = 5


class AutoMLEngine:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def run(
        self,
        frame: pd.DataFrame,
        target_col: str,
        time_column: str | None = None,
        exclude_columns: Iterable[str] = (),
        known_columns: Iterable[str] = (),
    ) -> AutoMLResult:
        """Sızıntısız model karşılaştırması.

        - Veri bir kez eğitim/test olarak bölünür: `time_column` verilirse kronolojik (son %20 test), yoksa
          rastgele (sınıflandırmada katmanlı).
        - Model seçimi yalnızca eğitim bölmesinde çapraz doğrulamayla yapılır; test bölmesi seçime katılmaz ve
          yalnızca nihai skoru ve taban karşılaştırmasını verir.
        - `exclude_columns`: tahmin anında bilinmeyen sütunlar, modelden çıkarılır.
        - `known_columns`: tahmin anında bilindiği kullanıcı tarafından onaylanan sütunlar; sızıntı uyarısı
          verilmez, onay kayda geçer.
        """
        if target_col not in frame.columns:
            raise ValueError(f"Hedef sütun bulunamadı: {target_col}")
        if self._is_identifier_name(str(target_col)):
            raise ValueError("Kimlik, isim, posta kodu veya referans sütunu hedef olamaz.")
        excluded = [str(column) for column in exclude_columns if str(column) in frame.columns and column != target_col]
        known = {str(column) for column in known_columns}
        data = frame.dropna(subset=[target_col]).copy()
        if time_column is not None:
            if time_column not in data.columns or not pd.api.types.is_datetime64_any_dtype(data[time_column]):
                raise ValueError(f"Zaman sırası için '{time_column}' bir tarih sütunu olmalı.")
            data = data.dropna(subset=[time_column]).sort_values(time_column, kind="stable")
        if data.empty:
            raise ValueError("Hedef sütunda kullanılabilir veri bulunmuyor.")
        y = data.pop(target_col)
        features = [column for column in self._feature_columns(data, target_col) if column not in excluded]
        if not features:
            raise ValueError("Model için uygun özellik sütunu bulunamadı.")
        X = self._prepare_features(data[features])
        if X.empty:
            raise ValueError("Model için uygun özellik sütunu bulunamadı.")
        problem_type = self._problem_type(y, target_col)
        if problem_type == "Classification" and pd.api.types.is_numeric_dtype(y):
            y = y.astype(str)
        classification = problem_type == "Classification"
        if time_column is not None:
            cut = int(len(X) * 0.8)
            if cut < 10 or len(X) - cut < 2:
                raise ValueError("Veri zamana göre eğitim/test ayrımı için yetersiz.")
            X_train, X_test, y_train, y_test = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
            split_method = f"zamana göre ({time_column}; son %20 test)"
        else:
            stratify = y if classification and y.value_counts().min() >= 2 else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=stratify
                )
            except ValueError as exc:
                raise ValueError(f"Veri eğitim/test ayrımı için yetersiz: {exc}") from exc
            split_method = "rastgele, katmanlı (%20 test)" if stratify is not None else "rastgele (%20 test)"
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
        folds = 5 if len(X_train) <= _CV_LARGE_ROWS else 3
        if time_column is not None:
            splitter: Any = TimeSeriesSplit(n_splits=folds)
        elif classification and y_train.value_counts().min() >= folds:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        scoring = "f1_weighted" if classification else "r2"
        sort_column = "F1-Score (Weighted)" if classification else "R2 Score"
        cv_column = f"CV {sort_column} (ort.)"

        baseline_name, baseline = self._baseline(problem_type)
        models = {**self._models(problem_type), baseline_name: baseline}
        scores: list[dict[str, float | str]] = []
        fitted: dict[str, Pipeline] = {}
        for name, model in models.items():
            pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", model)])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    clone(pipeline), X_train, y_train, cv=splitter, scoring=scoring, error_score=np.nan,
                )
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            row: dict[str, float | str] = {
                "Model": name,
                cv_column: float(np.nanmean(cv_scores)) if np.isfinite(cv_scores).any() else np.nan,
                "CV std": float(np.nanstd(cv_scores)) if np.isfinite(cv_scores).any() else np.nan,
            }
            if classification:
                row["Accuracy"] = accuracy_score(y_test, predictions)
                row["F1-Score (Weighted)"] = f1_score(y_test, predictions, average="weighted", zero_division=0)
            else:
                row["R2 Score"] = r2_score(y_test, predictions)
                row["MAE"] = mean_absolute_error(y_test, predictions)
                row["RMSE"] = mean_squared_error(y_test, predictions) ** 0.5
            scores.append(row)
            fitted[name] = pipeline
        # Sıralama ve seçim yalnızca çapraz doğrulamaya göre; test sütunları seçime katılmaz.
        leaderboard = pd.DataFrame(scores).sort_values(cv_column, ascending=False, na_position="last").reset_index(drop=True)
        is_baseline = leaderboard["Model"] == baseline_name
        candidates = leaderboard.loc[~is_baseline & leaderboard[cv_column].notna()]
        if candidates.empty:
            raise ValueError("Çapraz doğrulamada hiçbir model değerlendirilemedi; veri çok küçük veya tek sınıflı.")
        best_row = candidates.iloc[0]
        best_model = str(best_row["Model"])
        best_score = float(best_row[sort_column])
        baseline_score = float(leaderboard.loc[is_baseline, sort_column].iloc[0])
        beats_baseline = best_score - baseline_score > (0.02 if classification else 0.05)
        base_text, best_text = (f"{score:.3f}".replace(".", ",") for score in (baseline_score, best_score))
        cv_text = (
            f"{folds} katlı çapraz doğrulamayla seçildi (CV {float(best_row[cv_column]):.3f} ± "
            f"{float(best_row['CV std']):.3f}); test skoru seçime katılmayan bölmeden"
        ).replace(".", ",")
        if beats_baseline:
            verdict = (
                f"{best_model}, hiçbir özelliğe bakmayan taban modeli geçti "
                f"(test {sort_column}: taban {base_text} → model {best_text}). Model {cv_text}."
            )
        else:
            verdict = (
                f"Modeller taban modeli anlamlı biçimde geçemedi (test {sort_column}: taban {base_text}, "
                f"en iyi model {best_text}). Bu hedef mevcut sütunlarla güvenilir biçimde tahmin edilemiyor. "
                f"Model {cv_text}."
            )
        importance = self._importance(fitted[best_model], X_test, y_test)
        suspects, warnings_text, acknowledged = (
            self._leakage_check(
                X_train, X_test, y_train, y_test, features, importance,
                problem_type, baseline_score, best_score, known,
            )
            if beats_baseline else ([], [], [])
        )
        figure = px.bar(
            importance.head(30).sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            title=f"{best_model} - Özellik Önem Düzeyleri",
        )
        return AutoMLResult(
            problem_type, leaderboard, best_model, importance, localize_figure(figure),
            metric=sort_column, baseline_model=baseline_name, baseline_score=baseline_score,
            best_score=best_score, beats_baseline=beats_baseline, verdict=verdict,
            leakage_suspects=suspects, leakage_warnings=warnings_text, target=str(target_col),
            model=fitted[best_model],
            background=X_test.sample(min(500, len(X_test)), random_state=self.random_state),
            train_rows=len(X_train), test_rows=len(X_test), cv_folds=folds, split_method=split_method,
            cv_score=float(best_row[cv_column]), cv_std=float(best_row["CV std"]),
            time_column=time_column, excluded_columns=excluded, acknowledged_columns=acknowledged,
        )

    def _leakage_check(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
        features: list[str], importance: pd.DataFrame, problem_type: str,
        baseline_score: float, best_score: float, known: set[str] | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        """Olası hedef sızıntısını iki imzayla arar; sonuç uyarıdır, sütun otomatik çıkarılmaz.

        1. Eksiklik imzası: bir sütunun yalnızca boş olup olmaması, modelin taban üzerindeki kazancının en az
           %50'sini açıklıyorsa (ör. "teslim tarihi boş" ⇔ "sipariş teslim edilmedi"). Sonuç gerçekleştikten
           sonra doldurulan alanların tipik izidir. Eksik değeri olan tüm özelliklerde ucuzca hesaplanır.
        2. Baskınlık imzası: en önemli özelliklerden biri, tek başına kazancın en az %90'ını sağlıyorsa
           (ör. hedefin kopyası veya dönüştürülmüş hali).
        Her ikisi de aynı test bölmesinde, aynı metrikle ölçülür.
        """
        lift = best_score - baseline_score
        if lift <= 0:
            return [], [], []
        known = known or set()
        acknowledged: list[str] = []
        classification = problem_type == "Classification"

        def metric(truth: pd.Series, predicted: Any) -> float:
            return (
                f1_score(truth, predicted, average="weighted", zero_division=0)
                if classification else r2_score(truth, predicted)
            )

        def source(column: str) -> str:
            for name in features:
                if column == name or column in (f"{name}_month", f"{name}_dayofweek"):
                    return name
            if column.endswith(" (gün)") and " − " in column:
                # Gün farkı, iki tarihten daha çok boş olanına bağlanır: boşluk imzası oradan gelir.
                later, earlier = column.removesuffix(" (gün)").split(" − ", 1)
                return max(
                    (later, earlier),
                    key=lambda name: X_train[f"{name}_month"].isna().sum() if f"{name}_month" in X_train else -1,
                )
            return column

        groups: dict[str, list[str]] = {}
        for column in X_train.columns:
            groups.setdefault(source(str(column)), []).append(column)
        suspects: list[str] = []
        warnings_text: list[str] = []

        def flag(name: str, share: float, detail: str) -> None:
            if name in known:
                # Kullanıcı bu bilginin tahmin anında bilindiğini onayladı: uyarı yerine kayıt.
                if name not in acknowledged:
                    acknowledged.append(name)
                return
            if name in suspects:
                return
            suspects.append(name)
            warnings_text.append(
                f"Olası hedef sızıntısı: “{name}” {detail} modelin taban üzerindeki başarısının "
                f"%{min(share, 1.0) * 100:.0f}'ini tek başına açıklıyor. Bu bilgi hedefle aynı anda ya da sonra "
                "oluşuyorsa (ör. teslimat tarihi → sipariş durumu) tahmin anında bilinmez; sütunu çıkarıp modeli "
                "yeniden değerlendirin."
            )

        # 1. Eksiklik imzası: ikili gösterge için en iyi tahmin, grup başına en sık sınıf / ortalamadır.
        for name, columns in groups.items():
            train_missing = X_train[columns].isna().any(axis=1)
            test_missing = X_test[columns].isna().any(axis=1)
            if not train_missing.any() or train_missing.all():
                continue
            if classification:
                by_group = y_train.groupby(train_missing).agg(lambda values: values.mode().iloc[0])
            else:
                by_group = y_train.groupby(train_missing).mean()
            fallback = by_group.iloc[0]
            predicted = test_missing.map(lambda missing: by_group.get(missing, fallback))
            share = (metric(y_test, predicted) - baseline_score) / lift
            if share >= LEAKAGE_MISSING_SHARE:
                detail = "sütununun boş olup olmaması"
                if classification:
                    present = y_train[~train_missing].value_counts(normalize=True)
                    if present.iloc[0] >= 0.95:
                        detail = (
                            f"sütununun boş olup olmaması (dolu olduğu kayıtların %{present.iloc[0] * 100:.0f}'inde "
                            f"hedef “{present.index[0]}”)"
                        )
                flag(name, share, detail)

        # 2. Baskınlık imzası: en önemli özellikler tek başına sığ bir modelle.
        ranked: list[str] = []
        for column in importance.sort_values("Importance", ascending=False)["Feature"]:
            name = source(str(column))
            if name not in ranked:
                ranked.append(name)
        for name in ranked[:_LEAKAGE_CANDIDATES]:
            if name in suspects:
                continue
            columns = groups.get(name, [])
            numeric = X_train[columns].select_dtypes(include="number").columns.tolist()
            categorical = [c for c in columns if c not in numeric]
            preprocessor = ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median", add_indicator=True), numeric),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="__eksik__")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
                ]), categorical),
            ])
            model = (
                HistGradientBoostingClassifier(max_depth=3, max_iter=60, random_state=self.random_state)
                if classification
                else HistGradientBoostingRegressor(max_depth=3, max_iter=60, random_state=self.random_state)
            )
            single = Pipeline([("preprocessor", preprocessor), ("model", model)])
            try:
                single.fit(X_train[columns].astype({c: str for c in categorical}), y_train)
                predictions = single.predict(X_test[columns].astype({c: str for c in categorical}))
            except ValueError:
                continue
            share = (metric(y_test, predictions) - baseline_score) / lift
            if share >= LEAKAGE_SHARE:
                flag(name, share, "sütunu")
        return suspects, warnings_text, acknowledged

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
        """Tarihleri ay/haftanın günü ve tarih çiftleri arasındaki gün farkına çevirir.

        Gün farkı satır içi bir hesaptır (iki sütunun farkı); veriden istatistik öğrenmez, bu yüzden
        bölmeden önce yapılması sızıntı yaratmaz. Ör. teslim − tahmini teslim = gecikme (gün).
        """
        prepared = pd.DataFrame(index=frame.index)
        dates: dict[str, pd.Series] = {}
        for column in frame.columns:
            series = frame[column]
            name = str(column)
            parsed = parse_datetime_text(series)
            if parsed is not None:
                prepared[f"{name}_month"] = parsed.dt.month
                prepared[f"{name}_dayofweek"] = parsed.dt.dayofweek
                dates[name] = parsed
            else:
                prepared[name] = series
        # En dolu en fazla beş tarih; her çift için "sonraki − önceki" (medyan zamana göre sıralı).
        chosen = sorted(dates, key=lambda name: dates[name].isna().mean())[:_MAX_DATE_COLUMNS]
        ordered = sorted(chosen, key=lambda name: dates[name].dropna().median())
        for index, earlier in enumerate(ordered):
            for later in ordered[index + 1:]:
                prepared[f"{later} − {earlier} (gün)"] = (
                    (dates[later] - dates[earlier]).dt.total_seconds() / 86_400
                )
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
