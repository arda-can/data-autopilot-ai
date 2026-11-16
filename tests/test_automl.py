from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.automl import AutoMLEngine


def test_identifier_cannot_be_target(orders: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        AutoMLEngine().run(orders, "order_id")


def test_measurement_columns_are_kept_as_features(orders: pd.DataFrame) -> None:
    features = AutoMLEngine._feature_columns(orders.drop(columns=["price"]), "price")
    assert "product_width_cm" in features
    assert "product_category_name" in features
    assert "order_id" not in features and "customer_id" not in features


def test_column_containing_id_substring_can_be_target() -> None:
    rng = np.random.default_rng(0)
    temperature = rng.normal(20, 5, 200)
    frame = pd.DataFrame({"temperature": temperature, "humidity": 80 - 1.5 * temperature + rng.normal(0, 1, 200)})
    result = AutoMLEngine().run(frame, "humidity")
    assert result.problem_type == "Regression"
    assert result.leaderboard["R2 Score"].max() > 0.8


def test_regression_learns_known_signal(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "price")
    assert result.problem_type == "Regression"
    assert result.leaderboard["R2 Score"].max() > 0.8
    assert "product_width_cm" in result.feature_importance["Feature"].tolist()


def test_baseline_is_reported_and_beaten_when_signal_exists(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "price")
    assert result.baseline_model in result.leaderboard["Model"].tolist()
    assert result.best_model != result.baseline_model
    assert result.beats_baseline and result.best_score > result.baseline_score
    assert "geçti" in result.verdict


def test_no_signal_is_reported_honestly() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "hedef_deger": rng.normal(size=300)})
    result = AutoMLEngine().run(frame, "hedef_deger")
    assert not result.beats_baseline
    assert "geçemedi" in result.verdict


def test_classification_baseline_is_majority_class(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "order_status")
    assert result.problem_type == "Classification"
    majority_share = orders["order_status"].value_counts(normalize=True).iloc[0]
    baseline_row = result.leaderboard[result.leaderboard["Model"] == result.baseline_model].iloc[0]
    assert baseline_row["Accuracy"] == pytest.approx(majority_share, abs=0.05)


def test_missing_values_in_features_are_handled_inside_pipeline(orders: pd.DataFrame) -> None:
    frame = orders.copy()
    frame.loc[frame.index[::7], "product_width_cm"] = np.nan
    frame.loc[frame.index[::5], "product_category_name"] = None
    result = AutoMLEngine().run(frame, "price")
    assert not result.leaderboard.empty


# --- Hedef sızıntısı uyarısı -----------------------------------------------------------------------


def test_post_outcome_column_is_flagged_as_leakage(orders: pd.DataFrame) -> None:
    """Teslim tarihi yalnızca teslim edilen siparişlerde dolu: sipariş durumunu ele verir."""
    from core.executor import ActionExecutor

    frame = ActionExecutor().parse_datetime_columns(orders)
    result = AutoMLEngine().run(frame, "order_status")
    assert result.beats_baseline
    assert result.leakage_suspects == ["order_delivered_customer_date"]
    warning = result.leakage_warnings[0]
    assert warning.startswith("Olası hedef sızıntısı: “order_delivered_customer_date”")
    assert "dolu olduğu kayıtların %100'inde hedef “delivered”" in warning


def test_legitimate_strong_predictors_are_not_flagged(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "price")
    assert result.beats_baseline and result.leakage_suspects == [] and result.leakage_warnings == []


def test_transformed_copy_of_target_is_flagged() -> None:
    rng = np.random.default_rng(2)
    target = rng.normal(100, 20, 400)
    frame = pd.DataFrame({
        "tutar": target,
        "kdv_dahil_tutar": target * 1.2,
        "a": rng.normal(size=400),
        "b": rng.normal(size=400),
    })
    result = AutoMLEngine().run(frame, "tutar")
    assert "kdv_dahil_tutar" in result.leakage_suspects
    assert "“kdv_dahil_tutar” sütunu modelin" in result.leakage_warnings[0]


def test_no_leakage_check_when_model_does_not_beat_baseline() -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "hedef_deger": rng.normal(size=300)})
    result = AutoMLEngine().run(frame, "hedef_deger")
    assert not result.beats_baseline and result.leakage_warnings == []


def test_date_difference_features() -> None:
    """Gecikme (teslim − tahmini teslim) puanı belirliyorsa model bunu gün farkı özelliğinden öğrenir."""
    rng = np.random.default_rng(5)
    rows = 600
    purchase = pd.Timestamp("2018-01-01") + pd.to_timedelta(rng.integers(0, 300, rows), unit="D")
    estimated = purchase + pd.Timedelta(days=20)
    delay = rng.integers(-10, 15, rows)
    delivered = estimated + pd.to_timedelta(delay, unit="D")
    frame = pd.DataFrame({
        "satin_alma": purchase, "tahmini_teslim": estimated, "teslim": delivered,
        "puan": np.clip(4.5 - 0.25 * np.clip(delay, 0, None) + rng.normal(0, 0.3, rows), 1, 5),
    })
    prepared = AutoMLEngine._prepare_features(frame.drop(columns=["puan"]))
    assert "teslim − tahmini_teslim (gün)" in prepared.columns
    assert prepared["teslim − tahmini_teslim (gün)"].tolist() == pytest.approx(delay.astype(float).tolist())
    result = AutoMLEngine().run(frame, "puan")
    assert result.beats_baseline and result.best_score > 0.7
    assert result.feature_importance["Feature"].iloc[0] in (
        "teslim − tahmini_teslim (gün)", "teslim − satin_alma (gün)",
    )


# --- Değerlendirme: çapraz doğrulamayla seçim, ayrı test, zaman bölmesi ve tahmin anı -------------


def test_model_is_selected_by_cross_validation_not_test(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "price")
    cv_column = "CV R2 Score (ort.)"
    assert {cv_column, "CV std", "R2 Score", "MAE", "RMSE"} <= set(result.leaderboard.columns)
    non_baseline = result.leaderboard[result.leaderboard["Model"] != result.baseline_model]
    assert result.best_model == non_baseline.sort_values(cv_column, ascending=False)["Model"].iloc[0]
    assert result.leaderboard[cv_column].is_monotonic_decreasing
    assert result.train_rows + result.test_rows == len(orders) and result.test_rows == 60
    assert result.cv_folds == 5 and result.split_method.startswith("rastgele")
    assert "5 katlı çapraz doğrulamayla seçildi" in result.verdict
    assert result.cv_score == pytest.approx(non_baseline[cv_column].max())


def test_time_split_tests_on_the_latest_period() -> None:
    rng = np.random.default_rng(6)
    rows = 400
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    frame = pd.DataFrame({"tarih": dates[rng.permutation(rows)], "x": rng.normal(size=rows)})
    frame["y"] = 3 * frame["x"] + rng.normal(0, 0.3, rows)
    result = AutoMLEngine().run(frame, "y", time_column="tarih")
    assert result.split_method.startswith("zamana göre (tarih") and result.time_column == "tarih"
    assert result.test_rows == 80 and result.beats_baseline
    test_months = set(result.background["tarih_month"])
    latest = frame.sort_values("tarih").iloc[-80:]
    assert test_months <= set(latest["tarih"].dt.month)
    with pytest.raises(ValueError, match="tarih sütunu olmalı"):
        AutoMLEngine().run(frame, "y", time_column="x")


def test_excluded_columns_are_not_used(orders: pd.DataFrame) -> None:
    result = AutoMLEngine().run(orders, "price", exclude_columns=["product_width_cm"])
    assert result.excluded_columns == ["product_width_cm"]
    assert "product_width_cm" not in set(result.feature_importance["Feature"])


def test_known_columns_suppress_leakage_warning_but_are_recorded(orders: pd.DataFrame) -> None:
    from core.executor import ActionExecutor

    frame = ActionExecutor().parse_datetime_columns(orders)
    result = AutoMLEngine().run(frame, "order_status", known_columns=["order_delivered_customer_date"])
    assert result.leakage_warnings == [] and result.leakage_suspects == []
    assert result.acknowledged_columns == ["order_delivered_customer_date"]
