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


def test_missing_values_in_features_are_handled_inside_pipeline(orders: pd.DataFrame) -> None:
    frame = orders.copy()
    frame.loc[frame.index[::7], "product_width_cm"] = np.nan
    frame.loc[frame.index[::5], "product_category_name"] = None
    result = AutoMLEngine().run(frame, "price")
    assert not result.leaderboard.empty
