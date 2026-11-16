from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.automl import AutoMLEngine
from core.explain import ModelExplainer


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(0)
    rows = 800
    frame = pd.DataFrame({
        "indirim": rng.uniform(0, 30, rows),
        "kanal": rng.choice(["mağaza", "web", "telefon"], rows),
        "gürültü": rng.normal(size=rows),
    })
    channel_effect = frame["kanal"].map({"mağaza": 0.0, "web": 15.0, "telefon": -10.0})
    frame["satış_tutarı"] = 100 + 4 * frame["indirim"] + channel_effect + rng.normal(0, 3, rows)
    return AutoMLEngine().run(frame, "satış_tutarı")


@pytest.fixture(scope="module")
def classification():
    rng = np.random.default_rng(1)
    rows = 900
    frame = pd.DataFrame({"bekleme_süresi": rng.uniform(0, 20, rows), "yaş": rng.integers(18, 70, rows)})
    probability = 1 / (1 + np.exp(-(frame["bekleme_süresi"] - 14)))
    frame["durum"] = np.where(rng.uniform(size=rows) < probability, "iptal", "tamam")
    return AutoMLEngine().run(frame, "durum")


def test_automl_exposes_model_and_background(regression) -> None:
    assert regression.model is not None and regression.target == "satış_tutarı"
    assert 0 < len(regression.background) <= 500
    assert set(regression.background.columns) == {"indirim", "kanal", "gürültü"}


def test_numeric_pdp_is_monotonic_and_explained(regression) -> None:
    explanation = ModelExplainer().explain(regression)
    curves = {curve.feature: curve for curve in explanation.curves}
    discount = curves["indirim"]
    assert discount.kind == "sayısal" and len(discount.numeric_grid) == len(discount.values) >= 10
    assert discount.values[-1] - discount.values[0] > 80          # ~4 × 25 puanlık aralık
    assert "arttıkça Tahmin edilen satış_tutarı" in discount.interpretation and "artıyor" in discount.interpretation
    assert explanation.output_label == "Tahmin edilen satış_tutarı"
    assert len(explanation.figures) == len(explanation.curves)


def test_categorical_pdp_names_extremes(regression) -> None:
    curves = {curve.feature: curve for curve in ModelExplainer().explain(regression, top=3).curves}
    channel = curves["kanal"]
    assert channel.kind == "kategorik" and set(channel.grid) == {"mağaza", "web", "telefon"}
    assert "en yüksek “web”" in channel.interpretation and "en düşük “telefon”" in channel.interpretation


def test_classification_uses_minority_class_probability(classification) -> None:
    explanation = ModelExplainer().explain(classification)
    assert explanation.output_label == "“iptal” olasılığı"
    wait = next(curve for curve in explanation.curves if curve.feature == "bekleme_süresi")
    assert wait.values[-1] > wait.values[0] + 0.5 and all(0 <= value <= 1 for value in wait.values)


def test_what_if_regression(regression) -> None:
    explainer = ModelExplainer()
    features = {item.name: item for item in explainer.what_if_features(regression)}
    discount = features["indirim"]
    assert discount.minimum <= discount.default <= discount.maximum
    assert features["kanal"].options and features["kanal"].default in features["kanal"].options
    low = explainer.what_if(regression, {"indirim": 2.0, "kanal": "telefon"})
    high = explainer.what_if(regression, {"indirim": 28.0, "kanal": "web"})
    assert high.value - low.value > 100
    assert high.text.startswith("Tahmin edilen satış_tutarı: ")


def test_what_if_classification(classification) -> None:
    explainer = ModelExplainer()
    calm = explainer.what_if(classification, {"bekleme_süresi": 2.0})
    busy = explainer.what_if(classification, {"bekleme_süresi": 19.5})
    assert calm.predicted_class == "tamam" and busy.predicted_class == "iptal"
    assert sum(busy.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert busy.text.startswith("Tahmin: “iptal” (olasılık %")


def test_leakage_suspects_are_shown_with_a_warning(orders: pd.DataFrame) -> None:
    from core.executor import ActionExecutor

    leaky = AutoMLEngine().run(ActionExecutor().parse_datetime_columns(orders), "order_status")
    assert leaky.leakage_suspects == ["order_delivered_customer_date"]
    explanation = ModelExplainer().explain(leaky, top=6)
    flagged = [curve for curve in explanation.curves if curve.feature.startswith("order_delivered_customer_date")
               or "order_delivered_customer_date" in curve.feature]
    assert flagged and all(curve.interpretation.startswith("Uyarı: olası hedef sızıntısı") for curve in flagged)


def test_errors(regression) -> None:
    from dataclasses import replace

    with pytest.raises(ValueError, match="kullanmıyor"):
        ModelExplainer().what_if(regression, {"yok": 1})
    with pytest.raises(ValueError, match="eğitilmiş model"):
        ModelExplainer().explain(replace(regression, model=None))
