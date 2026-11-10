"""Sohbet histogramı aralık etiketlerinin Türkçe sunumu."""

import pandas as pd
import pytest

from core.chat import DataChatEngine


@pytest.mark.parametrize(("values", "expected"), [
    ([1.25, 2.5, 3.75, 4.5], "(1,246 … 2,062]"),
    ([1250.0, 2500.0, 3750.0, 4500.0], "(1.246,749 … 2.062,500]"),
    ([-1.25, 0.0, 1.25, 2.5], "(-1,255 … -0,312]"),
])
def test_histogram_interval_labels_use_turkish_decimals(values, expected) -> None:
    frame = pd.DataFrame({"fiyat": [*values, None]})
    original = frame.copy(deep=True)

    table, _, figure, query = DataChatEngine().answer(frame, "fiyat histogram")

    assert query.operation == "histogram"
    assert table.iloc[0]["Aralık"] == expected
    assert all(" … " in label and "," in label for label in table["Aralık"])
    assert table["Sayı"].sum() == len(values)
    assert list(figure.data[0].x) == values
    pd.testing.assert_frame_equal(frame, original)
