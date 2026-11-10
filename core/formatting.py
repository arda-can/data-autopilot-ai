"""Veriyi değiştirmeden Türkçe sayı ve grafik sunumu."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import pandas as pd

_TR_SEPARATORS = str.maketrans({",": ".", ".": ","})


def decimal_tr(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}".translate(_TR_SEPARATORS)


def integer_tr(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Tablo sunum kopyası; hesaplama ve CSV'deki sayısal tipler korunur."""
    def cell(value: Any) -> Any:
        if isinstance(value, Real) and not isinstance(value, (Integral, bool)):
            return "" if pd.isna(value) else decimal_tr(value, 3)
        if isinstance(value, (list, tuple)):
            return " | ".join(str(cell(item)) for item in value)
        return value

    return frame.map(cell)


def localize_figure(figure: Any, *, light: bool = False) -> Any:
    """Bağımsız sunum kopyasında eksen, renk ölçeği ve hücre metinlerini yerelleştirir."""
    import plotly.graph_objects as go

    result = go.Figure(figure)
    result.update_layout(separators=",.")
    if light:
        result.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                             font_color="#16202e")
    for trace in result.data:
        if trace.type == "heatmap" and trace.z is not None:
            trace.text = [[decimal_tr(float(value), 2) for value in row] for row in trace.z]
            trace.texttemplate = "%{text}"
        if trace.type == "parcoords":
            for dimension in trace.dimensions:
                values = pd.Series(dimension.values).dropna()
                if not values.empty:
                    low, high = float(values.min()), float(values.max())
                    ticks = [low + (high - low) * index / 4 for index in range(5)]
                    dimension.tickvals = ticks
                    dimension.ticktext = [decimal_tr(value, 2) for value in ticks]
    return result
