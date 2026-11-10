"""PDF için açık temalı grafik resimleri; Kaleido başarısızsa Matplotlib."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np

from core.formatting import decimal_tr, localize_figure

_SKIPPED = frozenset({"scatter3d", "surface", "mesh3d", "cone", "volume", "isosurface", "parcoords"})


def can_export(figure: Any) -> bool:
    return bool(figure.data) and not any(trace.type in _SKIPPED for trace in figure.data)


def chart_png(figure: Any) -> bytes:
    """Girdiyi değiştirmez; Chrome veya Kaleido engeli PDF üretimini durdurmaz."""
    if not can_export(figure):
        raise ValueError("3B ve paralel koordinatlar PDF'e aktarılmaz.")
    light = localize_figure(figure, light=True)
    try:
        return light.to_image(format="png", width=1000, height=560, scale=1.5)
    except Exception:
        return _matplotlib_png(light)


def _matplotlib_png(figure: Any) -> bytes:
    # pyplot/GUI kullanılmaz; Streamlit ve Windows üzerinde başsız çizim.
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    palette = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#0891b2"]
    with rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.usetex": False}):
        canvas = Figure(figsize=(10, 5.6), dpi=150, facecolor="white", layout="constrained")
        FigureCanvasAgg(canvas)
        polar = all(trace.type == "scatterpolar" for trace in figure.data)
        axis = canvas.add_subplot(111, projection="polar" if polar else None)
        axis.set_facecolor("white")
        formatter = FuncFormatter(lambda value, position: decimal_tr(value, 2))
        axis.yaxis.set_major_formatter(formatter)
        axis.xaxis.set_major_formatter(formatter)
        labels_present = False
        numeric_x, numeric_y = True, True
        for index, trace in enumerate(figure.data):
            color = palette[index % len(palette)]
            label = str(trace.name or "")
            labels_present |= bool(label)
            kind = trace.type
            if kind == "heatmap":
                z = np.asarray(trace.z, dtype=float)
                image = axis.imshow(z, cmap="viridis", aspect="auto")
                x_labels = trace.x if trace.x is not None else range(z.shape[1])
                y_labels = trace.y if trace.y is not None else range(z.shape[0])
                axis.set_xticks(range(z.shape[1]), [str(item) for item in x_labels], rotation=35, ha="right")
                axis.set_yticks(range(z.shape[0]), [str(item) for item in y_labels])
                numeric_x = numeric_y = False
                for (row, column), value in np.ndenumerate(z):
                    axis.text(column, row, decimal_tr(value, 2), ha="center", va="center", fontsize=8)
                canvas.colorbar(image, ax=axis, format=formatter)
            elif kind == "scatterpolar":
                theta = list(trace.theta)
                angles = np.linspace(0, 2 * np.pi, len(theta) - 1, endpoint=False)
                angles = np.append(angles, angles[0])
                axis.plot(angles, trace.r, label=label, color=color)
                axis.fill(angles, trace.r, alpha=0.12, color=color)
                axis.set_xticks(angles[:-1], [str(item) for item in theta[:-1]])
                numeric_x = False
            elif kind in {"sunburst", "treemap"}:
                # Hiyerarşi, aynı toplulaştırılmış yaprak değerleriyle yatay çubuklara aktarılır.
                ids = list(trace.ids if trace.ids is not None else trace.labels)
                parents = set(trace.parents)
                leaves = [i for i, identifier in enumerate(ids) if identifier not in parents]
                axis.barh([str(ids[i]) for i in leaves], [trace.values[i] for i in leaves], color=color)
                numeric_y = False
                axis.set_xlabel("Kayıt Sayısı")
            elif kind == "pie":
                axis.pie(trace.values, labels=trace.labels,
                         autopct=lambda value: f"%{decimal_tr(value)}", colors=palette)
                numeric_x = numeric_y = False
            elif kind == "bar":
                if trace.orientation == "h":
                    axis.barh(list(trace.y), trace.x, color=color, label=label)
                    numeric_y = False
                else:
                    axis.bar(list(trace.x), trace.y, color=color, label=label)
                    numeric_x = np.asarray(trace.x).dtype.kind in "iuf"
            elif kind == "violin":
                if any(item.type == "histogram2dcontour" for item in figure.data):
                    continue
                values = np.asarray(trace.y, dtype=float)
                values = values[np.isfinite(values)]
                if len(values) > 1 and np.ptp(values) > 0:
                    axis.violinplot([values], positions=[index + 1], showmedians=True)
                elif len(values):
                    axis.scatter([index + 1], values[:1], color=color)
                axis.set_xticks(range(1, len(figure.data) + 1), [str(t.name or "Dağılım") for t in figure.data])
                numeric_x = False
            elif kind == "histogram2dcontour":
                image = axis.hist2d(np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float), bins=25, cmap="Blues")
                canvas.colorbar(image[3], ax=axis, format=formatter)
            elif kind == "histogram":
                # Kontur grafiğinin kenar histogramları ana eksene karıştırılmaz.
                if any(item.type == "histogram2dcontour" for item in figure.data):
                    continue
                values = trace.x if trace.x is not None else trace.y
                axis.hist(np.asarray(values, dtype=float), bins=20, color=color, alpha=0.7, label=label)
            elif kind in {"scatter", "scattergl"}:
                x, y = list(trace.x), list(trace.y)
                if "lines" in (trace.mode or "") or trace.fill:
                    axis.plot(x, y, color=color, label=label)
                    if trace.fill:
                        axis.fill_between(x, y, color=color, alpha=0.15)
                else:
                    axis.scatter(x, y, s=10, alpha=0.6, color=color, label=label)
                numeric_x = np.asarray(x).dtype.kind in "iuf"
                numeric_y = np.asarray(y).dtype.kind in "iuf"
            else:
                raise ValueError(f"Statik çizim desteklenmiyor: {kind}")
        if numeric_x:
            axis.xaxis.set_major_formatter(formatter)
        if numeric_y:
            axis.yaxis.set_major_formatter(formatter)
        if figure.layout.xaxis.title.text:
            axis.set_xlabel(figure.layout.xaxis.title.text)
        if figure.layout.yaxis.title.text:
            axis.set_ylabel(figure.layout.yaxis.title.text)
        axis.set_title(figure.layout.title.text or "")
        if labels_present and not any(trace.type in {"pie", "heatmap"} for trace in figure.data):
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
        buffer = BytesIO()
        canvas.savefig(buffer, format="png", facecolor="white")
        canvas.clear()
        return buffer.getvalue()
