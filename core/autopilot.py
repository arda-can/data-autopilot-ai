"""Sıfır tıklamalı, güvenli ve deterministik veri bilimi otopilotu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from core.automl import AutoMLEngine, AutoMLResult
from core.columns import is_geo_coordinate_name, is_identifier_column
from core.executor import ActionExecutor, AuditLog
from core.formatting import decimal_tr, localize_figure
from core.health import DataHealthAnalyzer, HealthReport, OutlierFinding
from core.metadata import DatasetMetadataCard, MetadataGenerator
from core.planner import STATISTICAL_ACTIONS, GeminiPlanner


@dataclass
class ChartInsight:
    title: str
    reason: str
    insight: str
    figure: Any
    sample_size: int | None = None  # grafik tüm veri yerine örneklemle çizildiyse satır sayısı


@dataclass
class AutopilotResult:
    cleaned_frame: pd.DataFrame
    before_report: HealthReport
    after_report: HealthReport
    audit_logs: list[AuditLog]
    metadata: DatasetMetadataCard
    charts: list[ChartInsight] = field(default_factory=list)
    insight_text: str = ""
    suggested_target: str | None = None
    automl_result: AutoMLResult | None = None
    automl_message: str = ""
    outliers: list[OutlierFinding] = field(default_factory=list)


class AutopilotEngine:
    """Veri yüklenir yüklenmez tüm güvenli analiz aşamalarını çalıştırır."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.analyzer = DataHealthAnalyzer()

    def run_full_autopilot(
        self, df: pd.DataFrame, api_key: str | None = None
    ) -> AutopilotResult:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("Otopilot için boş olmayan bir pandas DataFrame gereklidir.")

        before_report = self.analyzer.analyze(df)
        metadata = MetadataGenerator().generate(df)
        plan = GeminiPlanner(api_key).create_plan(df, metadata)
        executor = ActionExecutor()
        # Yalnızca yapısal hijyen uygulanır. Eksik doldurma ve aykırı değer kırpma istatistikleri
        # veri sızıntısını önlemek için burada değil, modelleme hattında eğitim bölmesinden öğrenilir.
        structural = [item for item in plan.actions if item.action not in STATISTICAL_ACTIONS]
        cleaned = executor.execute_plan(df, structural)

        # Planın kapsamına girmeyen güvenli hijyen adımları da otomatik tamamlanır.
        cleaned = executor.drop_constant_columns(cleaned)
        cleaned = executor.drop_high_null_columns(cleaned)
        cleaned = executor.remove_duplicates(cleaned)
        cleaned = executor.strip_whitespace(cleaned)
        cleaned = executor.clean_currency_and_numeric_strings(cleaned)
        cleaned = executor.parse_datetime_columns(cleaned)

        after_report = self.analyzer.analyze(cleaned)
        outliers = self.analyzer.find_outliers(cleaned)
        charts, insights = self._select_charts(cleaned)
        target = self._suggest_target(cleaned)
        automl_result: AutoMLResult | None = None
        automl_message = ""
        if target is None:
            automl_message = (
                "Bu veri seti referans/konum tablosu veya kimlik ağırlıklı olduğu için "
                "denetimli model eğitimi otomatik atlandı."
            )
        else:
            sample = cleaned
            if len(sample) > 10_000:
                sample = sample.sample(10_000, random_state=self.random_state)
            try:
                automl_result = AutoMLEngine(self.random_state).run(sample, target)
            except (ValueError, TypeError, KeyError) as exc:
                automl_message = f"Uygun hedef bulundu ancak AutoML güvenli şekilde atlandı: {exc}"

        return AutopilotResult(
            cleaned_frame=cleaned,
            before_report=before_report,
            after_report=after_report,
            audit_logs=executor.audit_logs,
            metadata=MetadataGenerator().generate(cleaned),
            charts=charts,
            insight_text="\n".join(insights),
            suggested_target=target,
            automl_result=automl_result,
            automl_message=automl_message,
            outliers=outliers,
        )

    def _select_charts(self, frame: pd.DataFrame) -> tuple[list[ChartInsight], list[str]]:
        numeric = [
            str(column) for column in frame.select_dtypes(include="number").columns
            if not self._is_identifier(frame, str(column))
        ]
        categorical = [
            str(column) for column in frame.columns
            if not pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_datetime64_any_dtype(frame[column])
            and frame[column].value_counts(dropna=True).max() >= 2
            and not self._is_identifier(frame, str(column))
        ]
        dates = [str(column) for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])]
        charts: list[ChartInsight] = []
        insights: list[str] = []
        palette = "Viridis" if len(numeric) % 3 == 0 else ("Plasma" if len(numeric) % 3 == 1 else "Turbo")
        sample = frame.sample(2500, random_state=42) if len(frame) > 2500 else frame
        variances = {column: float(frame[column].var()) for column in numeric}
        ranked_numeric = sorted(numeric, key=lambda column: variances[column], reverse=True)
        # Renk/eksen kırılımı için yalnızca az sınıflı kategori: onlarca sınıf lejantı grafiğin önüne geçirir.
        legend_category = next((column for column in categorical if frame[column].nunique() <= 12), None)

        sampled = len(sample) < len(frame)

        def add(title: str, reason: str, insight: str, figure: Any, sample_size: int | None = None) -> None:
            charts.append(ChartInsight(title, reason, insight, localize_figure(figure), sample_size))
            insights.append(insight)

        if len(categorical) >= 2:
            hierarchy = categorical[:2]
            counts = frame.groupby(hierarchy).size().reset_index(name="Kayıt Sayısı")
            counts = counts.sort_values("Kayıt Sayısı", ascending=False).head(15)
            add(
                "Hiyerarşik Güneş Patlaması",
                "İki anlamlı kategorik/bölgesel sütun bulunduğu için hiyerarşi seçildi.",
                f"{hierarchy[0]} ve {hierarchy[1]} kırılımında en yoğun 15 alt grup gösterildi.",
                px.sunburst(counts, path=hierarchy, values="Kayıt Sayısı", color="Kayıt Sayısı",
                            color_continuous_scale=palette),
            )

        model_frame = frame[numeric].dropna() if numeric else pd.DataFrame()
        cluster_labels: pd.Series | None = None
        if len(numeric) >= 3 and len(model_frame) >= 3:
            scaled = StandardScaler().fit_transform(model_frame)
            cluster_labels = pd.Series(
                KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(scaled),
                index=model_frame.index,
                name="Küme",
            )
            pca = PCA(n_components=2, random_state=42)
            components = pca.fit_transform(scaled)
            pca_frame = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1],
                                      "Küme": cluster_labels.astype(str).to_numpy()}, index=model_frame.index)
            explained = float(pca.explained_variance_ratio_.sum() * 100)
            add(
                "Yapay Zeka Kümeleme Haritası",
                "Üç veya daha fazla metrik PCA ve KMeans ile otomatik segmentlere ayrıldı.",
                f"Üç küme bulundu; ilk iki temel bileşen toplam varyansın %{decimal_tr(explained)} oranını açıklıyor.",
                px.scatter(pca_frame, x="PC1", y="PC2", color="Küme", symbol="Küme",
                           color_discrete_sequence=px.colors.qualitative.Bold,
                           title="PCA + KMeans segmentasyonu"),
            )
            radar_source = model_frame.copy()
            radar_source["Küme"] = cluster_labels.astype(str)
            radar_values = radar_source.groupby("Küme")[numeric].mean()
            radar_values = pd.DataFrame(MinMaxScaler().fit_transform(radar_values),
                                        index=radar_values.index, columns=numeric)
            radar = go.Figure()
            for cluster, row in radar_values.iterrows():
                radar.add_trace(go.Scatterpolar(r=row.tolist() + [row.iloc[0]],
                                                theta=numeric + [numeric[0]],
                                                fill="toself", name=f"Küme {cluster}"))
            radar.update_layout(title="Çok Boyutlu Küme Profilleri", polar={"radialaxis": {"range": [0, 1]}})
            add("Radar / Örümcek Ağı", "Kümelerin normalize edilmiş metrik profilleri karşılaştırıldı.",
                "Kümelerin hangi metriklerde ayrıştığı 0-1 ölçeğinde görülüyor.", radar)

            axes = ranked_numeric[:3]
            scatter3 = sample[axes].dropna()
            color = legend_category
            if color:
                scatter3 = sample[[*axes, color]].dropna()
            add("3 Boyutlu Metrik Uzayı", "Varyansı en yüksek üç güvenli metrik X/Y/Z eksenlerine yerleştirildi.",
                f"{', '.join(axes)} birlikte incelenerek çok boyutlu aykırı ve grup yapısı görülebilir.",
                px.scatter_3d(scatter3, x=axes[0], y=axes[1], z=axes[2], color=color,
                              color_discrete_sequence=px.colors.qualitative.Bold,
                              title="3D metrik uzayı"),
                sample_size=len(scatter3) if sampled else None)
        if len(numeric) >= 3:
            correlation = frame[numeric].corr()
            add("Korelasyon Isı Haritası", "Üç veya daha fazla güvenli sayısal metrik bulundu.",
                "Sayısal metrikler arasındaki doğrusal ilişkiler birlikte incelendi.",
                px.imshow(correlation.round(2), text_auto=True, color_continuous_scale=palette))

        if len(numeric) >= 2:
            correlation = frame[numeric].corr()
            upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool))
            pairs = upper.stack().sort_values(key=lambda values: values.abs(), ascending=False)
            if not pairs.empty:
                first_pair = pairs.index[0]
                pair = [str(first_pair[0]), str(first_pair[1])]
                density = sample[pair].dropna()
                add("2B Yoğunluk Kontur Haritası", "En güçlü sayısal ilişkinin yoğunluk merkezleri seçildi.",
                    f"{pair[0]} ve {pair[1]} arasındaki yoğun bölgeler ve saçılım gösterildi.",
                    px.density_contour(density, x=pair[0], y=pair[1], marginal_x="histogram",
                                       marginal_y="violin"),
                    sample_size=len(density) if sampled else None)

        if numeric:
            value = ranked_numeric[0]
            violin_frame = sample[[value, legend_category]].dropna() if legend_category else sample[[value]].dropna()
            skew = float(frame[value].skew())
            kurtosis = float(frame[value].kurtosis())
            shape = "sağa çarpık" if skew > 0.5 else ("sola çarpık" if skew < -0.5 else "simetrik")
            add("Violin ve Çarpıklık Analizi", "En değişken güvenli metrik için dağılımın tamamı seçildi.",
                f"{value} dağılımı {shape}; çarpıklık {decimal_tr(skew, 2)}, basıklık {decimal_tr(kurtosis, 2)}.",
                px.violin(violin_frame, y=value, x=legend_category,
                          color=legend_category, box=True, points="outliers",
                          color_discrete_sequence=px.colors.qualitative.Bold, title=f"{value} keman dağılımı"),
                sample_size=len(violin_frame) if sampled else None)

        if len(numeric) >= 4:
            parallel = frame[numeric[:6]].dropna()
            parallel_sampled = len(parallel) > 2000
            if parallel_sampled:
                parallel = parallel.sample(2000, random_state=self.random_state)
            parallel = parallel.copy()
            parallel["Profil"] = parallel[numeric[0]]
            add("Paralel Koordinatlar", "Dört veya daha fazla güvenli metrik arasındaki çoklu akış seçildi.",
                "Kayıtların metrikler boyunca izlediği profil ve ayrışmalar gösterildi.",
                px.parallel_coordinates(parallel, color="Profil", dimensions=numeric[:6],
                                         color_continuous_scale=palette),
                sample_size=len(parallel) if parallel_sampled else None)

        if dates:
            date_column = dates[0]
            time_frame = frame[[date_column]].dropna().copy()
            time_frame["Ay"] = time_frame[date_column].dt.to_period("M").astype(str)
            trend = time_frame.groupby("Ay").size().reset_index(name="Kayıt Sayısı")
            add("Zaman Serisi Alan Grafiği", "Tarih sütunundan aylık kayıt trendi çıkarıldı.",
                f"{date_column} için {len(trend)} aylık hareket alan grafiğiyle gösterildi.",
                px.area(trend, x="Ay", y="Kayıt Sayısı", color_discrete_sequence=px.colors.qualitative.Bold))

        if not numeric and categorical:
            column = categorical[0]
            counts = frame[column].value_counts().head(10).rename_axis(column).reset_index(name="Sayı")
            add("Modern Donut Dağılımı", "Güvenli sayısal metrik bulunmadığı için kategori payları seçildi.",
                f"{column} sütunundaki ilk 10 kategorinin payları gösterildi.",
                px.pie(counts, names=column, values="Sayı", hole=0.45,
                       color_discrete_sequence=px.colors.qualitative.Bold))
            add("Kategorik Ağaç Haritası", "Kategorik referans verisinin yoğunluğu hiyerarşik olarak gösterildi.",
                f"{column} değerleri alan büyüklükleriyle karşılaştırıldı.",
                px.treemap(counts, path=[column], values="Sayı", color="Sayı", color_continuous_scale=palette))

        # Isı haritası ve örneklemli paralel koordinatlar altı grafik sınırında da görünür kalır.
        priority = {"Korelasyon Isı Haritası": 0, "Paralel Koordinatlar": 1}
        selected = sorted(charts, key=lambda chart: priority.get(chart.title, 2))[:6]
        return selected, [chart.insight for chart in selected]

    @staticmethod
    def _is_identifier(frame: pd.DataFrame, column: str) -> bool:
        return is_identifier_column(frame[column], column)

    def _suggest_target(self, frame: pd.DataFrame) -> str | None:
        preferred = (
            "target", "hedef", "label", "churn", "score", "status", "durum",
            "sonuç", "sonuc", "class", "harcama", "price", "value", "consumption",
            "weight", "freight", "power", "energy",
        )
        candidates: list[str] = []
        for column in frame.columns:
            name = str(column)
            series = frame[column]
            if (
                series.isna().all()
                or self._is_identifier(frame, name)
                or pd.api.types.is_datetime64_any_dtype(series)
                or is_geo_coordinate_name(name)
            ):
                continue
            if series.nunique(dropna=True) < 2:
                continue
            candidates.append(name)
        if not candidates:
            return None
        preferred_candidates = [column for column in candidates if any(token in column.lower() for token in preferred)]
        if preferred_candidates:
            return max(preferred_candidates, key=lambda column: self._target_score(frame[column], column))
        numeric = [
            column for column in candidates
            if pd.api.types.is_numeric_dtype(frame[column])
        ]
        if numeric:
            return max(numeric, key=lambda column: self._target_score(frame[column], column))
        return None

    def _target_score(self, series: pd.Series, column: str) -> float:
        lowered = column.lower()
        score = 0.0
        if pd.api.types.is_numeric_dtype(series):
            score += 5
        if any(token in lowered for token in ("target", "hedef", "churn", "score", "status", "sonuç", "sonuc")):
            score += 20
        if any(token in lowered for token in ("harcama", "price", "value", "consumption", "weight", "freight", "power", "energy")):
            score += 15
        if 2 <= series.nunique(dropna=True) <= 25:
            score += 2
        return score
