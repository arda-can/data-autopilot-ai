"""A/B test analizi: oran ve ortalama testleri, güven aralıkları, güç/örneklem ve SRM kontrolü.

- Oran (satır düzeyi): iki oranlı z-testi (p-değeri birleşik varyansla), fark için Wald güven aralığı.
- Ortalama ya da birim düzeyi: Welch t-testi ve Welch serbestlik dereceli güven aralığı.
- Birim (randomizasyon birimi, ör. müşteri) verilirse metrik önce birim başına ortalanır; aynı müşterinin
  siparişleri bağımsız sayılmaz. Her iki grupta görünen birimler (atama bozulmuş) analiz dışı bırakılır.
- SRM: gözlenen birim sayılarının beklenen paya uyumu ki-kare testiyle sınanır; p < 0,001 alarmdır ve
  sonuç yorumlanmaz.
- Güç: hedef göreli etki (MDE) için grup başına gereken örneklem, mevcut örneklemle yakalanabilecek en küçük
  göreli fark ve hedef etkide güç; normal yaklaşımla.

Göreli artışın güven aralığı, kontrol ortalamasının belirsizliğini yok sayan yaklaşık bir aralıktır.
Testin sonunda tek bir bakış varsayılır; ara bakışlar (peeking) ve çok sayıda metrik yanlış pozitif oranını artırır.
"""

from __future__ import annotations

import re
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from core.columns import is_identifier_name, is_text_series, name_tokens

SRM_THRESHOLD = 0.001
_GROUP_TOKENS = frozenset({"deney", "grup", "grubu", "variant", "varyant", "experiment", "test", "ab", "arm",
                           "treatment", "kol", "cohort_ab"})
_CONTROL_PATTERN = re.compile(r"^(a|kontrol|control|eski|old|baseline|mevcut|base)$", re.IGNORECASE)
_SUCCESS_VALUES = ("Evet", "evet", "EVET", "True", "true", "1", "yes", "Yes", "Var", "var", "Başarılı")
_METRIC_TOKENS = frozenset({"dönüşüm", "donusum", "conversion", "converted", "eklendi", "satın", "satin",
                            "purchase", "tıklama", "tiklama", "click", "kayıt", "signup", "abone"})
_UNIT_TOKENS = frozenset({"müşteri", "musteri", "customer", "client", "kullanıcı", "kullanici", "user", "üye", "uye",
                          "visitor", "ziyaretçi", "oturum", "session"})
METHOD_NOTE = (
    "Oran metriğinde satır düzeyinde iki oranlı z-testi, ortalama metriğinde ya da birim düzeyinde Welch t-testi "
    "kullanılır. Güven aralıkları çift yönlüdür; göreli artışın aralığı kontrol ortalamasının belirsizliğini yok "
    "sayan yaklaşık bir aralıktır. SRM ki-kare testiyle sınanır (p < 0,001 alarm). Güç hesabı normal yaklaşımladır. "
    "Testin sonunda tek bir kez bakıldığı varsayılır; ara bakışlar ve çok sayıda metrik yanlış pozitif oranını artırır."
)


class ExperimentSpec(BaseModel):
    group_column: str
    control: str
    treatment: str
    metric_column: str
    metric_kind: Literal["oran", "ortalama"]
    success_value: str | None = None
    unit_column: str | None = None
    expected_control_share: float = Field(0.5, gt=0, lt=1)
    alpha: float = Field(0.05, gt=0, lt=0.5)
    power: float = Field(0.8, gt=0.5, lt=1)
    mde_relative: float = Field(0.10, gt=0)

    @model_validator(mode="after")
    def _check(self) -> ExperimentSpec:
        if self.control == self.treatment:
            raise ValueError("Kontrol ve deney grubu farklı olmalı.")
        if self.metric_kind == "oran" and self.success_value is None:
            raise ValueError("Oran metriği için başarı değeri seçilmeli.")
        if self.metric_column in (self.group_column, self.unit_column):
            raise ValueError("Metrik sütunu grup ya da birim sütunu olamaz.")
        return self

    def metric_label(self) -> str:
        if self.metric_kind == "oran":
            return f"“{self.metric_column} = {self.success_value}” oranı"
        return f"{self.metric_column} ortalaması"


class GroupStats(BaseModel):
    label: str
    units: int
    rows: int
    mean: float
    sd: float
    ci_low: float
    ci_high: float


class SRMCheck(BaseModel):
    control_units: int
    treatment_units: int
    expected_control_share: float
    observed_control_share: float
    p_value: float
    alarm: bool


class PowerAnalysis(BaseModel):
    mde_relative: float
    power_target: float
    required_per_group: int | None
    power_at_mde: float | None
    detectable_relative: float | None


class ExperimentReport(BaseModel):
    spec: ExperimentSpec
    metric_label: str
    analysis_unit: str
    control: GroupStats
    treatment: GroupStats
    difference: float
    ci_low: float
    ci_high: float
    relative_lift: float | None
    relative_ci_low: float | None
    relative_ci_high: float | None
    p_value: float
    test_name: str
    significant: bool
    srm: SRMCheck
    power: PowerAnalysis
    excluded_units: int = 0
    verdict: str
    notes: list[str] = Field(default_factory=list)
    method_note: str = METHOD_NOTE


class ExperimentResult:
    def __init__(self, report: ExperimentReport, figure: go.Figure) -> None:
        self.report, self.figure = report, figure


def _pct(value: float, digits: int = 1) -> str:
    return f"%{value * 100:.{digits}f}".replace(".", ",")


def _num(value: float) -> str:
    digits = 4 if abs(value) < 1 else 2
    return f"{value:,.{digits}f}".rstrip("0").rstrip(".").replace(",", "_").replace(".", ",").replace("_", ".")


def _p(value: float) -> str:
    return "< 0,001" if value < 0.001 else f"{value:.3f}".replace(".", ",")


def detect_experiment(frame: pd.DataFrame) -> ExperimentSpec | None:
    """Grup, metrik ve birim sütunlarını ad ve içerikten önerir; bulunamazsa None (arayüz elle seçtirir)."""
    groups = [
        str(column) for column in frame.columns
        if is_text_series(frame[column]) and 2 <= frame[column].nunique() <= 5
        and set(name_tokens(column)) & _GROUP_TOKENS
    ]
    if not groups:
        return None
    group = groups[0]
    labels = sorted(frame[group].dropna().astype(str).unique())
    control = next((label for label in labels if _CONTROL_PATTERN.match(label)), labels[0])
    treatment = next(label for label in labels if label != control)
    binary = [
        str(column) for column in frame.columns
        if str(column) != group and is_text_series(frame[column]) and frame[column].nunique() == 2
        and not is_identifier_name(column)
    ]
    binary.sort(key=lambda column: not set(name_tokens(column)) & _METRIC_TOKENS)
    metric = next((column for column in binary
                   if set(frame[column].dropna().astype(str)) & set(_SUCCESS_VALUES)), None)
    if metric is None:
        return None
    success = next(value for value in _SUCCESS_VALUES if value in set(frame[metric].dropna().astype(str)))
    unit = next((
        str(column) for column in frame.columns
        if is_text_series(frame[column]) and is_identifier_name(column) and set(name_tokens(column)) & _UNIT_TOKENS
        and frame[column].nunique() < frame[column].notna().sum()
    ), None)
    return ExperimentSpec(group_column=group, control=control, treatment=treatment, metric_column=metric,
                          metric_kind="oran", success_value=success, unit_column=unit)


def _values(frame: pd.DataFrame, spec: ExperimentSpec) -> pd.Series:
    column = frame[spec.metric_column]
    if spec.metric_kind == "oran":
        return (column.astype(str) == spec.success_value).astype(float).where(column.notna())
    if not pd.api.types.is_numeric_dtype(column) or pd.api.types.is_bool_dtype(column):
        raise ValueError(f"“{spec.metric_column}” sayısal değil; ortalama testi yapılamaz.")
    return column.astype(float)


def _group_stats(label: str, values: np.ndarray, rows: int, z: float) -> GroupStats:
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    half = z * sd / np.sqrt(len(values))
    return GroupStats(label=label, units=len(values), rows=rows, mean=mean, sd=sd, ci_low=mean - half,
                      ci_high=mean + half)


def _figure(report: ExperimentReport) -> go.Figure:
    groups = (report.control, report.treatment)
    scale = 100 if report.spec.metric_kind == "oran" else 1
    figure = go.Figure(go.Bar(
        x=[f"{group.label} ({'kontrol' if group is report.control else 'deney'})" for group in groups],
        y=[group.mean * scale for group in groups],
        error_y={"type": "data", "symmetric": False,
                 "array": [(group.ci_high - group.mean) * scale for group in groups],
                 "arrayminus": [(group.mean - group.ci_low) * scale for group in groups]},
        marker_color=["#5b6778", "#1d5fd1"],
        text=[_pct(group.mean) if scale == 100 else _num(group.mean) for group in groups], textposition="outside",
    ))
    figure.update_layout(
        title=f"{report.metric_label} ({_pct(1 - report.spec.alpha, 0)} güven aralığıyla)", separators=",.",
        yaxis_title="%" if scale == 100 else report.spec.metric_column, showlegend=False,
    )
    return figure


class ExperimentAnalyzer:
    def analyze(self, frame: pd.DataFrame, spec: ExperimentSpec) -> ExperimentResult:
        for column in (spec.group_column, spec.metric_column, spec.unit_column):
            if column is not None and column not in frame.columns:
                raise ValueError(f"“{column}” sütunu veride yok.")
        groups = frame[spec.group_column].astype(str).where(frame[spec.group_column].notna())
        data = pd.DataFrame({"group": groups, "value": _values(frame, spec)})
        if spec.unit_column:
            data["unit"] = frame[spec.unit_column].astype(str).where(frame[spec.unit_column].notna())
        data = data[data["group"].isin([spec.control, spec.treatment])]
        notes: list[str] = []
        excluded_units = 0
        unit_label = "kayıt"
        if spec.unit_column:
            data = data[data["unit"].notna()]
            group_count = data.groupby("unit")["group"].nunique()
            mixed = group_count[group_count > 1].index
            excluded_units = len(mixed)
            if excluded_units:
                notes.append(f"{excluded_units} birim her iki grupta da görünüyor (atama tutarsız); analiz dışı bırakıldı.")
                data = data[~data["unit"].isin(mixed)]
            unit_label = spec.unit_column
        for label in (spec.control, spec.treatment):
            if not (data["group"] == label).any():
                raise ValueError(f"“{label}” grubunda kayıt yok.")

        # SRM, metrik eksik olsa da atanmış tüm birimler üzerinden sınanır.
        if spec.unit_column:
            counts = data.groupby("group")["unit"].nunique()
        else:
            counts = data.groupby("group").size()
        control_units, treatment_units = int(counts.get(spec.control, 0)), int(counts.get(spec.treatment, 0))
        total = control_units + treatment_units
        srm_p = float(stats.chisquare(
            [control_units, treatment_units],
            f_exp=[total * spec.expected_control_share, total * (1 - spec.expected_control_share)],
        ).pvalue)
        srm = SRMCheck(control_units=control_units, treatment_units=treatment_units,
                       expected_control_share=spec.expected_control_share,
                       observed_control_share=control_units / total, p_value=srm_p, alarm=srm_p < SRM_THRESHOLD)

        measured = data[data["value"].notna()]
        rows = measured.groupby("group").size()
        if spec.unit_column:
            measured = measured.groupby(["unit", "group"], as_index=False)["value"].mean()
        control_values = measured.loc[measured["group"] == spec.control, "value"].to_numpy(dtype=float)
        treatment_values = measured.loc[measured["group"] == spec.treatment, "value"].to_numpy(dtype=float)
        if min(len(control_values), len(treatment_values)) < 2:
            raise ValueError("Her grupta metriği dolu en az iki birim gerekir.")
        z = float(stats.norm.ppf(1 - spec.alpha / 2))
        control = _group_stats(spec.control, control_values, int(rows.get(spec.control, 0)), z)
        treatment = _group_stats(spec.treatment, treatment_values, int(rows.get(spec.treatment, 0)), z)
        difference = treatment.mean - control.mean
        n1, n2 = len(control_values), len(treatment_values)
        var1, var2 = control.sd ** 2, treatment.sd ** 2

        if spec.metric_kind == "oran" and not spec.unit_column:
            pooled = (control_values.sum() + treatment_values.sum()) / (n1 + n2)
            pooled_se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
            p_value = float(2 * stats.norm.sf(abs(difference) / pooled_se)) if pooled_se > 0 else 1.0
            se = np.sqrt(control.mean * (1 - control.mean) / n1 + treatment.mean * (1 - treatment.mean) / n2)
            critical = z
            test_name = "İki oranlı z-testi"
            var1 = control.mean * (1 - control.mean)
            var2 = treatment.mean * (1 - treatment.mean)
        else:
            se = np.sqrt(var1 / n1 + var2 / n2)
            welch = stats.ttest_ind(treatment_values, control_values, equal_var=False)
            p_value = float(welch.pvalue) if np.isfinite(welch.pvalue) else 1.0
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / max(n1 - 1, 1) + (var2 / n2) ** 2 / max(n2 - 1, 1)
            ) if se > 0 else n1 + n2 - 2
            critical = float(stats.t.ppf(1 - spec.alpha / 2, df))
            test_name = "Welch t-testi" + (f" ({spec.unit_column} düzeyinde)" if spec.unit_column else "")
        ci_low, ci_high = difference - critical * se, difference + critical * se
        relative = difference / control.mean if control.mean else None
        rel_low = ci_low / control.mean if control.mean else None
        rel_high = ci_high / control.mean if control.mean else None
        significant = p_value < spec.alpha

        z_power = float(stats.norm.ppf(spec.power))
        target = spec.mde_relative * control.mean
        if spec.metric_kind == "oran" and not spec.unit_column:
            shifted = min(max(control.mean * (1 + spec.mde_relative), 0.0), 1.0)
            target_variance = control.mean * (1 - control.mean) + shifted * (1 - shifted)
        else:
            target_variance = var1 + var2
        required = int(np.ceil((z + z_power) ** 2 * target_variance / target ** 2)) if target else None
        power_at = float(stats.norm.cdf(abs(target) / se - z)) if se > 0 and target else None
        detectable = (z + z_power) * se / abs(control.mean) if control.mean and se > 0 else None
        power = PowerAnalysis(mde_relative=spec.mde_relative, power_target=spec.power, required_per_group=required,
                              power_at_mde=power_at, detectable_relative=detectable)

        label = spec.metric_label()
        if srm.alarm:
            verdict = (f"Örneklem oranı uyumsuz (SRM): kontrol payı {_pct(srm.observed_control_share)}, beklenen "
                       f"{_pct(srm.expected_control_share)} (p {_p(srm.p_value)}). Atama ya da kayıt bozuk olabilir; "
                       "sonucu yorumlamadan önce nedenini bulun.")
        elif significant:
            direction = "yüksek" if difference > 0 else "düşük"
            verdict = (f"“{spec.treatment}” grubunda {label} kontrolden anlamlı biçimde {direction}: "
                       f"{_fmt(spec, control.mean)} → {_fmt(spec, treatment.mean)}"
                       + (f", göreli fark {_signed_pct(relative)} (güven aralığı {_signed_pct(rel_low)} – "
                          f"{_signed_pct(rel_high)})" if relative is not None else "")
                       + f"; p {_p(p_value)}.")
        else:
            verdict = (f"Gruplar arasında anlamlı fark yok (p {_p(p_value)}; fark {_fmt(spec, difference, signed=True)}, "
                       f"güven aralığı {_fmt(spec, ci_low, signed=True)} – {_fmt(spec, ci_high, signed=True)}).")
            if detectable is not None:
                verdict += (f" Bu örneklemle ancak {_pct(detectable)} ve üzeri göreli farklar "
                            f"{_pct(spec.power, 0)} güçle yakalanabilir")
                verdict += (f"; {_pct(spec.mde_relative, 0)} fark için grup başına {required:,} birim gerekir."
                            .replace(",", ".") if required else ".")
        if spec.unit_column:
            notes.append(f"Analiz birimi “{spec.unit_column}”: metrik önce birim başına ortalandı; aynı birimin "
                         "kayıtları bağımsız sayılmadı.")
        report = ExperimentReport(
            spec=spec, metric_label=label, analysis_unit=unit_label, control=control, treatment=treatment,
            difference=difference, ci_low=ci_low, ci_high=ci_high, relative_lift=relative, relative_ci_low=rel_low,
            relative_ci_high=rel_high, p_value=p_value, test_name=test_name, significant=significant,
            srm=srm, power=power, excluded_units=excluded_units, verdict=verdict, notes=notes,
        )
        return ExperimentResult(report, _figure(report))


def _signed_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return ("+" if value > 0 else "") + _pct(value)


def _fmt(spec: ExperimentSpec, value: float, signed: bool = False) -> str:
    if spec.metric_kind == "oran":
        text = f"{value * 100:.2f}".replace(".", ",") + " puan" if signed else _pct(value, 2)
    else:
        text = _num(value)
    return ("+" if signed and value > 0 else "") + text
