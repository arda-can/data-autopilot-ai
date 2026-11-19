"""Tek tıkla yönetici raporu: serileştirilebilir rapor modeli, HTML ve PDF çıktısı.

Rapor yalnızca toplulaştırılmış bulguları içerir; ham satırlar ve audit log'daki
önce/sonra örnekleri (kişisel veri taşıyabilir) rapora girmez.
"""

from __future__ import annotations

import html
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from pydantic import BaseModel, Field

from core.autopilot import AutopilotResult
from core.health import LOG_METHOD, OutlierFinding
from core.formatting import decimal_tr, integer_tr, localize_figure
from core.timeseries import TimeSeriesSummary, format_period
from core.customers import CustomerSummary
from core.drift import DriftReport
from core.experiment import ExperimentReport
from core.rootcause import RootCauseReport
from core.rules import RuleReport
from core.executor import RejectedAction
from core.explain import PDP_NOTE, ModelExplanation, PartialDependence
from core.planner import PlanOrigin
from core.relations import JoinLog

_PROBLEM_TYPES = {"Classification": "Sınıflandırma", "Regression": "Regresyon"}


# --- Rapor modeli ----------------------------------------------------------------------------


class ReportAction(BaseModel):
    column: str
    action: str
    reason: str
    affected_count: int


class ReportMissing(BaseModel):
    column: str
    ratio: float


class ReportChart(BaseModel):
    title: str
    reason: str
    insight: str
    sample_size: int | None = None


class ReportFeature(BaseModel):
    name: str
    importance: float


class ReportModelSummary(BaseModel):
    target: str
    problem_type: str
    best_model: str
    metric: str
    baseline_score: float
    best_score: float
    beats_baseline: bool
    verdict: str
    leaderboard: list[dict[str, Any]] = Field(default_factory=list)
    leakage_warnings: list[str] = Field(default_factory=list)
    top_features: list[ReportFeature] = Field(default_factory=list)
    split_method: str = ""
    train_rows: int = 0
    test_rows: int = 0
    rows_available: int = 0
    cv_folds: int = 0
    cv_score: float | None = None
    cv_std: float | None = None
    excluded_columns: list[str] = Field(default_factory=list)
    acknowledged_columns: list[str] = Field(default_factory=list)


class ExecutiveReport(BaseModel):
    title: str
    dataset_name: str
    generated_at: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    quality_before: float
    quality_after: float
    missing_ratio_after: float
    headline: list[str]
    recommendations: list[str]
    issues: list[str]
    actions: list[ReportAction]
    missing_columns: list[ReportMissing]
    outliers: list[OutlierFinding]
    charts: list[ReportChart]
    model: ReportModelSummary | None = None
    model_message: str = ""
    timeseries: TimeSeriesSummary | None = None
    timeseries_message: str = ""
    drift: DriftReport | None = None
    rules: RuleReport | None = None
    customers: CustomerSummary | None = None
    root_cause: RootCauseReport | None = None
    experiment: ExperimentReport | None = None
    pdp: list[PartialDependence] = Field(default_factory=list)
    join_logs: list[JoinLog] = Field(default_factory=list)
    plan_origin: PlanOrigin | None = None
    rejected_actions: list[RejectedAction] = Field(default_factory=list)


# --- Türkçe biçimlendirme --------------------------------------------------------------------


def _int(value: int) -> str:
    return integer_tr(value)


def _dec(value: float, digits: int = 1) -> str:
    return decimal_tr(value, digits)


def _num(value: float) -> str:
    """Tam sayıysa binlik ayraçlı, değilse tek ondalıklı: 30000.0 → `30.000`, 12.5 → `12,5`."""
    return _int(int(value)) if float(value).is_integer() else _dec(value)


def _pct(ratio: float) -> str:
    return f"%{_dec(ratio * 100)}"


def _cell(value: Any) -> str:
    if isinstance(value, bool):
        return "Evet" if value else "Hayır"
    if isinstance(value, float):
        return _dec(value, 3)
    if isinstance(value, int):
        return _int(value)
    return str(value)


# --- Raporun kurulması -----------------------------------------------------------------------


def _model_summary(result: AutopilotResult) -> ReportModelSummary | None:
    automl = result.automl_result
    if automl is None or result.suggested_target is None:
        return None
    importance = automl.feature_importance
    top = importance[importance["Importance"] > 0].head(8)
    return ReportModelSummary(
        target=result.suggested_target,
        problem_type=_PROBLEM_TYPES.get(automl.problem_type, automl.problem_type),
        best_model=automl.best_model,
        metric=automl.metric,
        baseline_score=automl.baseline_score,
        best_score=automl.best_score,
        beats_baseline=automl.beats_baseline,
        verdict=automl.verdict,
        leakage_warnings=list(automl.leakage_warnings),
        split_method=automl.split_method,
        train_rows=automl.train_rows,
        test_rows=automl.test_rows,
        rows_available=int(result.cleaned_frame[result.suggested_target].notna().sum())
        if result.suggested_target in result.cleaned_frame.columns else 0,
        cv_folds=automl.cv_folds,
        cv_score=None if automl.cv_score != automl.cv_score else float(automl.cv_score),
        cv_std=None if automl.cv_std != automl.cv_std else float(automl.cv_std),
        excluded_columns=list(automl.excluded_columns),
        acknowledged_columns=list(automl.acknowledged_columns),
        leaderboard=[
            {key: (None if isinstance(value, float) and value != value else
                   float(value) if isinstance(value, float) else value) for key, value in row.items()}
            for row in automl.leaderboard.round(4).to_dict("records")
        ],
        top_features=[
            ReportFeature(name=str(row.Feature), importance=round(float(row.Importance), 4))
            for row in top.itertuples(index=False)
        ],
    )


def build_report(
    result: AutopilotResult,
    dataset_name: str,
    generated_at: datetime | None = None,
    *,
    drift: DriftReport | None = None,
    explanation: ModelExplanation | None = None,
    join_logs: list[JoinLog] | None = None,
    rules: RuleReport | None = None,
    customers: CustomerSummary | None = None,
    root_cause: RootCauseReport | None = None,
    experiment: ExperimentReport | None = None,
) -> ExecutiveReport:
    """Otopilot sonucundan JSON'a çevrilebilir, sunuma hazır yönetici raporu üretir."""
    before, after = result.before_report, result.after_report
    model = _model_summary(result)
    missing_columns = sorted(
        (ReportMissing(column=item.sütun, ratio=item.eksik_oranı) for item in after.sütun_raporları if item.eksik_oranı > 0),
        key=lambda item: item.ratio,
        reverse=True,
    )
    outliers = sorted(result.outliers, key=lambda item: item.ratio, reverse=True)
    no_issue = ["Belirgin bir kalite sorunu tespit edilmedi."]

    headline = [
        f"{_int(before.satır_sayısı)} satır ve {before.sütun_sayısı} sütunluk veri incelendi; "
        f"temizlik sonrası {_int(after.satır_sayısı)} satır ve {after.sütun_sayısı} sütun kaldı.",
        f"Yapısal hijyen skoru {_dec(before.kalite_skoru)} → {_dec(after.kalite_skoru)} (100 üzerinden; "
        "boşluk, tekrar ve tip sorunlarını ölçer, değerlerin doğruluğunu ölçmez).",
        f"{len(result.audit_logs)} otomatik işlem uygulandı; tamamı kayıt altında."
        if result.audit_logs else "Veride yapısal bir düzeltme gerekmedi.",
    ]
    if after.eksik_veri_oranı > 0:
        headline.append(
            f"Temiz verideki eksik hücre oranı {_pct(after.eksik_veri_oranı)}; bu değerler bilinçli olarak doldurulmadı."
        )
    if outliers:
        headline.append(
            f"{len(outliers)} sütunda şüpheli uç değer bulundu (en belirgini: {outliers[0].column}); "
            "değerler değiştirilmedi, yalnızca raporlandı."
        )
    if rules is not None and rules.compliance_score is not None:
        headline.append(
            f"İş kuralı uyumu %{_dec(rules.compliance_score)}: {len(rules.applied)} kuraldan en az birini ihlal eden "
            f"{_int(rules.violating_rows)} kayıt var (değerler değiştirilmedi)."
        )
    if customers is not None and customers.segments:
        top = max(customers.segments, key=lambda item: item.revenue_share)
        headline.append(
            f"{_int(customers.customers)} müşterinin {_pct(customers.repeat_customer_share)}'i birden fazla işlem yaptı; "
            f"gelirin en büyük payı ({_pct(top.revenue_share)}) “{top.segment}” segmentinde."
        )
    timeseries = result.timeseries.summary if result.timeseries is not None else None
    if timeseries is not None:
        headline.append(timeseries.trend_text)
    if drift is not None:
        headline.append(
            f"Dönem karşılaştırması ({drift.reference_label} → {drift.current_label}): "
            f"{drift.counts['kırmızı']} kırmızı, {drift.counts['sarı']} sarı, {drift.counts['yeşil']} yeşil sütun."
        )
    if root_cause is not None:
        headline.append(root_cause.headline)
    if experiment is not None:
        headline.append(f"A/B testi ({experiment.metric_label}): {experiment.verdict}")
    headline.append(model.verdict if model else (result.automl_message or "Tahmin modeli eğitilmedi."))
    if model and model.leakage_warnings:
        headline.append(
            f"Model sonucu güvenilir olmayabilir: {len(model.leakage_warnings)} özellikte olası hedef sızıntısı var "
            "(ayrıntı önerilerde)."
        )

    recommendations: list[str] = []
    for finding in after.domain_findings:
        recommendations.append(
            f"“{finding.column}” sütununda {_int(finding.count)} negatif değer şüpheli; "
            "alan kuralı gereği negatif olmamalı. Kaynak kayıtlar incelenmeli; değerler değiştirilmedi."
        )
    for finding in after.category_variants:
        recommendations.append(
            f"“{finding.column}” sütununda aksan veya büyük/küçük harf farkıyla oluşmuş "
            f"{_int(finding.group_count)} olası kategori kopyası grubu ({_int(finding.affected_count)} kayıt) var. "
            "Birleştirme öncesinde incelenmesi önerilir; değerler değiştirilmedi."
        )
    if rules is not None:
        broken = sorted((item for item in rules.applied if item.violations), key=lambda item: -item.violations)
        for item in broken[:3]:
            recommendations.append(
                f"Kural ihlali: {item.description} — {_int(item.violations)} kayıt uymuyor. "
                "İhlal eden kayıtlar uygulamadan indirilip kaynağında düzeltilmeli."
            )
    if customers is not None:
        for name in ("Kaybedilmemeli", "Risk altında"):
            segment = next((item for item in customers.segments if item.segment == name), None)
            if segment is not None and segment.revenue_share >= 0.05:
                recommendations.append(
                    f"“{name}” segmentinde {_int(segment.customers)} müşteri gelirin {_pct(segment.revenue_share)}'ini "
                    f"oluşturuyor (ortalama {_dec(segment.recency_days, 0)} gündür işlem yok). {segment.action}"
                )
    if before.mükerrer_satır_sayısı:
        recommendations.append(
            f"{_int(before.mükerrer_satır_sayısı)} mükerrer kayıt bulundu ve kaldırıldı; "
            "veri girişinde tekilleştirme kontrolü önerilir."
        )
    for item in [entry for entry in missing_columns if entry.ratio >= 0.20][:3]:
        recommendations.append(
            f"“{item.column}” sütununda eksik oranı {_pct(item.ratio)}; kaynağında tamamlanması analiz ve model kalitesini artırır."
        )
    for finding in outliers[:3]:
        extremes = []
        if finding.max > finding.upper_bound:
            extremes.append(f"en büyük değer {_num(finding.max)} (beklenen üst sınır {_num(finding.upper_bound)})")
        if finding.min < finding.lower_bound:
            extremes.append(f"en küçük değer {_num(finding.min)} (beklenen alt sınır {_num(finding.lower_bound)})")
        advice = (
            "Dağılım çarpık olabilir; bu değerler gerçek olabilir, modelleme öncesinde incelenmeli."
            if finding.ratio > 0.05 else "Veri girişi hatası olup olmadığı kontrol edilmeli."
        )
        if finding.method == LOG_METHOD:
            advice += " Dağılım çok çarpık olduğu için sınırlar log ölçekte hesaplandı."

        recommendations.append(
            f"“{finding.column}” sütununda {_int(finding.count)} şüpheli uç değer var: {' ve '.join(extremes)}. {advice}"
        )
    if timeseries is not None and timeseries.anomalies:
        details = "; ".join(
            f"{item.label} ({item.direction}: gözlenen {_num(item.observed)}, beklenen {_num(item.expected)})"
            for item in timeseries.anomalies[:3]
        )
        recommendations.append(
            f"{timeseries.value_label} serisinde olağan dışı dönemler: {details}. "
            "Kampanya, tatil veya veri hatası olup olmadığı doğrulanmalı."
        )
    if drift is not None and drift.counts.get("kırmızı"):
        worst = next(item for item in drift.columns if item.status == "kırmızı")
        recommendations.append(
            f"Dönem karşılaştırmasında ({drift.reference_label} → {drift.current_label}) "
            f"{drift.counts['kırmızı']} sütun kırmızı; en büyük değişim “{worst.column}”: {worst.explanation} "
            "Modeller ve raporlar bu değişimi hesaba katmalı."
        )
    if timeseries is not None and not timeseries.beats_baseline:
        recommendations.append(
            "Zaman serisi tahmini naif yöntemi anlamlı biçimde geçemedi; planlamada yalnızca kaba yön göstergesi "
            "olarak kullanılmalı."
        )
    if model and model.leakage_warnings:
        recommendations.extend(model.leakage_warnings)
    elif model and model.beats_baseline and model.top_features:
        names = ", ".join(feature.name for feature in model.top_features[:3])
        recommendations.append(f"“{model.target}” üzerinde en etkili değişkenler: {names}. Bu değişkenler öncelikli izlenmeli.")
    elif model:
        recommendations.append(
            f"“{model.target}” için model taban çizgisini geçemedi; ek açıklayıcı veri toplanmadan tahminler karar için kullanılmamalı."
        )
    if not recommendations:
        recommendations.append("Belirgin bir aksiyon gerektiren bulgu yok.")

    return ExecutiveReport(
        title=f"Yönetici Raporu — {dataset_name}",
        dataset_name=dataset_name,
        generated_at=(generated_at or datetime.now()).strftime("%d.%m.%Y %H:%M"),
        rows_before=before.satır_sayısı,
        rows_after=after.satır_sayısı,
        columns_before=before.sütun_sayısı,
        columns_after=after.sütun_sayısı,
        quality_before=before.kalite_skoru,
        quality_after=after.kalite_skoru,
        missing_ratio_after=after.eksik_veri_oranı,
        headline=headline,
        recommendations=recommendations,
        issues=[] if before.sorunlar == no_issue else list(before.sorunlar),
        actions=[
            ReportAction(column=log.column, action=log.action, reason=log.reason, affected_count=log.affected_count)
            for log in result.audit_logs
        ],
        missing_columns=missing_columns,
        outliers=outliers,
        charts=[
            ReportChart(title=chart.title, reason=chart.reason, insight=chart.insight, sample_size=chart.sample_size)
            for chart in result.charts
        ],
        model=model,
        model_message=result.automl_message,
        timeseries=timeseries,
        timeseries_message=result.timeseries_message,
        drift=drift,
        rules=rules,
        customers=customers,
        root_cause=root_cause,
        experiment=experiment,
        pdp=list(explanation.curves) if explanation is not None else [],
        join_logs=list(join_logs or []),
        plan_origin=result.plan_origin,
        rejected_actions=list(result.rejected_actions),
    )


def _evaluation_text(model: ReportModelSummary) -> str:
    """Modelin nasıl değerlendirildiği: bölme, satır sayıları, çapraz doğrulama, örneklem ve tahmin anı."""
    if not model.train_rows:
        return ""
    parts = [f"Değerlendirme: {model.split_method}",
             f"eğitim {_int(model.train_rows)} / test {_int(model.test_rows)} satır"]
    if model.cv_folds and model.cv_score is not None:
        parts.append(
            f"model {model.cv_folds} katlı çapraz doğrulamayla seçildi (CV {_dec(model.cv_score, 3)} ± "
            f"{_dec(model.cv_std or 0.0, 3)})"
        )
    text = " · ".join(parts) + "."
    used = model.train_rows + model.test_rows
    if model.rows_available > used:
        text += f" Model, {_int(model.rows_available)} satırın {_int(used)} satırlık rastgele örneklemiyle eğitildi."
    if model.excluded_columns:
        text += f" Tahmin anında bilinmediği için çıkarılan sütunlar: {', '.join(model.excluded_columns)}."
    if model.acknowledged_columns:
        text += f" Tahmin anında bilindiği onaylanan sütunlar: {', '.join(model.acknowledged_columns)}."
    return text


def _plan_text(report: ExecutiveReport) -> str:
    if report.plan_origin is None:
        return ""
    return f"Temizlik planı: {report.plan_origin.source} — {report.plan_origin.reason}"


def _timeseries_facts(series: TimeSeriesSummary) -> list[tuple[str, str]]:
    period = (
        f"{format_period(series.start, series.frequency)} – {format_period(series.end, series.frequency)} "
        f"({_int(series.periods)} {series.period_unit})"
    )
    return [
        ("Tarih sütunu", series.date_column), ("Ölçü", series.value_label),
        ("Frekans", series.frequency_label), ("Dönem", period), ("Model", series.model_name),
    ]


def _customer_facts(customers: CustomerSummary) -> list[str]:
    facts = [
        f"{_int(customers.customers)} müşteri, {_int(customers.transactions)} işlem; tekrar eden müşteri payı "
        f"{_pct(customers.repeat_customer_share)}. Sütunlar: müşteri “{customers.columns.customer}”, tarih "
        f"“{customers.columns.date}”, tutar “{customers.columns.amount}”.",
        f"RFM penceresi: {customers.window_label} ({customers.window_start} – {customers.analysis_date}).",
    ]
    if customers.retention:
        facts.append("Ortalama tutundurma: " + ", ".join(
            f"{month}. ay {_pct(value)}" for month, value in customers.retention.items()) + ".")
    return facts


def _kpis(report: ExecutiveReport) -> list[tuple[str, str, str]]:
    return [
        ("Yapısal Hijyen", f"{_dec(report.quality_after)}/100", f"önce {_dec(report.quality_before)}"),
        *([("İş Kuralı Uyumu", f"%{_dec(report.rules.compliance_score)}",
            f"{_int(report.rules.violating_rows)} kayıt ihlalli")]
          if report.rules is not None and report.rules.compliance_score is not None else []),
        ("Satır", _int(report.rows_after), f"önce {_int(report.rows_before)}"),
        ("Sütun", _int(report.columns_after), f"önce {_int(report.columns_before)}"),
        ("Otomatik İşlem", _int(len(report.actions)), "tamamı kayıtlı"),
    ]


def _tables(report: ExecutiveReport) -> dict[str, tuple[list[str], list[list[str]]]]:
    """HTML ve PDF'in ortak kullandığı tablo içerikleri (başlıklar, satırlar)."""
    tables: dict[str, tuple[list[str], list[list[str]]]] = {}
    if report.missing_columns:
        tables["missing"] = (
            ["Sütun", "Eksik Oranı"],
            [[item.column, _pct(item.ratio)] for item in report.missing_columns[:15]],
        )
    if report.outliers:
        tables["outliers"] = (
            ["Sütun", "Aykırı Sayısı", "Oran", "Beklenen Aralık", "En Küçük", "En Büyük", "Yöntem"],
            [[
                item.column, _int(item.count), _pct(item.ratio),
                f"{_num(item.lower_bound)} – {_num(item.upper_bound)}", _num(item.min), _num(item.max), item.method,
            ] for item in report.outliers[:15]],
        )
    if report.actions:
        tables["actions"] = (
            ["Sütun", "İşlem", "Gerekçe", "Etkilenen"],
            [[item.column, item.action, item.reason, _int(item.affected_count)] for item in report.actions[:40]],
        )
    if report.timeseries and report.timeseries.anomalies:
        tables["anomalies"] = (
            ["Dönem", "Gözlenen", "Beklenen", "Sapma (z)", "Yön"],
            [[item.label, _num(item.observed), _num(round(item.expected, 1)), _dec(item.z), item.direction]
             for item in report.timeseries.anomalies],
        )
    if report.rules is not None and report.rules.results:
        tables["rules"] = (
            ["Kural", "Kaynak", "Kontrol", "İhlal", "Uyum"],
            [[item.description, item.rule.source, _int(item.checked), _int(item.violations),
              _pct(item.compliance) if item.compliance is not None else (item.error or "—")]
             for item in report.rules.results[:30]],
        )
    if report.customers is not None:
        tables["segments"] = (
            ["Segment", "Müşteri", "Müşteri payı", "Gelir payı", "Ort. yakınlık (gün)", "Ort. sıklık"],
            [[item.segment, _int(item.customers), _pct(item.customer_share), _pct(item.revenue_share),
              _dec(item.recency_days, 0), _dec(item.frequency, 2)] for item in report.customers.segments],
        )
        tables["clv"] = (
            ["Edinimden sonraki ay", "Gözlenen müşteri", "Müşteri başına gelir", "Müşteri başına işlem"],
            [[str(row.horizon_months), _int(row.customers),
              "—" if row.revenue_per_customer is None else _num(round(row.revenue_per_customer, 2)),
              "—" if row.orders_per_customer is None else _dec(row.orders_per_customer, 2)]
             for row in report.customers.clv],
        )
    if report.root_cause is not None and report.root_cause.dimensions:
        cause = report.root_cause
        tables["root_dimensions"] = (
            ["Boyut", "Beklenenden sapma", "Özet"],
            [[item.dimension, _dec(item.score, 2), item.summary] for item in cause.dimensions[:8]],
        )
        tables["root_groups"] = (
            ["Grup", "Referans", "Güncel", "Katkı", "Değişimdeki pay", "Beklenen pay"],
            [[item.group,
              "—" if item.reference_value is None else _num(round(item.reference_value, 3)),
              "—" if item.current_value is None else _num(round(item.current_value, 3)),
              _num(round(item.contribution, 3)), _pct(item.share_of_change),
              _pct(item.expected / cause.change) if cause.change else "—"]
             for item in cause.dimensions[0].groups[:10]],
        )
    if report.experiment is not None:
        test = report.experiment
        scale, unit = (100, " puan") if test.spec.metric_kind == "oran" else (1, "")
        tables["experiment"] = (
            ["Grup", "Birim", "Ortalama", "Güven aralığı"],
            [[f"{group.label} ({role})", _int(group.units), _num(round(group.mean * scale, 3)),
              f"{_num(round(group.ci_low * scale, 3))} – {_num(round(group.ci_high * scale, 3))}"]
             for group, role in ((test.control, "kontrol"), (test.treatment, "deney"))],
        )
        tables["experiment_facts"] = (
            ["Ölçü", "Değer"],
            [
                ["Fark (deney − kontrol)", f"{_num(round(test.difference * scale, 3))}{unit} "
                                          f"(GA {_num(round(test.ci_low * scale, 3))} – {_num(round(test.ci_high * scale, 3))})"],
                ["Göreli fark", "—" if test.relative_lift is None else _pct(test.relative_lift)],
                ["p-değeri", "< 0,001" if test.p_value < 0.001 else _dec(test.p_value, 3)],
                ["Test", f"{test.test_name}; analiz birimi: {test.analysis_unit}"],
                ["SRM", f"kontrol payı {_pct(test.srm.observed_control_share)} (beklenen "
                        f"{_pct(test.srm.expected_control_share)}), p = {_dec(test.srm.p_value, 4)}"
                        + (" — ALARM" if test.srm.alarm else "")],
                ["Yakalanabilir en küçük fark", "—" if test.power.detectable_relative is None
                 else f"{_pct(test.power.detectable_relative)} (güç {_pct(test.power.power_target)})"],
                [f"{_pct(test.power.mde_relative)} fark için grup başına örneklem",
                 "—" if test.power.required_per_group is None else _int(test.power.required_per_group)],
            ],
        )
    if report.drift and report.drift.columns:
        changed = [item for item in report.drift.columns if item.status != "yeşil"] or report.drift.columns[:5]
        tables["drift"] = (
            ["Durum", "Sütun", "Tür", "PSI", "Açıklama"],
            [[item.status, item.column, item.kind, _dec(item.psi, 3), item.explanation] for item in changed[:15]],
        )
    if report.join_logs:
        tables["joins"] = (
            ["Tablo", "Tür", "Anahtar", "Satır", "Eşleşme"],
            [[log.table, log.mode, log.key, _int(log.rows_after), _pct(log.match_rate)] for log in report.join_logs],
        )
    if report.rejected_actions:
        tables["rejected"] = (
            ["Sütun", "Öneri", "Neden uygulanmadı"],
            [[item.column, item.action, item.rejection] for item in report.rejected_actions],
        )
    if report.model and report.model.leaderboard:
        headers = list(report.model.leaderboard[0])
        tables["leaderboard"] = (headers, [[_cell(row[key]) for key in headers] for row in report.model.leaderboard])
    return tables


# --- HTML ------------------------------------------------------------------------------------

_CSS = """
:root{--ink:#16202e;--muted:#5b6778;--line:#e3e8ef;--bg:#f5f7fa;--card:#fff;--accent:#1d5fd1;--good:#0f7b4f;--warn:#a05a00}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 "Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:40px 20px 64px}
header{margin-bottom:28px}
.eyebrow{margin:0;color:var(--accent);font-size:13px;font-weight:600;letter-spacing:.08em;text-transform:uppercase}
h1{margin:6px 0 4px;font-size:32px;line-height:1.2;text-wrap:balance}
h2{margin:0 0 14px;font-size:20px}
h3{margin:0 0 8px;font-size:16px}
.meta,.note{margin:0;color:var(--muted);font-size:14px}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:24px;margin-bottom:20px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;background:none;border:0;padding:0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.kpi span{display:block;color:var(--muted);font-size:13px}
.kpi strong{display:block;font-size:26px;line-height:1.3;font-variant-numeric:tabular-nums}
ul,ol{margin:0;padding-left:22px}
li{margin-bottom:8px}
.verdict{border-left:4px solid var(--warn);background:#fff7e8;padding:12px 16px;border-radius:6px;margin:0 0 16px}
.verdict.good{border-color:var(--good);background:#eaf7f0}
.facts{display:flex;flex-wrap:wrap;gap:8px 28px;margin:0 0 16px;font-size:15px}
.facts b{color:var(--muted);font-weight:600}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.charts{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,440px),1fr));gap:20px;align-items:start}
.chart{border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0;overflow:hidden}
.chart p{margin:6px 0 0;font-size:14px}
footer{color:var(--muted);font-size:13px;text-align:center;margin-top:28px}
@media print{body{background:#fff}main{padding:0}section,.kpi,.chart{break-inside:avoid}}
"""


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _html_list(items: Sequence[str], ordered: bool = False) -> str:
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + f"</{tag}>"


def render_html(
    report: ExecutiveReport,
    figures: Sequence[Any] = (),
    model_figure: Any | None = None,
    embed_plotly: bool = True,
    timeseries_figures: Sequence[Any] = (),
    pdp_figures: Sequence[Any] = (),
) -> str:
    """Kendi kendine yeten tek dosyalık HTML raporu.

    `figures`, `report.charts` ile aynı sıradaki Plotly figürleridir. `embed_plotly=True`
    iken plotly.js dosyaya gömülür (çevrimdışı açılır, ~5 MB); `False` iken CDN'den yüklenir.
    `timeseries_figures`: zaman serisi tahmin ve mevsimsellik grafikleri (varsa).
    """
    plotly_loaded = False

    def plot(figure: Any) -> str:
        nonlocal plotly_loaded
        include: bool | str = False if plotly_loaded else (True if embed_plotly else "cdn")
        plotly_loaded = True
        # Sabit yükseklik şart: ızgara hücresinde %100 yükseklik sıfıra çöker ve grafikler üst üste biner.
        # Izgara `auto-fill` olmalı: `auto-fit` ilk grafiği tam genişlikte çizdirir, sonra kart dışına taşar.
        return localize_figure(figure).to_html(
            full_html=False, include_plotlyjs=include, default_height="420px",
            config={"displaylogo": False, "responsive": True},
        )

    escape = html.escape
    tables = _tables(report)
    parts = [
        '<!doctype html><html lang="tr"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(report.title)}</title><style>{_CSS}</style></head><body><main>",
        '<header><p class="eyebrow">Data Autopilot AI · Yönetici Raporu</p>',
        f'<h1>{escape(report.dataset_name)}</h1><p class="meta">Oluşturulma: {escape(report.generated_at)}</p></header>',
        '<section class="kpis">',
        *(
            f'<div class="kpi"><span>{escape(label)}</span><strong>{escape(value)}</strong><span>{escape(note)}</span></div>'
            for label, value, note in _kpis(report)
        ),
        "</section>",
        f"<section><h2>Yönetici Özeti</h2>{_html_list(report.headline)}</section>",
        f"<section><h2>Öneriler</h2>{_html_list(report.recommendations, ordered=True)}</section>",
    ]

    if report.timeseries:
        series = report.timeseries
        parts.append("<section><h2>Zaman Serisi</h2>")
        parts.append(f'<p class="verdict{" good" if series.beats_baseline else ""}">{escape(series.verdict)}</p>')
        parts.append(
            '<p class="facts">' + "".join(f"<span><b>{escape(label)}:</b> {escape(value)}</span>"
                                          for label, value in _timeseries_facts(series)) + "</p>"
        )
        parts.append(_html_list([series.trend_text, series.seasonality_text, series.forecast_text]))
        for figure in timeseries_figures:
            parts.append(f'<div class="chart" style="margin-top:16px">{plot(figure)}</div>')
        if "anomalies" in tables:
            parts.append('<h3 style="margin-top:20px">Olağan dışı dönemler</h3>' + _html_table(*tables["anomalies"]))
        parts.extend(f'<p class="note" style="margin-top:8px">Not: {escape(item)}</p>' for item in series.notes)
        parts.append("</section>")
    elif report.timeseries_message:
        parts.append(f"<section><h2>Zaman Serisi</h2><p>{escape(report.timeseries_message)}</p></section>")

    if "rules" in tables:
        parts.append("<section><h2>İş Kuralı Uyumu</h2>")
        parts.append(_html_table(*tables["rules"]))
        parts.append(f'<p class="note" style="margin-top:8px">{escape(report.rules.method_note)}</p></section>')

    if report.root_cause is not None:
        cause = report.root_cause
        parts.append("<section><h2>Neden Değişti?</h2>")
        parts.append(f"<p>{escape(cause.headline)}</p>")
        if "root_dimensions" in tables:
            parts.append(_html_table(*tables["root_dimensions"]))
            parts.append(f'<h3 style="margin-top:16px">“{escape(cause.dimensions[0].dimension)}” grupları</h3>'
                         + _html_table(*tables["root_groups"]))
        parts.extend(f'<p class="note" style="margin-top:8px">{escape(item)}</p>'
                     for item in [*cause.skipped, cause.method_note])
        parts.append("</section>")

    if report.experiment is not None:
        test = report.experiment
        verdict_class = "verdict" if test.srm.alarm or not test.significant else "verdict good"
        parts.append("<section><h2>A/B Testi</h2>")
        parts.append(f'<p class="{verdict_class}">{escape(test.verdict)}</p>')
        parts.append(_html_table(*tables["experiment"]))
        parts.append('<div style="margin-top:12px">' + _html_table(*tables["experiment_facts"]) + "</div>")
        parts.extend(f'<p class="note" style="margin-top:8px">{escape(item)}</p>'
                     for item in [*test.notes, test.method_note])
        parts.append("</section>")

    if report.customers is not None:
        parts.append("<section><h2>Müşteri Analizi</h2>")
        parts.append(_html_list(_customer_facts(report.customers)))
        parts.append('<h3 style="margin-top:16px">RFM segmentleri</h3>' + _html_table(*tables["segments"]))
        parts.append('<h3 style="margin-top:16px">Tarihsel müşteri değeri</h3>' + _html_table(*tables["clv"]))
        parts.extend(f'<p class="note" style="margin-top:8px">{escape(item)}</p>'
                     for item in [*report.customers.definitions.values(), *report.customers.notes])
        parts.append("</section>")

    if report.drift:
        drift = report.drift
        parts.append("<section><h2>Dönem Karşılaştırması</h2>")
        parts.append(f"<p>{escape(drift.summary)}</p>")
        parts.extend(
            f'<p class="verdict">Şema: “{escape(item.column)}” {escape(item.change)} ({escape(item.detail)})</p>'
            for item in drift.schema_changes
        )
        if "drift" in tables:
            parts.append(_html_table(*tables["drift"]))
        parts.append(f'<p class="note" style="margin-top:8px">{escape(drift.method_note)}</p></section>')

    if report.charts and figures:
        parts.append('<section><h2>Öne Çıkan Grafikler</h2><div class="charts">')
        for chart, figure in zip(report.charts, figures):
            parts.append(
                f'<article class="chart"><h3>{escape(chart.title)}</h3>{plot(figure)}'
                f"<p><strong>Neden seçildi:</strong> {escape(chart.reason)}</p>"
                f"<p><strong>Bulgu:</strong> {escape(chart.insight)}</p>"
                + (f'<p class="note">{escape(_sample_note(chart.sample_size))}</p>' if chart.sample_size else "")
                + "</article>"
            )
        parts.append("</div></section>")

    parts.append("<section><h2>Tahmin Modeli</h2>")
    if report.model:
        model = report.model
        trusted = model.beats_baseline and not model.leakage_warnings
        parts.append(f'<p class="verdict{" good" if trusted else ""}">{escape(model.verdict)}</p>')
        parts.extend(f'<p class="verdict">{escape(item)}</p>' for item in model.leakage_warnings)
        parts.append(
            '<p class="facts">'
            f"<span><b>Hedef:</b> {escape(model.target)}</span>"
            f"<span><b>Problem tipi:</b> {escape(model.problem_type)}</span>"
            f"<span><b>En iyi model:</b> {escape(model.best_model)}</span></p>"
        )
        if "leaderboard" in tables:
            parts.append(_html_table(*tables["leaderboard"]))
        if model_figure is not None:
            parts.append(f'<div class="chart" style="margin-top:16px">{plot(model_figure)}</div>')
        if _evaluation_text(model):
            parts.append(f'<p class="note">{escape(_evaluation_text(model))}</p>')
        parts.append(
            '<p class="note">Eksik doldurma ve ölçekleme yalnızca eğitim verisinden öğrenildi; '
            "skorlar modelin hiç görmediği test bölmesinde ölçüldü.</p>"
        )
    else:
        parts.append(f"<p>{escape(report.model_message or 'Tahmin modeli eğitilmedi.')}</p>")
    parts.append("</section>")

    if report.pdp:
        parts.append(f'<section><h2>Model Neden Böyle Tahmin Ediyor?</h2><p class="note">{escape(PDP_NOTE)}</p>')
        for index, curve in enumerate(report.pdp):
            if index < len(pdp_figures):
                parts.append(f'<div class="chart" style="margin-top:16px">{plot(pdp_figures[index])}</div>')
            css = "verdict" if curve.interpretation.startswith("Uyarı") else "note"
            parts.append(f'<p class="{css}" style="margin-top:8px">{escape(curve.interpretation)}</p>')
        parts.append("</section>")

    parts.append("<section><h2>Veri Kalitesi</h2>")
    parts.append(_html_list(report.issues) if report.issues else "<p>Başlangıçta belirgin bir kalite sorunu tespit edilmedi.</p>")
    if "missing" in tables:
        parts.append("<h3 style=\"margin-top:20px\">Temiz veride eksik değer içeren sütunlar</h3>" + _html_table(*tables["missing"]))
    if "outliers" in tables:
        parts.append("<h3 style=\"margin-top:20px\">Şüpheli uç değerler (değiştirilmedi)</h3>" + _html_table(*tables["outliers"]))
        if any(item.method == LOG_METHOD for item in report.outliers):
            parts.append(f'<p class="note" style="margin-top:8px">{escape(_LOG_NOTE)}</p>')
    parts.append("</section>")

    if "joins" in tables:
        parts.append(
            "<section><h2>Veri Birleştirme</h2>" + _html_table(*tables["joins"])
            + '<p class="note" style="margin-top:8px">Birleştirme satır sayısını değiştirmez; eşleşmeyen '
            "satırlarda eklenen sütunlar boş kalır.</p></section>"
        )
    if "actions" in tables or report.plan_origin or "rejected" in tables:
        parts.append("<section><h2>Uygulanan İşlemler</h2>")
        if _plan_text(report):
            parts.append(f'<p class="note">{escape(_plan_text(report))}</p>')
        if "actions" in tables:
            parts.append(_html_table(*tables["actions"]))
        if "rejected" in tables:
            parts.append('<h3 style="margin-top:20px">Uygulanmayan plan önerileri (veri koruma)</h3>'
                         + _html_table(*tables["rejected"]))
        parts.append("</section>")

    parts.append(
        "<footer>Bu rapor Data Autopilot AI tarafından deterministik kurallarla üretildi. "
        "Ham satırlar ve kişisel veriler rapora dahil edilmez.</footer></main></body></html>"
    )
    return "".join(parts)


# --- PDF -------------------------------------------------------------------------------------

_LOG_NOTE = (
    "IQR (log ölçek): çarpıklığı 2'den büyük ve negatif değer içermeyen sütunlarda beklenen aralık "
    "log ölçekte hesaplanır; doğrusal ölçek bu sütunlarda uzun kuyruğun tamamını aykırı sayar."
)
def _sample_note(size: int) -> str:
    return f"Not: grafik, verinin tamamı yerine {_int(size)} satırlık rastgele örneklemle çizildi."


_PDF_CHART_WIDTH = 0.70  # içerik genişliğine oran; A4'te üç grafik + başlık + not bir sayfaya sığar
_BULLET_GAP = 14  # punto; bir gövde satırı kadar. 11 ve altında pdftotext maddeleri hâlâ birleştiriyor (ölçüldü).

_TURKISH_PROBE ="ğşıİĞŞöüçÖÜÇ"
_ASCII_FALLBACK = str.maketrans({
    "ğ": "g", "Ğ": "G", "ş": "s", "Ş": "S", "ı": "i", "İ": "I",
    "→": "->", "–": "-", "—": "-", "•": "-", "“": '"', "”": '"', "’": "'", "·": "-",
})


def _font_candidates() -> list[tuple[Path, Path]]:
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    linux = Path("/usr/share/fonts/truetype")
    return [
        (windows / "arial.ttf", windows / "arialbd.ttf"),
        (windows / "segoeui.ttf", windows / "segoeuib.ttf"),
        (linux / "dejavu/DejaVuSans.ttf", linux / "dejavu/DejaVuSans-Bold.ttf"),
        (linux / "liberation/LiberationSans-Regular.ttf", linux / "liberation/LiberationSans-Bold.ttf"),
        (linux / "liberation2/LiberationSans-Regular.ttf", linux / "liberation2/LiberationSans-Bold.ttf"),
        (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
    ]


def _ascii_safe(text: str) -> str:
    """Türkçe glifli font bulunamazsa: yerleşik PDF fontunun basabildiği karakterlere indirger."""
    return text.translate(_ASCII_FALLBACK).encode("latin-1", "replace").decode("latin-1")


@lru_cache(maxsize=1)
def _pdf_fonts() -> tuple[str, str, Callable[[str], str]]:
    """Türkçe karakterleri basabilen bir sistem fontu kaydeder; yoksa Helvetica'ya düşer."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFError, TTFont

    for regular_path, bold_path in _font_candidates():
        if not regular_path.is_file():
            continue
        try:
            regular = TTFont("AutopilotSans", str(regular_path))
            glyphs = regular.face.charToGlyph
            if not all(ord(character) in glyphs for character in _TURKISH_PROBE):
                continue
            pdfmetrics.registerFont(regular)
            bold_name = "AutopilotSans"
            if bold_path.is_file():
                pdfmetrics.registerFont(TTFont("AutopilotSans-Bold", str(bold_path)))
                bold_name = "AutopilotSans-Bold"
        except (TTFError, OSError):
            continue

        def safe(text: str, glyphs: dict = glyphs) -> str:
            return "".join(
                character if ord(character) in glyphs else _ascii_safe(character) for character in text
            )

        return "AutopilotSans", bold_name, safe
    return "Helvetica", "Helvetica-Bold", _ascii_safe


def render_pdf(
    report: ExecutiveReport, figures: Sequence[Any] = (), timeseries_figures: Sequence[Any] = (),
    pdp_figures: Sequence[Any] = (),
) -> bytes:
    """A4 PDF; rapor modelinden ayrı alınan grafikler en fazla altı statik görsele çevrilir."""
    from core.chart_export import can_export, chart_png
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.shapes import Drawing
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    regular, bold, safe = _pdf_fonts()
    ink, muted, line, accent = (colors.HexColor(code) for code in ("#16202e", "#5b6778", "#e3e8ef", "#1d5fd1"))
    content_width = A4[0] - 36 * mm

    def style(name: str, font: str = regular, size: float = 10, color: Any = ink, **extra: Any) -> ParagraphStyle:
        return ParagraphStyle(name, fontName=font, fontSize=size, leading=size * 1.45, textColor=color, **extra)

    body = style("body")
    small = style("small", size=8.5)
    small_bold = style("small_bold", font=bold, size=8.5, color=muted)
    note = style("note", size=8.5, color=muted)
    h1 = style("h1", font=bold, size=20, spaceAfter=2)
    h2 = style("h2", font=bold, size=13, spaceBefore=14, spaceAfter=6)
    h3 = style("h3", font=bold, size=10, spaceBefore=8, spaceAfter=4)
    eyebrow = style("eyebrow", font=bold, size=8.5, color=accent)
    kpi_value = style("kpi_value", font=bold, size=16)
    # Maddeler arası boşluk satır aralığından belirgin büyük olmalı: aksi halde PDF okuyucular ve metin
    # çıkarıcılar ardışık maddeleri tek paragraf sayıp birleştirir ("...görülüyor.• 3 Boyutlu").
    bullet = style("bullet", leftIndent=14, firstLineIndent=-10, spaceAfter=_BULLET_GAP, bulletFontName=regular)

    def para(text: Any, paragraph_style: ParagraphStyle = body) -> Paragraph:
        return Paragraph(xml_escape(safe(str(text))), paragraph_style)

    def bullets(items: Sequence[str], numbered: bool = False) -> list[Paragraph]:
        return [
            # İşaret ve içerik aynı metin bloğunda: PDF metin çıkarımında da maddeler ayrılır.
            Paragraph(xml_escape(safe(f"{index}. {item}" if numbered else f"• {item}")), bullet)
            for index, item in enumerate(items, start=1)
        ]

    def table(headers: list[str], rows: list[list[str]], widths: list[float] | None = None) -> Table:
        data = [[para(item, small_bold) for item in headers]] + [[para(cell, small) for cell in row] for row in rows]
        column_widths = [content_width * share for share in widths] if widths else [content_width / len(headers)] * len(headers)
        flowable = Table(data, colWidths=column_widths, repeatRows=1)
        flowable.setStyle(TableStyle([
            ("LINEBELOW", (0, 0), (-1, -1), 0.5, line),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f7fa")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return flowable

    tables = _tables(report)
    story: list[Any] = [
        para("DATA AUTOPILOT AI - YÖNETİCİ RAPORU", eyebrow),
        para(report.dataset_name, h1),
        para(f"Oluşturulma: {report.generated_at}", note),
        Spacer(1, 10),
    ]
    kpis = _kpis(report)
    kpi_table = Table(
        [[[para(label, note), para(value, kpi_value), para(detail, note)] for label, value, detail in kpis]],
        colWidths=[content_width / len(kpis)] * len(kpis),
    )
    kpi_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, line), ("INNERGRID", (0, 0), (-1, -1), 0.5, line),
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [kpi_table, para("Yönetici Özeti", h2), *bullets(report.headline)]
    story += [para("Öneriler", h2), *bullets(report.recommendations, numbered=True)]

    fallbacks = 0
    timeseries_images = 0
    if report.timeseries:
        series = report.timeseries
        facts = "   |   ".join(f"{label}: {value}" for label, value in _timeseries_facts(series))
        story += [para("Zaman Serisi", h2), para(series.verdict), Spacer(1, 4), para(facts, note), Spacer(1, 6)]
        story += bullets([series.trend_text, series.seasonality_text, series.forecast_text])
        for figure in timeseries_figures:
            if not can_export(figure):
                continue
            png = chart_png(figure)
            fallbacks += int(getattr(png, "fallback", False))
            story.append(Image(BytesIO(png), width=content_width, height=content_width * 0.56))
            timeseries_images += 1
        if "anomalies" in tables:
            story += [
                para("Olağan dışı dönemler", h3),
                table(*tables["anomalies"], widths=[0.28, 0.18, 0.18, 0.18, 0.18]),
            ]
        story += [para(f"Not: {item}", note) for item in series.notes]
    elif report.timeseries_message:
        story += [para("Zaman Serisi", h2), para(report.timeseries_message)]

    if "rules" in tables:
        story += [
            para("İş Kuralı Uyumu", h2),
            table(*tables["rules"], widths=[0.52, 0.12, 0.12, 0.12, 0.12]),
            para(report.rules.method_note, note),
        ]

    if report.root_cause is not None:
        cause = report.root_cause
        story += [para("Neden Değişti?", h2), para(cause.headline)]
        if "root_dimensions" in tables:
            story += [
                table(*tables["root_dimensions"], widths=[0.18, 0.14, 0.68]),
                para(f"“{cause.dimensions[0].dimension}” grupları", h3),
                table(*tables["root_groups"], widths=[0.24, 0.14, 0.14, 0.16, 0.16, 0.16]),
            ]
        story += [para(item, note) for item in [*cause.skipped, cause.method_note]]

    if report.experiment is not None:
        test = report.experiment
        story += [
            para("A/B Testi", h2), para(test.verdict),
            table(*tables["experiment"], widths=[0.30, 0.16, 0.22, 0.32]),
            Spacer(1, 6),
            table(*tables["experiment_facts"], widths=[0.36, 0.64]),
            *[para(item, note) for item in [*test.notes, test.method_note]],
        ]

    if report.customers is not None:
        story += [para("Müşteri Analizi", h2), *bullets(_customer_facts(report.customers)),
                  para("RFM segmentleri", h3), table(*tables["segments"], widths=[0.26, 0.13, 0.15, 0.15, 0.17, 0.14]),
                  para("Tarihsel müşteri değeri", h3), table(*tables["clv"], widths=[0.25, 0.25, 0.25, 0.25])]
        story += [para(item, note) for item in [*report.customers.definitions.values(), *report.customers.notes]]

    if report.drift:
        drift = report.drift
        story += [para("Dönem Karşılaştırması", h2), para(drift.summary)]
        story += bullets([f"Şema: “{item.column}” {item.change} ({item.detail})" for item in drift.schema_changes])
        if "drift" in tables:
            story.append(table(*tables["drift"], widths=[0.10, 0.20, 0.11, 0.09, 0.50]))
        story.append(para(drift.method_note, note))

    story.append(para("Tahmin Modeli", h2))
    if report.model:
        model = report.model
        story += [
            para(model.verdict),
            *[para(item) for item in model.leakage_warnings],
            Spacer(1, 4),
            para(f"Hedef: {model.target}   |   Problem tipi: {model.problem_type}   |   En iyi model: {model.best_model}", note),
            Spacer(1, 6),
        ]
        if "leaderboard" in tables:
            headers, rows = tables["leaderboard"]
            # Model adlarına ayrılan alan en uzun sklearn adını da tek satırda tutar.
            count = len(headers)
            widths = [0.48, *([0.52 / (count - 1)] * (count - 1))] if count > 1 else [1.0]
            story.append(table(headers, rows, widths=widths))
        if model.top_features:
            features = model.top_features[::-1]
            height = 22 + 16 * len(features)
            drawing = Drawing(content_width, height)
            chart = HorizontalBarChart()
            chart.x, chart.y, chart.width, chart.height = 170, 14, content_width - 200, height - 20
            chart.data = [[feature.importance for feature in features]]
            chart.categoryAxis.categoryNames = [safe(feature.name[:32]) for feature in features]
            chart.categoryAxis.labels.fontName = regular
            chart.categoryAxis.labels.fontSize = 8
            chart.categoryAxis.labels.boxAnchor = "e"
            chart.categoryAxis.labels.dx = -6
            chart.categoryAxis.strokeColor = line
            chart.valueAxis.valueMin = 0
            chart.valueAxis.labels.fontName = regular
            chart.valueAxis.labels.fontSize = 7
            chart.valueAxis.strokeColor = line
            chart.valueAxis.labelTextFormat = lambda value: _dec(value, 3)
            chart.bars[0].fillColor = accent
            chart.bars[0].strokeColor = None
            drawing.add(chart)
            story += [para("Özellik önemi (test verisinde permütasyon etkisi)", h3), drawing]
        if _evaluation_text(model):
            story.append(para(_evaluation_text(model), note))
        story.append(para(
            "Eksik doldurma ve ölçekleme yalnızca eğitim verisinden öğrenildi; skorlar modelin hiç görmediği "
            "test bölmesinde ölçüldü.", note,
        ))
    else:
        story.append(para(report.model_message or "Tahmin modeli eğitilmedi."))

    pdp_images = 0
    if report.pdp:
        pdp_budget = max(0, 6 - int(bool(report.model and report.model.top_features)) - timeseries_images)
        story += [para("Model Neden Böyle Tahmin Ediyor?", h2), para(PDP_NOTE, note)]
        for index, curve in enumerate(report.pdp):
            block: list[Any] = []
            if index < len(pdp_figures) and pdp_images < pdp_budget and can_export(pdp_figures[index]):
                png = chart_png(pdp_figures[index])
                fallbacks += int(getattr(png, "fallback", False))
                image_width = content_width * _PDF_CHART_WIDTH
                block.append(Image(BytesIO(png), width=image_width, height=image_width * 0.56))
                pdp_images += 1
            block.append(para(curve.interpretation))
            story.append(KeepTogether(block))

    story.append(para("Veri Kalitesi", h2))
    story += bullets(report.issues) if report.issues else [para("Başlangıçta belirgin bir kalite sorunu tespit edilmedi.")]
    if "missing" in tables:
        story += [para("Temiz veride eksik değer içeren sütunlar", h3), table(*tables["missing"], widths=[0.7, 0.3])]
    if "outliers" in tables:
        story += [
            para("Şüpheli uç değerler (değiştirilmedi)", h3),
            table(*tables["outliers"], widths=[0.24, 0.11, 0.09, 0.20, 0.10, 0.12, 0.14]),
        ]
        if any(item.method == LOG_METHOD for item in report.outliers):
            story.append(para(_LOG_NOTE, note))
    if report.charts:
        # Görsel, sayfaya üç grafik sığacak boyutta; her grafik kendi başlığı ve notuyla bölünmeden akar,
        # zorunlu sayfa sonu yoktur. Bölüm başlığı ilk grafikle aynı sayfada kalır.
        image_width = content_width * _PDF_CHART_WIDTH
        heading: list[Any] = [para("Öne Çıkan Grafikler", h2)]
        budget = 6 - int(bool(report.model and report.model.top_features)) - timeseries_images - pdp_images
        exported = 0
        shown: set[int] = set()  # PDF'te görseli basılan grafiklerin sırası
        for position, (chart_summary, figure) in enumerate(zip(report.charts, figures)):
            if exported >= budget:
                break
            if not can_export(figure):
                continue
            png = chart_png(figure)
            fallbacks += int(getattr(png, "fallback", False))
            story.append(KeepTogether([
                *heading,
                para(chart_summary.title, h3),
                Image(BytesIO(png), width=image_width, height=image_width * 0.56),
                para(chart_summary.insight, note),
                *([para(_sample_note(chart_summary.sample_size), note)] if chart_summary.sample_size else []),
            ]))
            heading = []
            exported += 1
            shown.add(position)
        # Görseli basılan grafiğin bulgusu zaten altında yazıyor; burada tekrar edilmez. Geriye yalnızca
        # PDF'e girmeyen grafikler (3B, paralel koordinat, sınır dışı) kalır ve nerede bulunacağı belirtilir.
        remaining = [
            f"{chart.title}: {chart.insight} (grafiği HTML raporunda)"
            for position, chart in enumerate(report.charts) if position not in shown
        ]
        if remaining:
            story.append(para("Diğer Bulgular" if shown else "Öne Çıkan Bulgular", h2))
            story += bullets(remaining)
        story.append(para("Etkileşimli grafikler HTML raporunda yer alır.", note))
    if fallbacks:
        story.append(para(
            f"Not: {fallbacks} grafik birincil çizici (Kaleido) kullanılamadığı için yedek çizici (Matplotlib) "
            "ile üretildi; görünümü HTML rapordaki etkileşimli grafikten farklı olabilir.", note,
        ))
    if "joins" in tables:
        story += [
            para("Veri Birleştirme", h2),
            table(*tables["joins"], widths=[0.24, 0.12, 0.38, 0.13, 0.13]),
            para("Birleştirme satır sayısını değiştirmez; eşleşmeyen satırlarda eklenen sütunlar boş kalır.", note),
        ]
    if "actions" in tables or report.plan_origin or "rejected" in tables:
        story.append(para("Uygulanan İşlemler", h2))
        if _plan_text(report):
            story.append(para(_plan_text(report), note))
        if "actions" in tables:
            story.append(table(*tables["actions"], widths=[0.22, 0.22, 0.44, 0.12]))
        if "rejected" in tables:
            story += [para("Uygulanmayan plan önerileri (veri koruma)", h3),
                      table(*tables["rejected"], widths=[0.22, 0.18, 0.60])]
    story += [
        Spacer(1, 14),
        para("Bu rapor deterministik kurallarla üretildi. Ham satırlar ve kişisel veriler rapora dahil edilmez.", note),
    ]

    buffer = BytesIO()
    SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=safe(report.title), author="Data Autopilot AI",
    ).build(story)
    return buffer.getvalue()
