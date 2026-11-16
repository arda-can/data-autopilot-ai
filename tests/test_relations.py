from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.relations import (
    JoinPlan,
    JoinStep,
    RelationFinder,
    apply_plan,
    default_base,
    prune_plan,
    relation_figure,
    short_name,
    suggest_plan,
)


@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    """Olist benzeri küçük şema: müşteri ← sipariş → kalem → ürün; sipariş → yorum; posta kodu N-N."""
    rng = np.random.default_rng(0)
    customers = pd.DataFrame({
        "customer_id": [f"c{index}" for index in range(60)],
        "customer_zip_code_prefix": rng.integers(1000, 1030, 60),
        "customer_city": rng.choice(["sao paulo", "rio"], 60),
    })
    orders = pd.DataFrame({
        "order_id": [f"o{index}" for index in range(80)],
        "customer_id": [f"c{index % 60}" for index in range(80)],
        "order_status": rng.choice(["delivered", "shipped"], 80, p=[0.9, 0.1]),
    })
    products = pd.DataFrame({
        "product_id": [f"p{index}" for index in range(25)],
        "product_category_name": rng.choice(["moveis", "esporte", "livros"], 25),
    })
    items = pd.DataFrame({
        "order_id": [f"o{index % 80}" for index in range(150)] + ["o_kayip"] * 2,
        "product_id": [f"p{index % 25}" for index in range(152)],
        "price": np.round(rng.uniform(10, 100, 152), 2),
        "item_rank": [1 + index // 80 for index in range(152)],
    })
    reviews = pd.DataFrame({
        "review_id": [f"r{index}" for index in range(76)],
        "order_id": [f"o{index}" for index in range(75)] + ["o0"],  # o0 iki yorumlu; o75–o79 yorumsuz
        "review_score": rng.integers(1, 6, 76),
    })
    geolocation = pd.DataFrame({
        "geolocation_zip_code_prefix": np.repeat(np.arange(1000, 1030), 3),
        "geolocation_lat": rng.normal(-23, 1, 90),
    })
    return {"customers": customers, "orders": orders, "products": products, "order_items": items,
            "order_reviews": reviews, "geolocation": geolocation}


@pytest.fixture(scope="module")
def relations(tables):
    return RelationFinder().discover(tables)


def _find(relations, left: str, right: str):
    return next(item for item in relations if {item.left_table, item.right_table} == {left, right})


def test_relations_and_kinds(relations) -> None:
    pairs = {(item.left_table, item.left_column, item.right_table, item.right_column, item.kind) for item in relations}
    assert ("customers", "customer_id", "orders", "customer_id", "1-1") not in pairs  # müşteri tekrar ediyor
    assert ("orders", "customer_id", "customers", "customer_id", "N-1") in pairs
    assert ("order_items", "order_id", "orders", "order_id", "N-1") in pairs
    assert ("order_items", "product_id", "products", "product_id", "N-1") in pairs
    assert ("order_reviews", "order_id", "orders", "order_id", "N-1") in pairs
    zip_relation = _find(relations, "customers", "geolocation")
    assert zip_relation.kind == "N-N" and not zip_relation.suggested and "Çoktan-çoğa" in zip_relation.note


def test_unrelated_small_integers_do_not_match(relations) -> None:
    assert not any("item_rank" in (item.left_column, item.right_column) for item in relations)
    assert not any("review_score" in (item.left_column, item.right_column) for item in relations)


def test_describe_lists_unique_keys(tables) -> None:
    info = {item.name: item for item in RelationFinder().describe(tables)}
    assert info["orders"].unique_keys == ["order_id"]
    assert info["customers"].unique_keys == ["customer_id"]
    assert info["order_items"].rows == 152


def test_plan_from_orders(tables, relations) -> None:
    plan = suggest_plan("orders", relations, tables)
    steps = {(step.table, step.mode, step.depth) for step in plan.steps}
    assert steps == {("customers", "eşleştir", 1), ("order_items", "topla", 1), ("order_reviews", "topla", 1)}
    # Ürünlere toplanan kalemler üzerinden geçilemez; coğrafya N-N olduğu için önerilmez.
    assert not any(step.table in ("products", "geolocation") for step in plan.steps)


def test_plan_from_items_reaches_two_hops(tables, relations) -> None:
    plan = suggest_plan("order_items", relations, tables)
    steps = {(step.table, step.mode, step.depth, step.left_table) for step in plan.steps}
    assert ("orders", "eşleştir", 1, "order_items") in steps
    assert ("products", "eşleştir", 1, "order_items") in steps
    assert ("customers", "eşleştir", 2, "orders") in steps
    assert ("order_reviews", "topla", 2, "orders") in steps


def test_apply_plan_keeps_grain_and_aggregates_correctly(tables, relations) -> None:
    snapshot = {name: frame.copy(deep=True) for name, frame in tables.items()}
    result = apply_plan(tables, suggest_plan("orders", relations, tables))
    frame = result.frame
    assert len(frame) == len(tables["orders"]) and frame["order_id"].tolist() == tables["orders"]["order_id"].tolist()
    expected_sum = tables["order_items"].groupby("order_id")["price"].sum()
    joined_sum = frame.set_index("order_id")["order_items_price_toplam"]
    assert joined_sum["o0"] == pytest.approx(expected_sum["o0"])
    assert frame.set_index("order_id").loc["o0", "order_reviews_kayıt_sayısı"] == 2
    assert frame.set_index("order_id").loc["o79", "order_reviews_kayıt_sayısı"] != frame.set_index("order_id").loc["o79", "order_reviews_kayıt_sayısı"]  # NaN
    assert "customer_city" in frame.columns and "customer_id" in frame.columns
    for name, original in snapshot.items():
        pd.testing.assert_frame_equal(tables[name], original)


def test_join_logs_report_match_rate(tables, relations) -> None:
    logs = {log.table: log for log in apply_plan(tables, suggest_plan("orders", relations, tables)).logs}
    assert all(log.rows_before == log.rows_after == 80 for log in logs.values())
    assert logs["customers"].match_rate == 1.0
    assert logs["order_reviews"].match_rate == pytest.approx(75 / 80)
    assert "Eşleşmeyen satırlarda yeni sütunlar boş kaldı." in logs["order_reviews"].note


def test_two_hop_join_and_column_collisions(tables, relations) -> None:
    frame = apply_plan(tables, suggest_plan("order_items", relations, tables)).frame
    assert len(frame) == 152
    assert {"order_status", "product_category_name", "customer_city", "order_reviews_review_score_ort"} <= set(frame.columns)
    unmatched = frame[frame["order_id"] == "o_kayip"]
    assert unmatched["order_status"].isna().all()


def test_lookup_on_non_unique_key_is_refused(tables) -> None:
    plan = JoinPlan(base="orders", steps=[JoinStep(
        table="order_items", left_table="orders", left_column="order_id", right_column="order_id",
        mode="eşleştir", depth=1,
    )])
    with pytest.raises(ValueError, match="benzersiz değil"):
        apply_plan(tables, plan)


def test_step_needs_joined_source(tables) -> None:
    plan = JoinPlan(base="orders", steps=[JoinStep(
        table="products", left_table="order_items", left_column="product_id", right_column="product_id",
        mode="eşleştir", depth=1,
    )])
    with pytest.raises(ValueError, match="henüz birleştirilmemiş"):
        apply_plan(tables, plan)


def test_integer_and_text_keys_match() -> None:
    left = pd.DataFrame({"zip_code": list(range(1, 16)) * 2, "x": range(30)})
    right = pd.DataFrame({"zip_code": [str(value) for value in range(1, 21)], "city": [f"k{value}" for value in range(20)]})
    tables = {"a": left, "b": right}
    plan = suggest_plan("a", RelationFinder().discover(tables), tables)
    assert [(step.table, step.mode) for step in plan.steps] == [("b", "eşleştir")]
    assert apply_plan(tables, plan).frame["city"].notna().all()


def test_helpers(tables, relations) -> None:
    assert short_name("data/olist_order_items_dataset.csv") == "order_items"
    assert short_name("product_category_name_translation.csv") == "product_category_name_translation"
    figure = relation_figure(RelationFinder().describe(tables), relations)
    assert len(figure.data) == 2 * len(relations) + 1
    with pytest.raises(ValueError, match="bulunamadı"):
        suggest_plan("yok", relations, tables)


def test_prune_plan_drops_dependent_steps(tables, relations) -> None:
    plan = suggest_plan("order_items", relations, tables)
    approved = {step.table for step in plan.steps} - {"orders"}
    pruned, dropped = prune_plan(plan, approved)
    assert "orders" not in [step.table for step in pruned.steps]
    assert not any(step.left_table == "orders" for step in pruned.steps)
    assert any("dayandığı “orders” birleştirilmedi" in reason for reason in dropped)
    assert apply_plan(tables, pruned).frame.shape[0] == 152


def test_default_base_is_the_hub_table(tables, relations) -> None:
    assert default_base(relations, list(tables)) == "orders"
