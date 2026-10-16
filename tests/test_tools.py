"""工具层测试。

工具是「让模型不要自己心算」的那一层，所以它的正确性必须是**确定的**、
可回归的 —— 这也正是把筛选逻辑写成代码而不是交给模型的原因。

下面几个用例都来自真实运行中发现的问题，属于「把失败固化成测试」。
"""
from __future__ import annotations

import json

from core.agent.tools import PackageCatalog, TOOL_SCHEMAS, execute_tool


def catalog() -> PackageCatalog:
    from core.config import settings

    return PackageCatalog(settings.packages_file)


# ── 推荐筛选 ──────────────────────────────────────────────────
def test_recommend_excludes_daily_and_directional_packages():
    """真实踩坑：定向流量包曾以 0.3 元/GB 排到第一，把真正合适的套餐挤下去。"""
    recs = catalog().recommend(max_price=40)
    names = [r["name"] for r in recs]
    assert "视频定向包" not in names, "定向流量包不应参与通用流量的性价比排序"
    assert "日租卡" not in names, "日付套餐不应参与月付口径的排序"


def test_recommend_respects_budget():
    cat = catalog()
    recs = cat.recommend(max_price=30)
    assert recs, "30 元以内应当有可选套餐"
    assert all(r["price"] <= 30 for r in recs)


def test_recommend_sorted_by_unit_price():
    recs = catalog().recommend(max_price=100, limit=5)
    units = [r["unit_price"] for r in recs]
    assert units == sorted(units), "应按每 GB 单价升序排列"


def test_recommend_returns_empty_when_over_constrained():
    """条件过严时返回空列表，由模型决定是否放宽 —— 而不是硬塞一个不满足条件的。"""
    recs = catalog().recommend(max_price=5, min_data_gb=50)
    assert recs == []


def test_recommend_zero_threshold_is_treated_as_no_filter():
    """min_data_gb=0 表示用户没提流量要求，不应筛掉任何套餐。"""
    a = catalog().recommend(max_price=40)
    b = catalog().recommend(max_price=40, min_data_gb=0)
    assert [x["name"] for x in a] == [x["name"] for x in b]


# ── 查询 ──────────────────────────────────────────────────────
def test_search_by_name_and_tag():
    cat = catalog()
    by_name = cat.find("畅享39")
    assert any(p["name"] == "畅享39" for p in by_name)

    by_tag = cat.find("学生")
    assert any(p["name"] == "学生专享包" for p in by_tag)

    assert cat.find("不存在的套餐名") == []
    assert len(cat.find("")) == len(cat.all()), "空关键词应返回全部"


# ── 工具执行（错误回灌约定）────────────────────────────────────
def test_execute_tool_returns_text_never_raises():
    cat = catalog()

    # 不存在的工具
    out = execute_tool("no_such_tool", "{}", cat)
    assert "工具不存在" in out

    # 参数不是合法 JSON
    out = execute_tool("search_package", "keyword=畅享39", cat)
    assert "不是合法 JSON" in out

    # 参数类型不对
    out = execute_tool("search_package", "[1,2,3]", cat)
    assert "JSON 对象" in out


def test_execute_tool_success_payload_is_json():
    out = execute_tool("search_package", json.dumps({"keyword": "畅享39"}), catalog())
    data = json.loads(out)
    assert isinstance(data, list)
    assert data[0]["名称"] == "畅享39"
    assert "39" in data[0]["月费"]


def test_tool_schemas_are_wellformed():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
