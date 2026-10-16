"""Agent 可调用的工具。

工具是「知识库问答」之外的第二条能力线：
知识库负责把**已有信息的解释**讲清楚，
工具负责回答**需要实时计算或筛选**的问题（推荐、比价、按预算筛选）。

两者分工的理由：把「查表」交给代码比交给模型可靠得多。
模型适合做表达和推理，不适合做精确的数值筛选 ——
让它自己从 8 个套餐里挑出「20 元以内且流量最多的」，它可能挑错。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import settings


class PackageCatalog:
    """套餐目录。数据来自 data/packages.json。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.packages_file)
        self._items: list[dict] = []
        self.reload()

    def reload(self) -> int:
        if not self.path.exists():
            self._items = []
            return 0
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = payload.get("packages", [])
        return len(self._items)

    def all(self) -> list[dict]:
        return list(self._items)

    def find(self, keyword: str = "") -> list[dict]:
        """按名称或标签模糊查找。"""
        kw = (keyword or "").strip().lower()
        if not kw:
            return self.all()
        out = []
        for p in self._items:
            haystack = " ".join(
                [p.get("name", ""), p.get("note", ""), " ".join(p.get("tags", []))]
            ).lower()
            if kw in haystack:
                out.append(p)
        return out

    def recommend(
        self,
        max_price: float | None = None,
        min_data_gb: float | None = None,
        need_voice: bool = False,
        prefer_cheap: bool = True,
        limit: int = 3,
    ) -> list[dict]:
        """按条件筛选并排序。

        排序策略：满足条件的前提下，优先「每 GB 单价最低」——
        因为用户问「推荐」时，真实诉求通常是性价比，而不是绝对价格最低。
        """
        candidates = []
        for p in self._items:
            if p.get("billing") == "日付":
                # 日租卡不参与月付口径的性价比比较，否则单价会被算歪
                continue
            if "定向流量" in p.get("tags", []):
                # 定向流量包也不能参与：它的 30GB 只能在指定 App 内使用，
                # 拿它和通用流量比「每 GB 单价」会得出严重失真的结论
                # （实测中它曾以 0.3 元/GB 排到第一，把真正合适的套餐挤下去）
                continue
            if max_price is not None and p.get("price", 0) > max_price:
                continue
            if min_data_gb is not None and p.get("data_gb", 0) < min_data_gb:
                continue
            if need_voice and "通话" not in " ".join(p.get("tags", [])) and p.get("price", 0) < 59:
                continue
            unit = p.get("price", 0) / max(p.get("data_gb", 1), 1)
            candidates.append({**p, "unit_price": round(unit, 3)})

        if prefer_cheap:
            candidates.sort(key=lambda x: x["unit_price"])
        else:
            candidates.sort(key=lambda x: x.get("price", 0))
        return candidates[:limit]


# ── 工具的 JSON Schema（给模型看的）──────────────────────────
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_package",
            "description": "按关键词查询流量套餐详情。关键词可以是套餐名（如「畅享39」）或标签（如「学生」「视频」）。不传关键词则返回全部套餐。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "套餐名或标签关键词，可为空"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_package",
            "description": (
                "按用户的预算和流量需求推荐套餐，返回按单位流量价格排序的结果。"
                "当用户问「推荐一个」「哪个划算」「XX 元以内有什么」时使用。"
                "重要：只传用户明确说出的条件 —— 用户没说预算就不要传 max_price，"
                "没说具体流量就不要传 min_data_gb。把「流量尽量多」这类偏好"
                "当成数字门槛，会把本来合适的套餐全部筛掉。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {"type": "number", "description": "月费上限，单位元"},
                    "min_data_gb": {"type": "number", "description": "所需最低流量，单位 GB"},
                    "need_voice": {"type": "boolean", "description": "是否需要包含通话时长"},
                },
                "required": [],
            },
        },
    },
]


def execute_tool(name: str, arguments: str | dict, catalog: PackageCatalog) -> str:
    """执行工具，返回给模型看的文本。

    与运行时里的约定一致：**失败也返回文本，不抛异常**。
    这样模型能看到错误并自行修正参数，而不是整轮中断。
    """
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError as e:
        return f"[工具参数错误] 不是合法 JSON（{e.msg}）。你传的是：{str(arguments)[:100]}"

    if not isinstance(args, dict):
        return "[工具参数错误] 参数必须是一个 JSON 对象"

    if name == "search_package":
        hits = catalog.find(args.get("keyword", ""))
        if not hits:
            return f"没有找到与「{args.get('keyword', '')}」相关的套餐。"
        return json.dumps(
            [
                {
                    "名称": p["name"],
                    "月费": f"{p['price']} 元",
                    "流量": f"{p['data_gb']}GB",
                    "有效期": f"{p['valid_days']} 天",
                    "标签": p.get("tags", []),
                    "说明": p.get("note", ""),
                }
                for p in hits
            ],
            ensure_ascii=False,
            indent=2,
        )

    if name == "recommend_package":
        recs = catalog.recommend(
            max_price=args.get("max_price"),
            min_data_gb=args.get("min_data_gb"),
            need_voice=bool(args.get("need_voice", False)),
        )
        if not recs:
            return "按这些条件没有筛选出合适的套餐，可以放宽一下预算或流量要求。"
        return json.dumps(
            [
                {
                    "名称": p["name"],
                    "月费": f"{p['price']} 元",
                    "流量": f"{p['data_gb']}GB",
                    "折合每 GB": f"{p['unit_price']} 元",
                    "说明": p.get("note", ""),
                }
                for p in recs
            ],
            ensure_ascii=False,
            indent=2,
        )

    return f"[工具不存在] 没有名为 {name} 的工具"
