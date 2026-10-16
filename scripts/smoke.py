#!/usr/bin/env python
"""真实端到端冒烟测试（会调用真实模型，会花钱/耗额度）。

这个脚本一次性验证简历里三条 bullet 描述的机制是否真的生效：

    第 1 条 → 多轮对话 + 历史会话加载
    第 2 条 → 范围判断（超范围拒答）+ 多轮查询重写
    第 3 条 → 工具调用 + 会话持久化 + 模型切换

跑法：
    .venv/Scripts/python.exe scripts/smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from fastapi.testclient import TestClient

from app.main import app

USER = "smoke-user"


def line(title: str) -> None:
    print()
    print("─" * 68)
    print(title)
    print("─" * 68)


def show(data: dict) -> None:
    print(f"回答：{data['answer'][:300]}")
    meta = data.get("retrieval") or {}
    print(
        f"  范围判断={meta.get('in_scope')}  改写轮数={meta.get('rewrite_rounds')}  "
        f"命中={meta.get('hit_count')}  最高相似度={meta.get('best_score')}"
    )
    if meta.get("rewritten_query"):
        print(f"  改写后查询：{meta['rewritten_query']}")
    if data.get("citations"):
        for c in data["citations"][:2]:
            print(f"  引用：{c.get('source')} · {c.get('section')}（{c.get('score')}）")
    if data.get("tool_calls"):
        for t in data["tool_calls"]:
            print(f"  工具：{t['name']} 参数={t['arguments']}")


def main() -> int:
    with TestClient(app) as client:
        health = client.get("/health").json()
        line("① 服务健康检查")
        print(health)

        # ── 用例 1：常规知识库问答 ──
        line("② 知识库问答（应命中检索，不使用工具）")
        r1 = client.post(
            "/chat", json={"message": "畅享39 多少钱？包含多少流量？", "user_id": USER}
        ).json()
        show(r1)
        conv_id = r1["conversation_id"]

        # ── 用例 2：多轮追问，依赖上文 ──
        line("③ 多轮追问（应触发查询重写，把「那它超出怎么算」补全）")
        r2 = client.post(
            "/chat",
            json={"message": "那它超出之后怎么算？", "conversation_id": conv_id, "user_id": USER},
        ).json()
        show(r2)

        # ── 用例 3：需要工具的场景 ──
        line("④ 推荐场景（应调用工具做筛选，而不是模型自己心算）")
        r3 = client.post(
            "/chat",
            json={
                "message": "帮我推荐一个月租 40 元以内、流量尽量多的套餐",
                "conversation_id": conv_id,
                "user_id": USER,
            },
        ).json()
        show(r3)

        # ── 用例 4：超范围问题 ──
        line("⑤ 超范围问题（应直接拒答，不检索、不硬答）")
        r4 = client.post(
            "/chat",
            json={"message": "今天上海天气怎么样？", "conversation_id": conv_id, "user_id": USER},
        ).json()
        show(r4)

        # ── 验证持久化 ──
        line("⑥ 历史会话恢复（从库里读回全部消息）")
        msgs = client.get(f"/conversations/{conv_id}/messages").json()
        print(f"会话 {conv_id} 共 {len(msgs)} 条消息：")
        for m in msgs:
            role = {"user": "用户", "assistant": "客服"}.get(m["role"], m["role"])
            cites = len(m.get("citations") or [])
            print(f"  #{m['seq']:<2} [{role}] {m['content'][:60]}{'…' if len(m['content']) > 60 else ''}"
                  f"{f'  （{cites} 条引用）' if cites else ''}")

        line("⑦ 用户长期记忆（跨会话抽取）")
        print(client.get("/memories", params={"user_id": USER}).json())

        line("⑧ 模型提供方与切换能力")
        print(client.get("/providers").json())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
