#!/usr/bin/env python
"""Gradio Web 界面。

**这个文件不 import 任何业务代码** —— 它只通过 HTTP 调用 FastAPI 的接口。
这就是「接口与界面解耦」的实际含义：

    换前端（Vue / 小程序 / 桌面端）→ 直接调同一批接口，后端零改动
    后端可以脱离界面独立测试 → 用 pytest + TestClient 就够
    界面的 bug 影响不到业务逻辑的稳定性

启动方式（需要先启动 API 服务）：
    # 终端 1
    .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
    # 终端 2
    .venv/Scripts/python.exe app/ui/gradio_app.py
"""
from __future__ import annotations

import os
import sys

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 120.0


# ── 与后端交互 ────────────────────────────────────────────────
def _get(path: str, **params):
    resp = httpx.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict):
    resp = httpx.post(f"{API_BASE}{path}", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_status() -> str:
    try:
        h = _get("/health")
    except Exception as exc:  # noqa: BLE001
        return f"❌ 无法连接后端（{API_BASE}）：{type(exc).__name__}"
    return (
        f"✅ 后端正常\n"
        f"模型：{h['provider']}\n"
        f"数据库：{h['database']}\n"
        f"知识库：{h['knowledge_chunks']} 个片段 · 套餐 {h['packages']} 条"
    )


def list_conversations(user_id: str):
    try:
        rows = _get("/conversations", user_id=user_id)
    except Exception:  # noqa: BLE001
        return gr.update(choices=[], value=None)
    choices = [(f"{r['title']}（{r['message_count']}条 · {r['updated_at']}）", r["id"]) for r in rows]
    return gr.update(choices=choices, value=choices[0][1] if choices else None)


def load_conversation(conv_id: str):
    if not conv_id:
        return [], []
    try:
        msgs = _get(f"/conversations/{conv_id}/messages")
    except Exception:  # noqa: BLE001
        return [], []

    chat_history = []
    citations = []
    for m in msgs:
        if m["role"] == "user":
            chat_history.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            chat_history.append({"role": "assistant", "content": m["content"]})
            for c in m.get("citations") or []:
                citations.append(
                    f"· {c.get('source')} · {c.get('section')}（相似度 {c.get('score')}）"
                )
    return chat_history, citations


def send(message: str, chat_history: list, conv_id: str, user_id: str):
    """发送一条消息。返回 (历史, 清空输入框, 会话id, 引用文本, 调试信息)。"""
    message = (message or "").strip()
    if not message:
        return chat_history, "", conv_id, "", ""

    history = list(chat_history or [])
    history.append({"role": "user", "content": message})

    try:
        data = _post(
            "/chat",
            {"message": message, "conversation_id": conv_id or None, "user_id": user_id or "anonymous"},
        )
    except Exception as exc:  # noqa: BLE001
        history.append(
            {"role": "assistant", "content": f"❌ 请求失败：{type(exc).__name__}: {exc}"}
        )
        return history, "", conv_id, "", ""

    history.append({"role": "assistant", "content": data["answer"]})

    cites = "\n".join(
        f"· {c.get('source')} · {c.get('section')}（{c.get('score')}）"
        for c in data.get("citations") or []
    ) or "（本轮没有引用知识库片段）"

    meta = data.get("retrieval") or {}
    debug = (
        f"范围判断：{'在范围内' if meta.get('in_scope') else '超出范围'}"
        f"（{meta.get('scope_reason', '')}）\n"
        f"改写后查询：{meta.get('rewritten_query') or '（未改写）'}\n"
        f"改写轮数：{meta.get('rewrite_rounds', 0)} · 命中片段：{meta.get('hit_count', 0)}"
        f" · 最高相似度：{meta.get('best_score', 0)}\n"
        f"工具调用：{data.get('tool_calls') or '（无）'}"
    )
    return history, "", data["conversation_id"], cites, debug


def new_conversation():
    return [], "", None, "", ""


def fetch_memories(user_id: str) -> str:
    try:
        mem = _get("/memories", user_id=user_id or "anonymous")
    except Exception as exc:  # noqa: BLE001
        return f"读取失败：{exc}"
    if not mem:
        return "（暂无长期记忆）"
    return "\n".join(f"· {k}：{v}" for k, v in mem.items())


def switch_model(provider: str) -> str:
    try:
        data = _post("/providers/switch", {"provider": provider})
        return f"✅ 已切换到：{data['switched_to']}"
    except Exception as exc:  # noqa: BLE001
        return f"❌ 切换失败：{exc}"


# ── 界面 ─────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="流量套餐智能客服", theme=gr.themes.Soft()) as demo:
        conv_state = gr.State(None)

        gr.Markdown("## 流量套餐智能客服\n基于 RAG + 工具调用的知识库问答 · 界面与后端通过 HTTP 解耦")

        with gr.Row():
            user_id = gr.Textbox(value="demo-user", label="用户标识", scale=2)
            status = gr.Textbox(label="服务状态", interactive=False, scale=3)
            refresh = gr.Button("刷新状态", scale=1)

        with gr.Tabs():
            # ── 对话 ──
            with gr.Tab("对话"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(type="messages", height=460, label="对话")
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="例如：39 元以内的套餐哪个划算？",
                                label="",
                                scale=5,
                                autofocus=True,
                            )
                            send_btn = gr.Button("发送", variant="primary", scale=1)
                        with gr.Row():
                            clear_btn = gr.Button("新会话")
                            conv_picker = gr.Dropdown(
                                label="历史会话", choices=[], interactive=True, scale=2
                            )
                            load_btn = gr.Button("加载历史")

                    with gr.Column(scale=2):
                        citations = gr.Textbox(
                            label="本轮引用（可追溯）", lines=6, interactive=False
                        )
                        debug = gr.Textbox(
                            label="检索链路诊断（可追溯）", lines=9, interactive=False
                        )
                        memories_box = gr.Textbox(
                            label="该用户的长期记忆", lines=5, interactive=False
                        )
                        mem_btn = gr.Button("刷新记忆")

            # ── 设置 ──
            with gr.Tab("模型设置"):
                gr.Markdown(
                    "**抽象模型调用层**：切换提供方不需要改任何业务代码，"
                    "因为三者都走同一套 OpenAI 兼容协议。"
                )
                with gr.Row():
                    provider = gr.Dropdown(
                        choices=["deepseek", "qwen", "zhipu", "ollama"],
                        value=os.getenv("LLM_PROVIDER", "zhipu"),
                        label="模型提供方",
                    )
                    switch_btn = gr.Button("切换", variant="primary")
                switch_out = gr.Textbox(label="结果", interactive=False)

        # ── 事件绑定 ──
        demo.load(fetch_status, outputs=status)
        refresh.click(fetch_status, outputs=status)

        send_btn.click(
            send,
            inputs=[msg, chatbot, conv_state, user_id],
            outputs=[chatbot, msg, conv_state, citations, debug],
        ).then(fetch_memories, inputs=user_id, outputs=memories_box).then(
            list_conversations, inputs=user_id, outputs=conv_picker
        )
        msg.submit(
            send,
            inputs=[msg, chatbot, conv_state, user_id],
            outputs=[chatbot, msg, conv_state, citations, debug],
        ).then(fetch_memories, inputs=user_id, outputs=memories_box).then(
            list_conversations, inputs=user_id, outputs=conv_picker
        )

        clear_btn.click(
            new_conversation,
            outputs=[chatbot, conv_state, conv_picker, citations, debug],
        )
        load_btn.click(
            load_conversation, inputs=conv_picker, outputs=[chatbot, citations]
        )
        conv_picker.change(
            lambda cid: cid, inputs=conv_picker, outputs=conv_state
        )

        mem_btn.click(fetch_memories, inputs=user_id, outputs=memories_box)
        user_id.change(fetch_memories, inputs=user_id, outputs=memories_box)
        user_id.change(list_conversations, inputs=user_id, outputs=conv_picker)

        switch_btn.click(switch_model, inputs=provider, outputs=switch_out).then(
            fetch_status, outputs=status
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
