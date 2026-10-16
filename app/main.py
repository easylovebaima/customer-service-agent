"""FastAPI 服务入口。

简历第一条 bullet 在这一层落地：
> 使用 FastAPI 提供对话 API、Gradio 搭建 Web 界面，接口与界面解耦。

## 「接口与界面解耦」具体指什么

Gradio 界面**不 import 这里面的任何业务代码**，它只通过 HTTP 调用这些 API。
所以：
- 换掉 Gradio 换成 Vue / 小程序，后端一行不改
- 后端可以在没有界面的情况下被 curl / 测试 / 其他服务调用
- 界面的 bug 不会影响后端的稳定性

```
app/
├── main.py            这个文件：FastAPI，提供 /chat 等接口
└── ui/gradio_app.py   Gradio 界面，只发 HTTP 请求，不碰业务代码
```
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.agent.chain import SupportAgent
from core.agent.tools import PackageCatalog
from core.config import describe, settings
from core.llm import available_providers, build_llm, switch_provider
from core.memory import store
from core.memory.db import backend_name, init_db
from core.rag.pipeline import RAGPipeline
from core.rag.vectorstore import VectorStore


# ── 生命周期 ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    vector_store = VectorStore.load(settings.index_path)
    catalog = PackageCatalog()
    llm = build_llm()
    rag = RAGPipeline(
        llm,
        vector_store,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
        max_rewrite_rounds=settings.max_rewrite_rounds,
    )

    app.state.llm = llm
    app.state.rag = rag
    app.state.store = vector_store
    app.state.catalog = catalog
    app.state.agent = SupportAgent(llm, rag, catalog)

    print("=" * 62)
    print("智能客服与知识库问答系统 已启动")
    print("=" * 62)
    print(describe())
    print(f"持久化后端 = {backend_name()}")
    print(f"知识库     = {len(vector_store)} 个片段")
    print(f"套餐数据   = {len(catalog.all())} 条")
    if len(vector_store) == 0:
        print("⚠ 知识库为空，请先运行：python scripts/ingest.py")
    print("=" * 62)
    yield


app = FastAPI(
    title="流量套餐智能客服",
    description="基于 RAG + 工具调用的知识库问答服务",
    version="1.0.0",
    lifespan=lifespan,
)


# ── 请求 / 响应模型 ──────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="用户提问")
    conversation_id: str | None = Field(None, description="会话 ID；不传则新建会话")
    user_id: str = Field("anonymous", description="用户标识，用于跨会话记忆")


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    citations: list[dict] = []
    retrieval: dict = {}
    tool_calls: list[dict] = []
    refused: bool = False


class SwitchRequest(BaseModel):
    provider: str = Field(..., description="deepseek / qwen / zhipu / ollama")


# ── 路由 ─────────────────────────────────────────────────────
@app.get("/health", summary="健康检查")
def health() -> dict:
    return {
        "status": "ok",
        "database": backend_name(),
        "knowledge_chunks": len(app.state.store),
        "packages": len(app.state.catalog.all()),
        "provider": app.state.llm.describe(),
    }


@app.post("/chat", response_model=ChatResponse, summary="对话（核心接口）")
def chat(req: ChatRequest) -> ChatResponse:
    try:
        answer = app.state.agent.respond(
            req.message, conversation_id=req.conversation_id, user_id=req.user_id
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"处理失败：{type(exc).__name__}: {exc}")
    return ChatResponse(**answer.to_dict())


@app.get("/conversations", summary="会话列表")
def conversations(user_id: str | None = None, limit: int = 50) -> list[dict]:
    return store.list_conversations(user_id, limit)


@app.get("/conversations/{conv_id}", summary="会话详情")
def conversation_detail(conv_id: str) -> dict:
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@app.get("/conversations/{conv_id}/messages", summary="加载历史消息")
def conversation_messages(conv_id: str, limit: int = 100) -> list[dict]:
    if not store.get_conversation(conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return store.load_messages_full(conv_id, limit)


@app.delete("/conversations/{conv_id}", summary="删除会话")
def conversation_delete(conv_id: str) -> dict:
    if not store.delete_conversation(conv_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"deleted": conv_id}


@app.get("/memories", summary="查看用户长期记忆")
def memories(user_id: str) -> dict:
    return store.get_memories(user_id)


@app.delete("/memories", summary="清空用户长期记忆")
def memories_clear(user_id: str) -> dict:
    return {"cleared": store.clear_memories(user_id)}


@app.get("/providers", summary="可用的模型提供方")
def providers() -> dict:
    return {
        "available": available_providers(),
        "current": app.state.llm.describe(),
        "provider_id": app.state.llm.provider,
    }


@app.post("/providers/switch", summary="运行时切换模型提供方")
def providers_switch(req: SwitchRequest) -> dict:
    try:
        llm = switch_provider(req.provider)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    rag = RAGPipeline(
        llm,
        app.state.store,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
        max_rewrite_rounds=settings.max_rewrite_rounds,
    )
    app.state.llm = llm
    app.state.rag = rag
    app.state.agent = SupportAgent(llm, rag, app.state.catalog)
    return {"switched_to": llm.describe(), "provider_id": llm.provider}


@app.get("/packages", summary="套餐列表")
def packages(keyword: str = "") -> list[dict]:
    return app.state.catalog.find(keyword)
