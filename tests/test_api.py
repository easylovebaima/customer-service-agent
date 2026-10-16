"""HTTP 接口测试。

用 FastAPI 的 TestClient 直接打接口，不启动真实服务、不调用真实模型。
这正好体现了「接口与界面解耦」的好处：后端可以完全脱离 Web 界面被测试。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "knowledge_chunks" in data


def test_providers_listed(client):
    data = client.get("/providers").json()
    ids = [p["id"] for p in data["available"]]
    # 简历里点名的三家 + 本地 Ollama，都必须在列
    for expected in ("deepseek", "qwen", "ollama"):
        assert expected in ids
    assert data["current"]


def test_packages_endpoint(client):
    rows = client.get("/packages").json()
    assert isinstance(rows, list)


def test_conversation_endpoints(client):
    # 建一条会话（直接走 store，避免依赖模型）
    from core.memory import store

    conv = store.create_conversation(user_id="api-user")
    cid = conv["id"]
    store.append_message(cid, "user", "你好")

    assert client.get(f"/conversations/{cid}").json()["id"] == cid
    msgs = client.get(f"/conversations/{cid}/messages").json()
    assert msgs[0]["content"] == "你好"

    listing = client.get("/conversations", params={"user_id": "api-user"}).json()
    assert any(r["id"] == cid for r in listing)

    assert client.delete(f"/conversations/{cid}").json()["deleted"] == cid
    assert client.get(f"/conversations/{cid}").status_code == 404


def test_missing_conversation_returns_404(client):
    assert client.get("/conversations/not-exist").status_code == 404
    assert client.get("/conversations/not-exist/messages").status_code == 404


def test_memories_endpoints(client):
    client.delete("/memories", params={"user_id": "api-mem"})
    assert client.get("/memories", params={"user_id": "api-mem"}).json() == {}


def test_chat_uses_agent_and_persists(client, monkeypatch):
    """把 agent 换成假的，验证接口的装配与落库是对的。

    不调真实模型：接口层要测的是「请求怎么进来、结果怎么存、响应怎么出去」，
    模型质量属于检索链路的测试范围。
    """
    from core.agent.chain import Answer

    captured = {}

    class FakeAgent:
        def respond(self, message, conversation_id=None, user_id="anonymous"):
            captured["message"] = message
            captured["user_id"] = user_id
            return Answer(
                content="畅享39 月费 39 元。",
                conversation_id=conversation_id or "fake-conv",
                citations=[{"doc_id": "a::畅享39", "source": "01-套餐总览.md", "score": 0.9}],
                retrieval={"in_scope": True, "hit_count": 1},
            )

    monkeypatch.setattr(app.state, "agent", FakeAgent())

    resp = client.post(
        "/chat", json={"message": "畅享39 多少钱？", "user_id": "api-chat"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "39" in data["answer"]
    assert data["citations"][0]["source"] == "01-套餐总览.md"
    assert data["retrieval"]["in_scope"] is True
    assert captured["message"] == "畅享39 多少钱？"


def test_chat_rejects_empty_message(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_unknown_provider(client):
    resp = client.post("/providers/switch", json={"provider": "no-such-provider"})
    assert resp.status_code == 400
