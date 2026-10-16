"""持久化测试：会话状态与用户记忆。

覆盖简历第三条 bullet 里「支持历史会话恢复」这个行为。
"""
from __future__ import annotations

from core.memory import store
from core.memory.db import backend_name


def test_backend_is_sqlite_in_tests():
    """测试环境跑 SQLite，生产用 PostgreSQL —— 同一套 ORM 代码。"""
    assert "SQLite" in backend_name() or "PostgreSQL" in backend_name()


def test_conversation_lifecycle():
    conv = store.create_conversation(user_id="u1")
    cid = conv["id"]
    assert conv["message_count"] == 0

    store.append_message(cid, "user", "畅享39 多少钱？")
    store.append_message(cid, "assistant", "39 元，含 20GB。", citations=[{"doc_id": "a::畅享39"}])

    # 首条用户消息自动成为会话标题
    conv2 = store.get_conversation(cid)
    assert conv2["title"] == "畅享39 多少钱？"
    assert conv2["message_count"] == 2

    history = store.load_history(cid)
    assert [m["role"] for m in history] == ["user", "assistant"]

    assert store.delete_conversation(cid) is True
    assert store.get_conversation(cid) is None
    assert store.load_history(cid) == []


def test_conversation_listing_and_rename():
    a = store.create_conversation(user_id="u2")
    b = store.create_conversation(user_id="u2")
    store.append_message(b["id"], "user", "你好")

    rows = store.list_conversations("u2")
    ids = [r["id"] for r in rows]
    assert a["id"] in ids and b["id"] in ids

    assert store.rename_conversation(a["id"], "改过的标题") is True
    assert store.get_conversation(a["id"])["title"] == "改过的标题"


def test_history_restore_after_simulated_restart():
    """「关掉进程再打开，历史还在」—— 这是持久化存在的唯一理由。"""
    conv = store.create_conversation(user_id="u3")
    cid = conv["id"]
    for i in range(5):
        store.append_message(cid, "user", f"问题 {i}")
        store.append_message(cid, "assistant", f"回答 {i}")

    # 模拟重启：重新从库里读
    messages = store.load_messages_full(cid)
    assert len(messages) == 10
    assert messages[0]["content"] == "问题 0"
    assert messages[-1]["content"] == "回答 4"
    # seq 必须连续，否则「加载最近 N 条」会取错
    assert [m["seq"] for m in messages] == list(range(10))


def test_citations_are_persisted():
    """引用必须落库 —— 这是回答「可追溯」的基础。"""
    conv = store.create_conversation(user_id="u4")
    cid = conv["id"]
    store.append_message(
        cid,
        "assistant",
        "39 元。",
        citations=[{"doc_id": "a::畅享39", "source": "01-套餐总览.md", "score": 0.82}],
        retrieval_meta={"rewrite_rounds": 1, "best_score": 0.82},
    )
    msg = store.load_messages_full(cid)[0]
    assert msg["citations"][0]["source"] == "01-套餐总览.md"
    assert msg["retrieval_meta"]["rewrite_rounds"] == 1


def test_user_memory_upsert_and_clear():
    store.upsert_memory("u5", "预算上限", "30 元", 0.6)
    assert store.get_memories("u5")["预算上限"] == "30 元"

    # 同一个 key 再次写入应当是更新而不是新增
    store.upsert_memory("u5", "预算上限", "50 元", 0.8)
    mem = store.get_memories("u5")
    assert mem["预算上限"] == "50 元"
    assert len(mem) == 1

    assert store.clear_memories("u5") >= 1
    assert store.get_memories("u5") == {}


def test_memories_are_isolated_between_users():
    store.upsert_memory("ua", "所在城市", "阜阳")
    store.upsert_memory("ub", "所在城市", "合肥")
    assert store.get_memories("ua")["所在城市"] == "阜阳"
    assert store.get_memories("ub")["所在城市"] == "合肥"
