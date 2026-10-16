"""会话与记忆的读写接口。

上层（Agent 编排、HTTP API、Gradio）只跟这里打交道，不直接碰 ORM。
好处是：换数据库、加缓存、改表结构，都只影响这一个文件。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime

from sqlalchemy import delete, desc, select

from .db import Conversation, Message, UserMemory, get_session


# ── 会话 ──────────────────────────────────────────────────────
def create_conversation(user_id: str = "anonymous", title: str = "新会话") -> dict:
    conv_id = uuid.uuid4().hex[:12]
    with get_session() as s:
        conv = Conversation(id=conv_id, user_id=user_id, title=title)
        s.add(conv)
        s.commit()
        return _conv_dict(conv)


def get_conversation(conv_id: str) -> dict | None:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        return _conv_dict(conv) if conv else None


def list_conversations(user_id: str | None = None, limit: int = 50) -> list[dict]:
    with get_session() as s:
        stmt = select(Conversation).order_by(desc(Conversation.updated_at)).limit(limit)
        if user_id:
            stmt = stmt.where(Conversation.user_id == user_id)
        return [_conv_dict(c) for c in s.scalars(stmt)]


def rename_conversation(conv_id: str, title: str) -> bool:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        if not conv:
            return False
        conv.title = title[:200]
        s.commit()
        return True


def delete_conversation(conv_id: str) -> bool:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        if not conv:
            return False
        s.delete(conv)          # cascade 会一并删掉消息
        s.commit()
        return True


# ── 消息 ──────────────────────────────────────────────────────
def append_message(
    conv_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
    retrieval_meta: dict | None = None,
) -> dict:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        if conv is None:
            raise KeyError(f"会话不存在：{conv_id}")

        msg = Message(
            conversation_id=conv_id,
            seq=conv.message_count,
            role=role,
            content=content,
            citations=json.dumps(citations or [], ensure_ascii=False),
            retrieval_meta=json.dumps(retrieval_meta or {}, ensure_ascii=False),
        )
        s.add(msg)
        conv.message_count = conv.message_count + 1
        conv.updated_at = datetime.now()

        # 首条用户消息顺手当标题，省得用户手动命名
        if role == "user" and conv.title == "新会话":
            conv.title = content.strip()[:40] or "新会话"

        s.commit()
        return _msg_dict(msg)


def load_history(conv_id: str, limit: int = 20) -> list[dict]:
    """按时间顺序取最近 limit 条，转成模型可用的 messages 格式。"""
    with get_session() as s:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(desc(Message.seq))
            .limit(limit)
        )
        rows = list(s.scalars(stmt))
    rows.reverse()
    return [{"role": m.role, "content": m.content} for m in rows]


def load_messages_full(conv_id: str, limit: int = 100) -> list[dict]:
    with get_session() as s:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.seq)
            .limit(limit)
        )
        return [_msg_dict(m) for m in s.scalars(stmt)]


def set_summary(conv_id: str, summary: str) -> None:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        if conv:
            conv.summary = summary
            s.commit()


def get_summary(conv_id: str) -> str:
    with get_session() as s:
        conv = s.get(Conversation, conv_id)
        return conv.summary if conv else ""


# ── 用户记忆（跨会话）──────────────────────────────────────────
def upsert_memory(user_id: str, key: str, value: str, confidence: float = 0.6) -> None:
    with get_session() as s:
        stmt = select(UserMemory).where(
            UserMemory.user_id == user_id, UserMemory.key == key
        )
        row = s.scalars(stmt).first()
        if row is None:
            s.add(
                UserMemory(
                    user_id=user_id, key=key, value=value, confidence=confidence
                )
            )
        else:
            row.value = value
            row.confidence = confidence
            row.updated_at = datetime.now()
        s.commit()


def get_memories(user_id: str) -> dict[str, str]:
    with get_session() as s:
        stmt = select(UserMemory).where(UserMemory.user_id == user_id)
        return {m.key: m.value for m in s.scalars(stmt)}


def clear_memories(user_id: str) -> int:
    with get_session() as s:
        result = s.execute(delete(UserMemory).where(UserMemory.user_id == user_id))
        s.commit()
        return result.rowcount or 0


# ── 内部转换 ──────────────────────────────────────────────────
def _conv_dict(c: Conversation) -> dict:
    return {
        "id": c.id,
        "user_id": c.user_id,
        "title": c.title,
        "message_count": c.message_count,
        "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": c.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _msg_dict(m: Message) -> dict:
    try:
        citations = json.loads(m.citations or "[]")
    except json.JSONDecodeError:
        citations = []
    try:
        meta = json.loads(m.retrieval_meta or "{}")
    except json.JSONDecodeError:
        meta = {}
    return {
        "id": m.id,
        "seq": m.seq,
        "role": m.role,
        "content": m.content,
        "citations": citations,
        "retrieval_meta": meta,
        "created_at": m.created_at.strftime("%Y-%m-%d %H:%M:%S"),
    }
