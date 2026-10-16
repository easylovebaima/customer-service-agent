"""数据持久化层：会话状态与用户记忆。

简历里这句在这一层落地：
> 使用 PostgreSQL 持久化会话状态与用户记忆，支持历史会话恢复。

设计取舍（面试会被问）：
1. **用 SQLAlchemy ORM，而不是手写 SQL。** 因为「PostgreSQL 还是 SQLite」
   这件事对业务代码应当透明 —— 换库只改连接串，不改一行查询逻辑。
   本地开发用 SQLite 跑通全部逻辑，生产切 PostgreSQL，代码零改动。
2. **消息与会话分表。** 会话表存元信息（标题、时间、用户记忆），
   消息表存每轮对话。这样「加载历史会话」是一次索引查询，
   而不是把一整坨 JSON 读出来再解析。
3. **用户记忆单独一张表。** 记忆是跨会话的长期信息（偏好、常用套餐），
   与会话生命周期解耦 —— 会话可以删，记忆要留下。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session as OrmSession,
    mapped_column,
    relationship,
    sessionmaker,
)

from ..config import ROOT, settings


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    """一次会话。"""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True, default="anonymous")
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    # 会话级摘要：长对话压缩后留在这里，下次加载直接复用（省一次模型调用）
    summary: Mapped[str] = mapped_column(Text, default="")

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.seq",
    )


class Message(Base):
    """会话里的一条消息。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer, default=0)   # 在会话内的顺序
    role: Mapped[str] = mapped_column(String(16))          # user / assistant / system
    content: Mapped[str] = mapped_column(Text, default="")
    # 检索溯源：这一轮回答引用了哪些知识片段（JSON 数组）
    citations: Mapped[str] = mapped_column(Text, default="[]")
    # 检索链路的中间结果，便于排查「为什么答非所问」
    retrieval_meta: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class UserMemory(Base):
    """跨会话的长期记忆。"""

    __tablename__ = "user_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    key: Mapped[str] = mapped_column(String(64), index=True)   # 例如 常用套餐 / 所在城市
    value: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now
    )


# ── 引擎与会话工厂 ────────────────────────────────────────────
def _build_engine():
    url = settings.database_url
    try:
        engine = create_engine(url, pool_pre_ping=True, future=True)
        with engine.connect():
            pass
        return engine, url, False
    except Exception as exc:  # noqa: BLE001
        if not settings.allow_sqlite_fallback or url.startswith("sqlite"):
            raise
        # PostgreSQL 连不上时退回 SQLite，保证本地开发能跑通全部逻辑。
        # 注意：这是**降级**不是替代，生产环境仍应使用 PostgreSQL。
        fallback = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
        Path(ROOT / "data").mkdir(parents=True, exist_ok=True)
        print(
            f"[持久化] PostgreSQL 连接失败（{type(exc).__name__}），"
            f"已降级为 SQLite：{fallback}\n"
            f"         生产环境请启动 PostgreSQL 或设置 DATABASE_URL。"
        )
        return create_engine(fallback, future=True), fallback, True


engine, ACTIVE_URL, IS_FALLBACK = _build_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session() -> OrmSession:
    return SessionLocal()


def backend_name() -> str:
    if ACTIVE_URL.startswith("postgresql"):
        return "PostgreSQL"
    if ACTIVE_URL.startswith("sqlite"):
        return "SQLite（降级模式）"
    return ACTIVE_URL.split(":")[0]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


__all__ = [
    "Base",
    "Conversation",
    "Message",
    "UserMemory",
    "engine",
    "SessionLocal",
    "init_db",
    "get_session",
    "backend_name",
    "ACTIVE_URL",
    "IS_FALLBACK",
    "select",
    "json",
]
