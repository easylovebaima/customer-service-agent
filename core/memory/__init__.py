"""持久化层：会话状态与用户记忆。"""
from .db import backend_name, engine, init_db
from .store import (
    append_message,
    clear_memories,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_memories,
    list_conversations,
    load_history,
    load_messages_full,
    rename_conversation,
    upsert_memory,
)

__all__ = [
    "init_db",
    "backend_name",
    "engine",
    "create_conversation",
    "get_conversation",
    "list_conversations",
    "rename_conversation",
    "delete_conversation",
    "append_message",
    "load_history",
    "load_messages_full",
    "upsert_memory",
    "get_memories",
    "clear_memories",
]
