"""pytest 全局配置。

注意：必须在导入 core 之前把数据库指向临时文件 ——
因为 core.config.settings 是模块级单例，import 时就固化了配置。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TEST_DB = Path(tempfile.gettempdir()) / "cs_agent_test.db"
if _TEST_DB.exists():
    _TEST_DB.unlink()

os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DB.as_posix()}")
os.environ.setdefault("ALLOW_SQLITE_FALLBACK", "1")
os.environ.setdefault("EMBEDDING_BACKEND", "local")
os.environ.setdefault("LLM_API_KEY", "test-key-not-used")

import pytest  # noqa: E402

from core.memory.db import init_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_db():
    init_db()
    yield


@pytest.fixture
def sandbox_kb(tmp_path):
    """一份最小知识库，避免测试依赖真实 data/ 目录。"""
    kb = tmp_path / "knowledge"
    kb.mkdir()
    (kb / "a.md").write_text(
        "# 套餐总览\n\n## 畅享39\n\n畅享39 月费 39 元，含 20GB 通用流量。\n",
        encoding="utf-8",
    )
    return kb
