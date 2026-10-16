"""全局配置。

所有可调项集中在这里，通过环境变量覆盖，代码里不出现任何硬编码的密钥或连接串。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """极简 .env 解析：不引入 python-dotenv 也能跑（装了则优先用它）。"""
    path = path or ENV_FILE
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            data[k.strip()] = v
    return data


_FILE_ENV = read_env_file()


def env(key: str, default: str = "") -> str:
    return os.getenv(key) or _FILE_ENV.get(key) or default


@dataclass
class Settings:
    # ── 模型 ──
    # provider 决定走哪个后端：deepseek / qwen / ollama / zhipu
    llm_provider: str = field(default_factory=lambda: env("LLM_PROVIDER", "zhipu"))
    llm_model: str = field(default_factory=lambda: env("LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: env("LLM_API_KEY", ""))
    llm_base_url: str = field(default_factory=lambda: env("LLM_BASE_URL", ""))
    llm_temperature: float = field(default_factory=lambda: float(env("LLM_TEMPERATURE", "0.2")))

    # ── 数据库 ──
    # 生产用 PostgreSQL；本地无 PG 时可切成 sqlite 跑通全部逻辑
    database_url: str = field(
        default_factory=lambda: env(
            "DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/cs_agent"
        )
    )
    allow_sqlite_fallback: bool = field(
        default_factory=lambda: env("ALLOW_SQLITE_FALLBACK", "1") == "1"
    )

    # ── 检索 ──
    embedding_backend: str = field(default_factory=lambda: env("EMBEDDING_BACKEND", "local"))
    top_k: int = field(default_factory=lambda: int(env("RETRIEVAL_TOP_K", "4")))
    min_relevance: float = field(default_factory=lambda: float(env("MIN_RELEVANCE", "0.12")))
    max_rewrite_rounds: int = field(default_factory=lambda: int(env("MAX_REWRITE_ROUNDS", "2")))
    index_path: str = field(default_factory=lambda: env("INDEX_PATH", str(ROOT / "data" / "index.json")))

    # ── 服务 ──
    host: str = field(default_factory=lambda: env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(env("PORT", "8000")))

    @property
    def knowledge_dir(self) -> Path:
        return ROOT / "data" / "knowledge"

    @property
    def packages_file(self) -> Path:
        return ROOT / "data" / "packages.json"


settings = Settings()


def describe() -> str:
    """给启动日志用的配置摘要（密钥打码）。"""
    key = settings.llm_api_key
    masked = f"{key[:6]}…{key[-4:]}" if len(key) > 12 else ("（未设置）" if not key else "***")
    return (
        f"模型提供方 = {settings.llm_provider}\n"
        f"数据库     = {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}\n"
        f"嵌入后端   = {settings.embedding_backend}\n"
        f"检索 top_k = {settings.top_k}，相关性阈值 = {settings.min_relevance}\n"
        f"密钥       = {masked}"
    )
