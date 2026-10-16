"""抽象模型调用层。

简历里这句话在这一层落地：
> 抽象模型调用层，支持 DeepSeek、通义千问及本地 Ollama 切换。

关键设计：**一个统一接口 + 一组预置 + 运行时按配置切换。**

三家厂商的差异被压缩成三个字段（base_url / model / 是否需要 key），
而这个函数的调用方（RAG 链路、Agent 编排、HTTP API、Gradio 界面）
完全感知不到背后用的是哪一家。

要新增一家厂商，只需要在 PROVIDER_PRESETS 里加一行 —— 不改任何上层代码。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator

from .config import settings


@dataclass
class ToolCall:
    id: str
    name: str
    # 保留原始 JSON 字符串：参数不是合法 JSON 也是一种需要回灌给模型的信息
    arguments: str

    def parsed(self) -> dict:
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass
class LLMReply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    provider: str = ""
    model: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# 四个预置：三个云端 + 一个本地。全部走 OpenAI 兼容协议，
# 所以本地 Ollama 和云端厂商共用同一套调用代码。
PROVIDER_PRESETS: dict[str, dict] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "needs_key": True,
    },
    "qwen": {
        "label": "通义千问",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "needs_key": True,
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "needs_key": True,
    },
    "ollama": {
        "label": "本地 Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b",
        "needs_key": False,
    },
}


class LLMClient:
    """统一的模型客户端。"""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
    ) -> None:
        from openai import OpenAI  # 延迟导入，方便单测替换

        self.provider = (provider or settings.llm_provider).lower()
        preset = PROVIDER_PRESETS.get(self.provider)
        if preset is None:
            raise ValueError(
                f"未知的模型提供方 '{self.provider}'，可选：{list(PROVIDER_PRESETS)}"
            )

        self.model = model or settings.llm_model or preset["model"]
        self.base_url = base_url or settings.llm_base_url or preset["base_url"]
        self.temperature = (
            temperature if temperature is not None else settings.llm_temperature
        )

        key = api_key or settings.llm_api_key
        if preset["needs_key"] and not key:
            raise RuntimeError(
                f"{preset['label']} 需要 API Key，请在 .env 里设置 LLM_API_KEY"
            )
        # 本地 Ollama 不校验 key，但 SDK 要求非空
        self._client = OpenAI(api_key=key or "ollama", base_url=self.base_url)

    # ── 主调用 ────────────────────────────────────────────────
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
    ) -> LLMReply:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
            )
            for tc in (getattr(msg, "tool_calls", None) or [])
        ]

        usage_obj = getattr(resp, "usage", None)
        usage = (
            {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
            }
            if usage_obj
            else {}
        )

        return LLMReply(
            content=msg.content or "",
            tool_calls=tool_calls,
            usage=usage,
            provider=self.provider,
            model=self.model,
        )

    # ── 流式（Gradio 界面用）────────────────────────────────
    def stream(self, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield delta.content

    # ── 给 LangChain 链用的适配 ──────────────────────────────
    def as_langchain(self):
        """返回等价的 LangChain ChatModel。

        RAG 的提示词编排用 LangChain 的 Runnable 组织，
        底层仍然走同一个 base_url / model —— 保证两条路径行为一致。
        """
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.model,
            api_key=self._client.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def describe(self) -> str:
        preset = PROVIDER_PRESETS.get(self.provider, {})
        return f"{preset.get('label', self.provider)} / {self.model}"


_default_client: LLMClient | None = None


def build_llm(**kwargs: Any) -> LLMClient:
    global _default_client
    if not kwargs and _default_client is not None:
        return _default_client
    client = LLMClient(**kwargs)
    if not kwargs:
        _default_client = client
    return client


def switch_provider(provider: str, **kwargs: Any) -> LLMClient:
    """运行时切换模型提供方。"""
    global _default_client
    _default_client = LLMClient(provider=provider, **kwargs)
    return _default_client


def available_providers() -> list[dict]:
    return [
        {"id": key, "label": val["label"], "model": val["model"]}
        for key, val in PROVIDER_PRESETS.items()
    ]
