"""Agent 编排：把检索、工具、会话、记忆串成一次回答。

简历第三条 bullet 在这一层落地：
> 基于 LangChain 组织模型调用、工具调用与 RAG 检索步骤；
> 使用 PostgreSQL 持久化会话状态与用户记忆，支持历史会话恢复；
> 抽象模型调用层，支持 DeepSeek、通义千问及本地 Ollama 切换。

## 一次回答经过哪些步骤

    读历史 + 读记忆
      → 范围判断（不在范围内直接拒答，省掉检索与生成）
      → 多轮查询重写 + 检索
      → 组装 Prompt（LangChain 模板）
      → 模型生成（可调用工具）
      → 工具执行回灌 → 再生成
      → 落库（消息 + 引用 + 检索元数据）
      → 更新长期记忆

## LangChain 在这里做什么

用来组织 Prompt 与步骤编排（ChatPromptTemplate + 消息序列），
底层 HTTP 调用仍走统一的 LLMClient —— 这样"换厂商"和"换编排方式"
两件事互不干扰。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.prompts import ChatPromptTemplate

from ..llm import LLMClient, ToolCall
from ..memory import store
from ..rag.pipeline import RAGPipeline, RetrievalResult
from .tools import TOOL_SCHEMAS, PackageCatalog, execute_tool

MAX_TOOL_ROUNDS = 3

SYSTEM_TEMPLATE = """你是某运营商「流量套餐」业务的在线客服。

回答要求：
1. 只依据下面提供的【知识库片段】和【工具返回结果】作答，不要凭常识补充。
2. 如果这些材料不足以回答，直接说明你不知道，并建议用户换个说法或提供更具体的信息。
   绝对不要编造套餐名称、价格、流量额度或有效期。
3. 涉及推荐、比价、筛选时，优先调用工具获取准确结果，不要自己心算。
4. 语气简洁亲切，不要输出 markdown 表格，用自然语言分点说明即可。
5. 调用推荐工具时，只传用户明确说出的条件。「流量尽量多」是一种偏好而不是硬性门槛，
   不要把「尽量」翻译成一个具体数字去做筛选，否则会把本来合适的套餐全部筛掉。
6. 工具返回空结果时，不要就此结束 —— 应当放宽条件（去掉你自己加上的限制）重试一次。

【用户画像】
{memories}

【知识库片段】
{context}
"""

USER_TEMPLATE = "{question}"


@dataclass
class Answer:
    content: str
    conversation_id: str
    citations: list[dict] = field(default_factory=list)
    retrieval: dict = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    refused: bool = False
    usage: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.content,
            "conversation_id": self.conversation_id,
            "citations": self.citations,
            "retrieval": self.retrieval,
            "tool_calls": self.tool_calls,
            "refused": self.refused,
        }


class SupportAgent:
    def __init__(
        self,
        llm: LLMClient,
        rag: RAGPipeline,
        catalog: PackageCatalog | None = None,
    ) -> None:
        self.llm = llm
        self.rag = rag
        self.catalog = catalog or PackageCatalog()
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_TEMPLATE), ("user", USER_TEMPLATE)]
        )

    # ── 主入口 ────────────────────────────────────────────────
    def respond(
        self,
        question: str,
        conversation_id: str | None = None,
        user_id: str = "anonymous",
    ) -> Answer:
        conv_id = conversation_id or store.create_conversation(user_id)["id"]
        history = store.load_history(conv_id, limit=10)
        memories = store.get_memories(user_id)

        store.append_message(conv_id, "user", question)

        # ① 检索链路（含范围判断与多轮重写）
        retrieval = self.rag.retrieve_with_rewrite(question, history)

        # ② 范围外 → 直接拒答，不生成、不调用工具
        if retrieval.refused:
            answer = Answer(
                content=retrieval.refuse_message,
                conversation_id=conv_id,
                citations=[],
                retrieval=retrieval.meta(),
                refused=True,
            )
            store.append_message(
                conv_id, "assistant", answer.content, citations=[], retrieval_meta=retrieval.meta()
            )
            return answer

        # ③ 组装提示词并生成（可带工具）
        context = _render_context(retrieval.hits)
        prompt_text = self._prompt.format_prompt(
            memories=_render_memories(memories),
            context=context,
            question=question,
        ).to_string()

        messages: list[dict] = [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": question},
        ]

        tool_log: list[dict] = []
        usage: dict = {}
        content = ""

        for _ in range(MAX_TOOL_ROUNDS):
            reply = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            usage = reply.usage or usage

            if not reply.wants_tools:
                content = reply.content
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": tc.arguments},
                        }
                        for tc in reply.tool_calls
                    ],
                }
            )
            for tc in reply.tool_calls:
                result = execute_tool(tc.name, tc.arguments, self.catalog)
                tool_log.append(
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_preview": result[:300],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )
        else:
            # 工具轮数用尽仍未收敛：给出兜底话术，而不是把空字符串返回给用户
            content = content or "抱歉，我在查询套餐时遇到了问题，请稍后再试或转人工客服。"

        answer = Answer(
            content=content,
            conversation_id=conv_id,
            citations=retrieval.citations(),
            retrieval=retrieval.meta(),
            tool_calls=tool_log,
            usage=usage,
        )

        # ④ 落库：消息 + 引用 + 检索元数据（引用与元数据是「可追溯」的关键）
        store.append_message(
            conv_id,
            "assistant",
            answer.content,
            citations=answer.citations,
            retrieval_meta=retrieval.meta(),
        )

        # ⑤ 顺手更新长期记忆（预算偏好这类信息跨会话有用）
        self._update_memories(user_id, question, retrieval)
        return answer

    # ── 记忆抽取 ──────────────────────────────────────────────
    def _update_memories(self, user_id: str, question: str, retrieval: RetrievalResult) -> None:
        """从用户话里抽取可长期复用的信息。

        这里用规则而不是模型：规则便宜、可预测、不会把用户没说的话"记"下来。
        真要上模型抽取，也应当只在规则命中失败时才启用。
        """
        q = question.lower()
        for p in self.catalog.all():
            if p["name"].lower() in q:
                store.upsert_memory(user_id, "关注套餐", p["name"], 0.7)
                break

        import re

        m = re.search(r"(\d+)\s*(?:元|块)", question)
        if m:
            store.upsert_memory(user_id, "预算上限", f"{m.group(1)} 元", 0.6)

        m = re.search(r"(\d+)\s*(?:个|)G", question, re.IGNORECASE)
        if m:
            store.upsert_memory(user_id, "流量需求", f"{m.group(1)}GB", 0.6)

        for city in ("阜阳", "合肥", "北京", "上海", "广州", "深圳"):
            if city in question:
                store.upsert_memory(user_id, "所在城市", city, 0.8)
                break


# ── 渲染辅助 ──────────────────────────────────────────────────
def _render_context(hits: list) -> str:
    if not hits:
        return "（没有检索到相关片段）"
    blocks = []
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("source", "未知来源")
        sec = h.metadata.get("section", "")
        title = f"{src}" + (f" · {sec}" if sec else "")
        blocks.append(f"[片段 {i}] 来源：{title}（相似度 {h.score:.3f}）\n{h.text}")
    return "\n\n".join(blocks)


def _render_memories(memories: dict[str, str]) -> str:
    if not memories:
        return "（暂无历史记录）"
    return "\n".join(f"- {k}：{v}" for k, v in memories.items())
