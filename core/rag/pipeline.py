"""RAG 检索链路。

简历第二条 bullet 在这一层落地：
> 针对召回片段不相关导致的答非所问，加入文档相关性判断与多轮查询重写，
> 先判断问题是否落在知识库范围内、再改写追问，降低无效检索与模型幻觉。

## 为什么需要"先判断范围、再改写"这两步

朴素的 RAG 是「无脑检索 → 硬塞给模型」。它有两个典型失败：

1. **问题根本不在知识库范围内**（比如问"今天天气怎么样"），
   检索器仍会返回 top-k 个"最不相关"的片段（因为总有东西排前面），
   模型拿到这些无关片段后，要么答非所问，要么直接编。
   → 所以要在检索之前先做**范围判断**，不在范围内就直接拒答，
     这一次拒答省掉的不仅是一次无效检索，更是一次幻觉的机会。

2. **追问依赖上下文**（比如上一轮问"A 套餐多少钱"，这一轮只问"那 B 呢"）。
   拿"那 B 呢"去做检索，向量里几乎没有有效信息，召回必然很差。
   → 所以要**多轮查询重写**：用对话历史把追问补全成完整问题再检索。

## 一个容易忽略的成本问题

范围判断和多轮重写都会**额外调用模型**，所以它们不是免费的。
本项目用两个手段控制成本：
  - 重写只在"首轮检索结果质量不足"时才触发（而不是每轮都改写）
  - 重写有严格的轮数上限（max_rewrite_rounds），避免无限循环
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..llm import LLMClient
from .vectorstore import Hit, VectorStore


@dataclass
class RetrievalResult:
    """一次检索的完整记录，会落进会话的 retrieval_meta，便于回溯。"""

    in_scope: bool = True
    scope_reason: str = ""
    rewritten_query: str = ""
    rewrite_rounds: int = 0
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    refuse_message: str = ""
    best_score: float = 0.0

    def meta(self) -> dict:
        return {
            "in_scope": self.in_scope,
            "scope_reason": self.scope_reason,
            "rewritten_query": self.rewritten_query,
            "rewrite_rounds": self.rewrite_rounds,
            "hit_count": len(self.hits),
            "best_score": round(self.best_score, 4),
            "refused": self.refused,
        }

    def citations(self) -> list[dict]:
        return [h.citation() for h in self.hits]


SCOPE_PROMPT = """你在为一个「流量套餐与资费」客服系统做检索前置判断。

判断用户这句话是否属于「手机流量套餐、资费、办理方式、使用规则」这个范围。

判断规则：
- 属于：询问套餐内容、价格、流量额度、有效期、办理/退订方式、使用限制、对比推荐
- 不属于：闲聊、问天气、问其他行业的问题、要求写代码、与通信资费完全无关的话题
- 如果这句话是依赖上文的追问（例如「那它呢」「便宜的有没有」），只要上文在范围内，也算范围内

只输出一个 JSON 对象，不要任何其他文字：
{"in_scope": true 或 false, "reason": "一句话说明理由"}"""

REWRITE_PROMPT = """你在为一个流量套餐客服系统做查询改写。

用户的最新一句话可能是依赖上文的追问。请结合对话历史，把它改写成
**一句可以独立检索的完整问题**。

要求：
- 补全被省略的主语/对象（例如把「那它呢」补成「XX 套餐的资费和流量是多少」）
- 保留用户原意，不要添加用户没提到的条件
- 只输出改写后的那一句话，不要解释，不要加引号"""


class RAGPipeline:
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        top_k: int = 4,
        min_relevance: float = 0.12,
        max_rewrite_rounds: int = 2,
    ) -> None:
        self.llm = llm
        self.store = store
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.max_rewrite_rounds = max_rewrite_rounds

    # ── 第一步：范围判断 ──────────────────────────────────────
    def judge_scope(self, question: str, history: list[dict] | None = None) -> tuple[bool, str]:
        """问题是否落在知识库范围内。

        失败时**默认放行**（返回 True）——这是一个刻意的取舍：
        宁可多检索一次，也不要因为判断器抽风而把正常问题挡在门外。
        客服场景下「答不了」比「答得慢」严重得多。
        """
        convo = _render_history(history, limit=4)
        user = f"对话历史：\n{convo}\n\n用户最新一句：{question}" if convo else f"用户说：{question}"
        try:
            reply = self.llm.chat(
                [
                    {"role": "system", "content": SCOPE_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            data = _extract_json(reply.content)
            return bool(data.get("in_scope", True)), str(data.get("reason", ""))
        except Exception as exc:  # noqa: BLE001
            return True, f"范围判断失败，已放行（{type(exc).__name__}）"

    # ── 第二步：多轮查询重写 ─────────────────────────────────
    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        """把依赖上文的追问补全成可独立检索的问题。"""
        convo = _render_history(history, limit=6)
        if not convo:
            return question
        try:
            reply = self.llm.chat(
                [
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": f"对话历史：\n{convo}\n\n用户最新一句：{question}"},
                ],
                temperature=0.0,
            )
            text = (reply.content or "").strip().strip('"').strip("「」")
            # 改写结果异常（太长/为空）时退回原句，保证链路不会因为改写失败而停摆
            if not text or len(text) > 200:
                return question
            return text
        except Exception:  # noqa: BLE001
            return question

    # ── 第三步：检索 + 质量不足时重试 ────────────────────────
    def retrieve_with_rewrite(
        self,
        question: str,
        history: list[dict] | None = None,
        skip_scope_check: bool = False,
    ) -> RetrievalResult:
        result = RetrievalResult()

        # ① 范围判断：不在范围内直接拒答，不浪费一次检索
        if not skip_scope_check:
            in_scope, reason = self.judge_scope(question, history)
            result.in_scope = in_scope
            result.scope_reason = reason
            if not in_scope:
                result.refused = True
                result.refuse_message = (
                    "这个问题超出了我的业务范围。我主要负责流量套餐、资费、"
                    "办理方式和使用规则方面的咨询，你可以换个相关问题问我。"
                )
                return result

        # ② 先拿原句检索一次
        query = question
        hits = self.store.search(query, top_k=self.top_k, min_score=self.min_relevance)
        best = hits[0].score if hits else 0.0

        # ③ 质量不足才改写重试（而不是每次改写 —— 改写要花一次模型调用）
        rounds = 0
        while best < self.min_relevance and rounds < self.max_rewrite_rounds:
            rounds += 1
            new_query = self.rewrite_query(question, history)
            if new_query == query:
                break                      # 改写没有变化，再试也是白费
            query = new_query
            hits = self.store.search(query, top_k=self.top_k, min_score=self.min_relevance)
            best = hits[0].score if hits else 0.0

        result.rewritten_query = query if query != question else ""
        result.rewrite_rounds = rounds
        result.hits = hits
        result.best_score = best

        # ④ 改写后仍然找不到 → 承认不知道，而不是硬答
        if not hits:
            result.refused = True
            result.refuse_message = (
                "抱歉，我在知识库里没有找到与这个问题相关的内容。"
                "你可以换个说法，或者提供更具体的信息（比如套餐名称）。"
            )
        return result


# ── 辅助 ──────────────────────────────────────────────────────
def _render_history(history: list[dict] | None, limit: int = 4) -> str:
    if not history:
        return ""
    rows = history[-limit:]
    lines = []
    for m in rows:
        who = {"user": "用户", "assistant": "客服"}.get(m.get("role", ""), m.get("role", ""))
        lines.append(f"{who}：{m.get('content', '')[:200]}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON。

    模型经常在 JSON 外面包一层 ```json 代码块或者加一句解释，
    所以不能直接 json.loads，必须先把第一个 {...} 抠出来。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型没有返回可解析的 JSON：{text[:120]}")
    return json.loads(text[start : end + 1])
