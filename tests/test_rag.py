"""检索链路测试：范围判断、多轮查询重写、检索命中。

这一组测试覆盖的正是简历第二条 bullet 描述的机制，
每条用例对应其中一个行为，跑通即代表该机制真的生效。
"""
from __future__ import annotations

from core.llm import LLMReply
from core.rag.pipeline import RAGPipeline, _extract_json
from core.rag.vectorstore import VectorStore


class ScriptedLLM:
    """按脚本返回响应的假模型。

    用它而不是真模型，是因为这一层测的是**链路逻辑**（什么时候改写、
    什么时候放弃），不是模型本身的质量。逻辑必须是确定的。
    """

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=None):
        self.calls.append(messages)
        if self._replies:
            text = self._replies.pop(0)
        else:
            text = self._replies_default() if hasattr(self, "_replies_default") else "{}"
        return LLMReply(content=text, provider="fake", model="fake")


def build_store() -> VectorStore:
    store = VectorStore()
    store.add_many(
        [
            (
                "a::畅享39",
                "畅享39 月费 39 元，含 20GB 通用流量，有效期 30 天。",
                {"source": "01-套餐总览.md", "section": "畅享39"},
            ),
            (
                "a::退订",
                "月付套餐可以随时退订，退订后当月剩余流量仍可使用至月末。",
                {"source": "02-办理与退订.md", "section": "如何退订"},
            ),
            (
                "a::超量",
                "畅享19 和畅享39 超出后按 3 元/GB 计费，当月流量费用最高 30 元封顶。",
                {"source": "03-常见问题.md", "section": "超出套餐流量怎么收费"},
            ),
        ]
    )
    return store


# ── 工具函数 ──────────────────────────────────────────────────
def test_extract_json_handles_code_fence():
    """模型经常在 JSON 外面包一层代码块，必须能抠出来。"""
    assert _extract_json('```json\n{"in_scope": true, "reason": "ok"}\n```')["in_scope"] is True
    assert _extract_json('前面有废话 {"in_scope": false, "reason": "闲聊"} 后面也有')["in_scope"] is False


# ── 范围判断 ──────────────────────────────────────────────────
def test_out_of_scope_is_refused_without_retrieval():
    """问题超出范围时应当直接拒答，且**不浪费一次检索**。"""
    llm = ScriptedLLM(['{"in_scope": false, "reason": "问的是天气，与资费无关"}'])
    rag = RAGPipeline(llm, build_store())

    result = rag.retrieve_with_rewrite("今天天气怎么样？")

    assert result.in_scope is False
    assert result.refused is True
    assert result.hits == []              # 没有检索
    assert len(llm.calls) == 1            # 只花了一次模型调用（范围判断本身）
    assert "超出" in result.refuse_message or "范围" in result.refuse_message


def test_in_scope_proceeds_to_retrieval():
    llm = ScriptedLLM(['{"in_scope": true, "reason": "询问套餐资费"}'])
    rag = RAGPipeline(llm, build_store())

    result = rag.retrieve_with_rewrite("畅享39 多少钱")

    assert result.in_scope is True
    assert result.refused is False
    assert result.hits, "应当命中知识库"
    assert result.best_score > 0


def test_scope_judge_failure_defaults_to_allow():
    """判断器失败时默认放行 —— 客服场景下「答不了」比「答得慢」严重。"""
    llm = ScriptedLLM(["这不是 JSON"])          # 模型输出无法解析
    rag = RAGPipeline(llm, build_store())

    result = rag.retrieve_with_rewrite("畅享39 多少钱")

    assert result.in_scope is True               # 失败也放行
    assert result.hits


# ── 查询重写 ──────────────────────────────────────────────────
def test_rewrite_triggers_when_first_pass_misses():
    """首轮召回质量不足时才改写 —— 改写要花一次模型调用，不能每轮都做。"""
    llm = ScriptedLLM(
        [
            '{"in_scope": true, "reason": "追问"}',      # 范围判断
            "畅享39 套餐的资费和流量是多少",                # 查询重写
        ]
    )
    rag = RAGPipeline(llm, build_store())

    result = rag.retrieve_with_rewrite(
        "那它呢？",
        history=[{"role": "user", "content": "畅享39 怎么样"}, {"role": "assistant", "content": "..."}],
    )

    assert result.rewrite_rounds >= 1
    assert result.rewritten_query != ""
    assert result.hits, "改写之后应当能命中"
    assert len(llm.calls) == 2


def test_no_rewrite_when_first_pass_is_good():
    """首轮就召回得很好时，不应触发改写（省一次模型调用）。"""
    llm = ScriptedLLM(['{"in_scope": true, "reason": "资费问题"}'])
    rag = RAGPipeline(llm, build_store())

    result = rag.retrieve_with_rewrite("畅享39 月费多少")

    assert result.rewrite_rounds == 0
    assert result.rewritten_query == ""
    assert len(llm.calls) == 1


def test_rewrite_capped_and_then_refuses():
    """改写轮数用尽仍无命中 → 承认不知道，而不是硬答。"""
    # 范围判断放行，之后每次改写都返回同样的话（触发 break），且知识库里没有对应内容
    llm = ScriptedLLM(
        [
            '{"in_scope": true, "reason": "看似相关"}',
            "完全不相关的查询词 zzz",
        ]
    )
    rag = RAGPipeline(llm, build_store(), max_rewrite_rounds=2, min_relevance=0.5)

    result = rag.retrieve_with_rewrite("量子计算的退相干时间是多久")

    assert result.refused is True
    assert "没有找到" in result.refuse_message


# ── 检索本身 ──────────────────────────────────────────────────
def test_vectorstore_hits_expected_section():
    store = build_store()
    hits = store.search("超出流量怎么收费", top_k=1)
    assert hits
    assert hits[0].metadata["section"] == "超出套餐流量怎么收费"


def test_vectorstore_roundtrip(tmp_path):
    store = build_store()
    path = store.save(tmp_path / "index.json")
    loaded = VectorStore.load(path)

    assert len(loaded) == len(store)
    a = store.search("退订", top_k=1)
    b = loaded.search("退订", top_k=1)
    assert a[0].doc_id == b[0].doc_id
    assert abs(a[0].score - b[0].score) < 1e-9
