"""文本向量化。

默认后端是**本地实现**（字符 n-gram + 特征哈希），理由很简单：
它零依赖、完全离线、可复现，而且对「流量包名称、资费数字、套餐关键词」
这类词面匹配为主的场景效果足够。

要换成真正的语义嵌入，只需实现一个 embed() 函数并改配置 ——
上层检索代码一行都不用动。这就是把嵌入单独抽一层的意义。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable

DIM = 512

_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[a-zA-Z0-9_.%]+")


def _tokenize(text: str) -> list[str]:
    """中英混合分词。

    中文取「单字 + 相邻双字」（bigram），因为中文里「流量」和「流」+「量」
    表达的信息量差别很大，只按单字会丢掉组合语义；
    英文与数字按词切，并保留百分号、小数点这类对资费表述有意义的后缀。
    """
    text = text.lower()
    tokens: list[str] = []

    # 中文：单字 + 双字组合
    cjk_chars = _CJK.findall(text)
    tokens.extend(cjk_chars)
    for i in range(len(cjk_chars) - 1):
        tokens.append(cjk_chars[i] + cjk_chars[i + 1])

    # 英文/数字
    tokens.extend(_WORD.findall(text))

    return tokens


def _hash_index(token: str) -> int:
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % DIM


def embed_local(text: str) -> list[float]:
    """本地嵌入：词频 + 特征哈希，最后做 L2 归一化。"""
    vec = [0.0] * DIM
    tokens = _tokenize(text)
    if not tokens:
        return vec

    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1

    for token, cnt in counts.items():
        # 次线性缩放：避免高频词（如「的」「是」）主导整个向量
        vec[_hash_index(token)] += 1.0 + math.log(cnt)

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def embed_api(text: str) -> list[float]:
    """调用厂商 embedding 接口（以智谱 embedding-3 为例）。

    默认不启用：它需要额外的 API 调用与额度。
    想用的时候把 EMBEDDING_BACKEND 设为 api 即可。
    """
    from ..config import settings

    from openai import OpenAI

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.embeddings.create(model="embedding-3", input=text)
    return list(resp.data[0].embedding)


_BACKENDS: dict[str, Callable[[str], list[float]]] = {
    "local": embed_local,
    "api": embed_api,
}


def get_embedder(name: str | None = None) -> Callable[[str], list[float]]:
    from ..config import settings

    key = (name or settings.embedding_backend).lower()
    if key not in _BACKENDS:
        raise ValueError(f"未知的嵌入后端 '{key}'，可选：{list(_BACKENDS)}")
    return _BACKENDS[key]


def cosine(a: list[float], b: list[float]) -> float:
    """两个已归一化向量的余弦相似度就是点积。"""
    if len(a) != len(b):
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
    return sum(x * y for x, y in zip(a, b))
