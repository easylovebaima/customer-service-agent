"""轻量向量库。

为什么不直接上 Chroma / FAISS？
因为本项目的知识库规模是「几十到几百个片段」这个量级，
在这个量级下，**暴力余弦相似度就足够快**，引入一个带持久化、
带索引构建的向量库反而是额外负担。

这里的实现和 LangChain 的 Retriever 接口语义一致（top_k + score + metadata），
真要换向量库时，把 search() 换掉即可，上层不用动。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from .embedding import cosine, get_embedder


@dataclass
class Hit:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    def citation(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.metadata.get("source", ""),
            "section": self.metadata.get("section", ""),
            "score": round(self.score, 4),
            "preview": self.text[:120],
        }


class VectorStore:
    def __init__(self, embedder: Callable[[str], list[float]] | None = None) -> None:
        self._embed = embedder or get_embedder()
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vecs: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, doc_id: str, text: str, metadata: dict | None = None) -> None:
        if not text.strip():
            return
        self._ids.append(doc_id)
        self._texts.append(text)
        self._metas.append(metadata or {})
        self._vecs.append(self._embed(text))

    def add_many(self, items: Iterable[tuple[str, str, dict]]) -> int:
        before = len(self)
        for doc_id, text, meta in items:
            self.add(doc_id, text, meta)
        return len(self) - before

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[Hit]:
        if not self._ids:
            return []
        qv = self._embed(query)
        scored = [
            Hit(
                doc_id=self._ids[i],
                text=self._texts[i],
                metadata=self._metas[i],
                score=cosine(qv, self._vecs[i]),
            )
            for i in range(len(self._ids))
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:top_k] if h.score >= min_score]

    # ── 持久化：索引直接存 JSON，便于人工查看与排查 ──
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": len(self._vecs[0]) if self._vecs else 0,
            "items": [
                {
                    "doc_id": self._ids[i],
                    "text": self._texts[i],
                    "metadata": self._metas[i],
                    "vector": self._vecs[i],
                }
                for i in range(len(self._ids))
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path, embedder=None) -> "VectorStore":
        path = Path(path)
        store = cls(embedder=embedder)
        if not path.exists():
            return store
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            store._ids.append(item["doc_id"])
            store._texts.append(item["text"])
            store._metas.append(item.get("metadata") or {})
            store._vecs.append(item["vector"])
        return store

    def stats(self) -> dict:
        return {
            "chunks": len(self._ids),
            "dim": len(self._vecs[0]) if self._vecs else 0,
            "sources": sorted({m.get("source", "?") for m in self._metas}),
        }
