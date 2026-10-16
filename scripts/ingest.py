#!/usr/bin/env python
"""知识库入库脚本。

把 data/knowledge/*.md 切片、向量化、写入索引文件。

用法：
    .venv/Scripts/python.exe scripts/ingest.py
    .venv/Scripts/python.exe scripts/ingest.py --rebuild      # 强制重建

切片策略（为什么这么切）：
1. **先按 Markdown 二级标题分节**，而不是无脑按字数切。
   因为知识库的每一节本身就是语义完整的单元（例如「如何退订」），
   从中间切断会让检索到的片段缺头少尾。
2. **节内再按段落聚合到目标长度**（约 400 字）。
   太短的片段信息量不足，太长的片段会稀释向量、降低区分度。
3. **保留来源与章节名作为元数据**，这样回答时能给出引用出处，
   排查「为什么答非所问」时也能立刻知道是哪段被召回了。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from core.config import settings
from core.rag.vectorstore import VectorStore

TARGET_CHARS = 400
MAX_CHARS = 800


def split_markdown(text: str, source: str) -> list[tuple[str, str, dict]]:
    """把一篇 Markdown 切成若干片段，返回 (doc_id, chunk, metadata)。"""
    chunks: list[tuple[str, str, dict]] = []

    # 一级标题作为文档标题
    lines = text.splitlines()
    doc_title = ""
    for ln in lines:
        if ln.startswith("# "):
            doc_title = ln[2:].strip()
            break

    # 按二级标题切节
    sections: list[tuple[str, list[str]]] = []
    current_title = doc_title
    current_body: list[str] = []
    for ln in lines:
        if ln.startswith("## "):
            if current_body:
                sections.append((current_title, current_body))
            current_title = ln[3:].strip()
            current_body = []
        elif ln.startswith("# "):
            continue
        else:
            current_body.append(ln)
    if current_body:
        sections.append((current_title, current_body))

    for section_title, body_lines in sections:
        # 按空行切成段落，再把段落聚合成目标长度
        paragraphs = [p.strip() for p in "\n".join(body_lines).split("\n\n") if p.strip()]
        buffer = ""
        index = 0
        for para in paragraphs:
            if buffer and len(buffer) + len(para) > TARGET_CHARS:
                index += 1
                chunks.append(
                    (
                        f"{source}::{section_title}::{index}",
                        buffer.strip(),
                        {"source": source, "section": section_title, "doc_title": doc_title},
                    )
                )
                buffer = para
            else:
                buffer = f"{buffer}\n{para}" if buffer else para

            # 单段过长就硬切
            while len(buffer) > MAX_CHARS:
                head, buffer = buffer[:MAX_CHARS], buffer[MAX_CHARS:]
                index += 1
                chunks.append(
                    (
                        f"{source}::{section_title}::{index}",
                        head.strip(),
                        {"source": source, "section": section_title, "doc_title": doc_title},
                    )
                )

        if buffer.strip():
            index += 1
            chunks.append(
                (
                    f"{source}::{section_title}::{index}",
                    buffer.strip(),
                    {"source": source, "section": section_title, "doc_title": doc_title},
                )
            )

    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="知识库入库")
    parser.add_argument("--rebuild", action="store_true", help="忽略已有索引，全量重建")
    parser.add_argument("--knowledge", default=str(settings.knowledge_dir))
    parser.add_argument("--index", default=str(settings.index_path))
    args = parser.parse_args(argv)

    kb_dir = Path(args.knowledge)
    if not kb_dir.exists():
        print(f"知识库目录不存在：{kb_dir}")
        return 2

    files = sorted(kb_dir.glob("*.md"))
    if not files:
        print(f"{kb_dir} 下没有 .md 文件")
        return 2

    store = VectorStore()
    total_chunks = 0
    for f in files:
        chunks = split_markdown(f.read_text(encoding="utf-8"), f.name)
        store.add_many(chunks)
        total_chunks += len(chunks)
        print(f"  {f.name:<24} → {len(chunks):>3} 个片段")

    index_path = store.save(args.index)
    print()
    print(f"入库完成：{len(files)} 个文档，{total_chunks} 个片段")
    print(f"索引文件：{index_path}")
    print(f"嵌入后端：{settings.embedding_backend}（维度 {len(store._vecs[0]) if len(store) else 0}）")

    # 自检：跑几条典型查询，确认索引真的能召回东西
    print()
    print("检索自检：")
    for q in ["畅享39 多少钱", "怎么退订套餐", "学生有优惠吗", "超出流量怎么收费"]:
        hits = store.search(q, top_k=2)
        top = f"{hits[0].metadata.get('source')}·{hits[0].metadata.get('section')}（{hits[0].score:.3f}）" if hits else "无命中"
        print(f"  {q:<20} → {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
