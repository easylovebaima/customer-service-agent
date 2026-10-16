#!/usr/bin/env python
"""从 .env.example 生成 .env，并把已有的 API Key 填进去。

为什么不手写 .env：密钥不该出现在命令行参数或聊天记录里。
这个脚本只从本地文件读取并把长度打印出来，不输出密钥内容。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = [
    Path(r"D:\AI-Agent-学习\02-练习代码\key.txt"),   # 练习目录里的 key
    ROOT / "key.txt",
]


def find_key() -> str:
    for path in CANDIDATES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip().strip('"').strip("'")
            if s and not s.startswith("#"):
                return s
    return ""


def main() -> int:
    key = find_key()
    if not key:
        print("没有找到可用的 API Key，请手动编辑 .env")
        return 1

    tpl = (ROOT / ".env.example").read_text(encoding="utf-8")
    out = ROOT / ".env"
    out.write_text(tpl.replace("在这里粘贴你的key", key), encoding="utf-8")

    print(f"已生成：{out}")
    print(f"key 长度 {len(key)} 字符，格式校验：{'通过（含点号）' if '.' in key else '未通过（智谱 key 应形如 xxx.yyy）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
