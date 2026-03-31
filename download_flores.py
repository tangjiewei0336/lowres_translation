#!/usr/bin/env python3
"""仅下载并解压官方 FLORES-200 tarball（不跑翻译）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowres_translation.flores_dataset import ensure_flores200_downloaded


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 FLORES-200 官方数据包到 Hugging Face 缓存目录")
    ap.add_argument("--split", type=str, default="dev", choices=["dev", "devtest"], help="用于检查是否已解压的文件")
    ap.add_argument(
        "--sample-lang",
        type=str,
        default="eng_Latn",
        help="用于检查是否已解压的语言代码（需与 split 对应文件存在）",
    )
    args = ap.parse_args()
    ensure_flores200_downloaded(split=args.split, sample_lang=args.sample_lang, verbose=True)


if __name__ == "__main__":
    main()
