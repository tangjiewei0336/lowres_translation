#!/usr/bin/env python3
"""从 OPUS 下载 CCMatrix 各语言对 zip，并写出每对前 N 条预览 jsonl。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowres_translation.ccmatrix_download import main  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    raise SystemExit(main())
