#!/usr/bin/env python3
"""将 CCMatrix 导出为 LLaMA-Factory Alpaca JSONL（入口脚本）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowres_translation.ccmatrix_llamafactory import main

if __name__ == "__main__":
    raise SystemExit(main())
