"""FLORES-200 evaluation and CCMatrix download helpers (submodules load lazily)."""

import os

# 默认走 HF 国内镜像；若需官方 Hub，请先 export HF_ENDPOINT=https://huggingface.co
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
