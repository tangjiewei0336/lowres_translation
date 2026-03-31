"""FLORES-200 official tarball download, extract, and line loading."""

from __future__ import annotations

import os
import random
import tarfile
import urllib.request
from pathlib import Path

from lowres_translation.lang_names import LANG_NAMES

FLORES200_TAR = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"


def get_lang_name(code: str) -> str:
    """获取语言显示名，未知则返回 code。"""
    return LANG_NAMES.get(code, code.replace("_", " "))


def parse_config(config: str) -> tuple[str, str, str, str]:
    """解析 config 如 'eng_Latn-zho_Hans' 得到 (src, tgt, src_col, tgt_col)。"""
    parts = config.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"config 应为 'src_tgt' 形式，例如 eng_Latn-zho_Hans，当前: {config}")
    src, tgt = parts[0].strip(), parts[1].strip()
    return src, tgt, f"sentence_{src}", f"sentence_{tgt}"


def load_flores200_from_tarball(config: str, split: str, verbose: bool = True) -> tuple[list[str], list[str]]:
    """从官方 tarball 加载语对，返回 (sources, references)。"""
    src_lang, tgt_lang, _, _ = parse_config(config)
    cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "flores200"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "flores200_dataset"
    src_file = extract_dir / split / f"{src_lang}.{split}"
    tgt_file = extract_dir / split / f"{tgt_lang}.{split}"

    if not src_file.exists() or not tgt_file.exists():
        tar_path = cache_dir / "flores200_dataset.tar.gz"
        if not tar_path.exists():
            if verbose:
                print(f"下载 FLORES-200: {FLORES200_TAR}")
            urllib.request.urlretrieve(FLORES200_TAR, tar_path)
        if verbose:
            print("解压 FLORES-200 ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            if hasattr(tarfile, "data_filter"):
                tf.extractall(path=cache_dir, filter=tarfile.data_filter)
            else:
                tf.extractall(path=cache_dir)

    def read_lines(p: Path) -> list[str]:
        with open(p, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    sources = read_lines(src_file)
    references = read_lines(tgt_file)
    if len(sources) != len(references):
        raise RuntimeError(f"FLORES-200 行数不一致: {src_file} {len(sources)} vs {tgt_file} {len(references)}")
    return sources, references


def load_sources_references(
    config: str,
    split: str,
    limit: int | None = None,
    random_sample: bool = False,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[list[str], list[str]]:
    """
    加载 FLORES-200 源句与参考译文。始终使用官方 tarball。
    limit: 只取/采样条数；random_sample: 为 True 时从全量中随机采样 limit 条（否则取前 limit 条）；seed: 随机种子，可复现。
    """
    sources, references = load_flores200_from_tarball(config, split, verbose=verbose)
    if limit is not None and limit > 0:
        n = len(sources)
        if random_sample:
            if seed is not None:
                random.seed(seed)
            k = min(limit, n)
            idx = random.sample(range(n), k)
            sources = [sources[i] for i in idx]
            references = [references[i] for i in idx]
            if verbose:
                print(f"  随机采样 {k} 条 (seed={seed})")
        else:
            sources = sources[:limit]
            references = references[:limit]
    return sources, references


def ensure_flores200_downloaded(
    split: str = "dev",
    sample_lang: str = "eng_Latn",
    verbose: bool = True,
) -> Path:
    """
    仅下载并解压官方 FLORES-200 tarball（不跑翻译）。
    通过检查某语言 split 文件是否存在来判断是否已就绪。
    """
    cache_dir = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "flores200"
    cache_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = cache_dir / "flores200_dataset"
    sample = extract_dir / split / f"{sample_lang}.{split}"
    if sample.exists():
        if verbose:
            print(f"FLORES-200 已就绪: {sample}")
        return cache_dir

    tar_path = cache_dir / "flores200_dataset.tar.gz"
    if not tar_path.exists():
        if verbose:
            print(f"下载 FLORES-200: {FLORES200_TAR}")
        urllib.request.urlretrieve(FLORES200_TAR, tar_path)
    if verbose:
        print("解压 FLORES-200 ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(path=cache_dir, filter=tarfile.data_filter)
        else:
            tf.extractall(path=cache_dir)
    if verbose:
        print(f"完成。缓存目录: {cache_dir}")
    return cache_dir
