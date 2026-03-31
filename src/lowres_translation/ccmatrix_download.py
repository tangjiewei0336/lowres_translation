"""Download CCMatrix language-pair zips from OPUS and write preview jsonl (no datasets script)."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import requests

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):  # type: ignore[misc]
        return x

DEFAULT_PAIRS_URL = "https://hf-mirror.com/datasets/yhavinga/ccmatrix/raw/main/language_pairs_cache.py"
OPUS_DOWNLOAD_URL = "https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/{}.txt.zip"
OPUS_FILE = "CCMatrix.{}.{}"


def _pairs_cache_url() -> str:
    return os.environ.get("CCMATRIX_PAIRS_URL", DEFAULT_PAIRS_URL)


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:200] if len(name) > 200 else name


def _fetch_language_pairs_from_hf(timeout_s: int = 60) -> List[Tuple[str, str]]:
    resp = requests.get(_pairs_cache_url(), timeout=timeout_s)
    resp.raise_for_status()
    text = resp.text
    m = re.search(r'string\s*=\s*"""(.*?)"""', text, re.S)
    if not m:
        raise RuntimeError("无法从 language_pairs_cache.py 解析语言对 string")
    pairs: List[Tuple[str, str]] = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        a, b = line.split("-", 1)
        pairs.append((a, b))
    return pairs


def _all_configs(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    configs = []
    for a, b in pairs:
        configs.append(f"{a}-{b}")
        configs.append(f"{b}-{a}")
    return sorted(set(configs))


def undirected_pair_ids(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """去重后的无向语对 id（与 OPUS zip 一致：两语码按字典序），如 en-zh。"""
    ids: set[str] = set()
    for a, b in pairs:
        x, y = sorted((a.strip(), b.strip()))
        ids.add(f"{x}-{y}")
    return sorted(ids)


def _download_pair_for_config(cfg: str) -> str:
    a, b = cfg.split("-", 1)
    x, y = (a, b) if a < b else (b, a)
    return f"{x}-{y}"


def _read_first_n_from_zip(
    zip_path: str,
    download_pair: str,
    lang1: str,
    lang2: str,
    n: int,
) -> List[Dict[str, Any]]:
    l1_name = OPUS_FILE.format(download_pair, lang1)
    l2_name = OPUS_FILE.format(download_pair, lang2)
    s_name = OPUS_FILE.format(download_pair, "scores")

    items: List[Dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(l1_name, "r") as f1, zf.open(l2_name, "r") as f2, zf.open(s_name, "r") as f3:
            t1 = io.TextIOWrapper(f1, encoding="utf-8", errors="replace", newline="")
            t2 = io.TextIOWrapper(f2, encoding="utf-8", errors="replace", newline="")
            ts = io.TextIOWrapper(f3, encoding="utf-8", errors="replace", newline="")
            for idx, (x, y, s) in enumerate(zip(t1, t2, ts)):
                if idx >= n:
                    break
                score_str = s.strip()
                try:
                    score: Any = float(score_str)
                except Exception:
                    score = score_str
                items.append(
                    {
                        "id": idx,
                        "score": score,
                        "translation": {lang1: x.rstrip("\n").strip(), lang2: y.rstrip("\n").strip()},
                    }
                )
    return items


def _ensure_zip_downloaded(download_pair: str, zip_cache_dir: str, timeout_s: int = 300) -> str:
    os.makedirs(zip_cache_dir, exist_ok=True)
    zip_path = os.path.join(zip_cache_dir, f"{download_pair}.txt.zip")
    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
        return zip_path

    url = OPUS_DOWNLOAD_URL.format(download_pair)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        tmp_path = zip_path + ".partial"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        os.replace(tmp_path, zip_path)
    return zip_path


def opus_archive_pair_id(lang_a: str, lang_b: str) -> str:
    """OPUS zip 文件名中的语言对，两语代码按字典序排列，如 en-zh。"""
    x, y = sorted([lang_a.strip(), lang_b.strip()])
    return f"{x}-{y}"


def iter_ccmatrix_parallel(
    zip_path: str,
    download_pair: str,
    lang_src: str,
    lang_tgt: str,
):
    """
    流式迭代 CCMatrix moses zip 中的平行句与分数行。
    产出 (源句, 目标句, score 原始字符串)，不整包读入内存。
    """
    l_src = OPUS_FILE.format(download_pair, lang_src)
    l_tgt = OPUS_FILE.format(download_pair, lang_tgt)
    s_name = OPUS_FILE.format(download_pair, "scores")
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(l_src, "r") as f1, zf.open(l_tgt, "r") as f2, zf.open(s_name, "r") as f3:
            t1 = io.TextIOWrapper(f1, encoding="utf-8", errors="replace", newline="")
            t2 = io.TextIOWrapper(f2, encoding="utf-8", errors="replace", newline="")
            ts = io.TextIOWrapper(f3, encoding="utf-8", errors="replace", newline="")
            for a, b, s in zip(t1, t2, ts):
                yield a.rstrip("\n").strip(), b.rstrip("\n").strip(), s.strip()


def load_download_pair_configs(path: str | Path) -> List[str]:
    """
    从 JSON 配置文件读取要下载/预览的有向语对列表（与 CCMatrix config 一致，如 en-zh、zh-en）。
    支持字段：
    - pairs: ["en-zh", "zh-en", ...]
    - language_pairs: [["en","zh"], ...] 或 [{"src":"en","tgt":"zh"}, ...]
    二者可同时出现，会合并后按出现顺序去重。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("CCMatrix 下载配置文件顶层必须是 JSON 对象")

    def _norm_pair(a: str, b: str) -> str:
        x, y = a.strip().lower(), b.strip().lower()
        if not x or not y or x == y:
            raise ValueError(f"无效语言代码对: {(a, b)!r}")
        return f"{x}-{y}"

    raw: List[str] = []

    pl = data.get("pairs")
    if pl is not None:
        if not isinstance(pl, list):
            raise ValueError('"pairs" 必须是字符串数组')
        for item in pl:
            s = str(item).strip().lower()
            if "-" not in s:
                raise ValueError(f'"pairs" 中每项应为 src-tgt，如 en-zh，当前: {item!r}')
            a, b = s.split("-", 1)
            raw.append(_norm_pair(a, b))

    lp = data.get("language_pairs")
    if lp is not None:
        if not isinstance(lp, list):
            raise ValueError('"language_pairs" 必须是数组')
        for item in lp:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                raw.append(_norm_pair(str(item[0]), str(item[1])))
            elif isinstance(item, dict) and "src" in item and "tgt" in item:
                raw.append(_norm_pair(str(item["src"]), str(item["tgt"])))
            else:
                raise ValueError(f"无效的 language_pairs 项: {item!r}")

    if not raw:
        raise ValueError('配置中需包含非空的 "pairs" 或 "language_pairs"')

    seen: set[str] = set()
    out: List[str] = []
    for c in raw:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def preview_one_config(
    config: str,
    max_examples: int,
    out_dir: str,
    split: str = "train",
) -> Tuple[bool, str]:
    if split != "train":
        return False, "该数据集只有 train split（OPUS 文件不区分 split）"

    try:
        lang1, lang2 = config.split("-", 1)
        download_pair = _download_pair_for_config(config)
        zip_cache_dir = os.path.join(out_dir, "_opus_zip_cache")
        zip_path = _ensure_zip_downloaded(download_pair, zip_cache_dir=zip_cache_dir)
        items = _read_first_n_from_zip(zip_path, download_pair, lang1, lang2, max_examples)
        if not items:
            return False, "该语言对数据为空（前 N 条为空）"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{_safe_filename(config)}__top{max_examples}__{stamp}.jsonl"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in items:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        return True, out_path
    except Exception as e:
        return False, repr(e)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 OPUS 下载 CCMatrix 各语言对 zip，并写出每对前 N 条预览 jsonl。"
    )
    ap.add_argument("--out_dir", default="ccmatrix_preview", help="输出目录")
    ap.add_argument("--max_examples", type=int, default=50, help="每个语言对预览条数")
    ap.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="JSON 配置文件路径：列出要下载预览的有向语对（见 ccmatrix_download_config.example.json）。"
        "若指定则不再从 Hub 拉全量列表，且忽略 --limit_configs。",
    )
    ap.add_argument("--limit_configs", type=int, default=0, help="只处理前 K 个语言对（0=全部；与 --config-file 互斥）")
    ap.add_argument("--skip_existing", action="store_true", help="跳过已有该 config 前缀的 jsonl")
    ap.add_argument("--split", default="train", help="仅支持 train")
    ap.add_argument(
        "--list-pairs",
        action="store_true",
        help="从 Hub 拉取语对缓存，列出全部可用语对后退出（不写预览文件）",
    )
    ap.add_argument(
        "--list-directed",
        action="store_true",
        help="与 --list-pairs 合用：列出有向语对（每个无向对拆成 a-b 与 b-a）；默认仅列出无向 id（与 OPUS zip 名一致）",
    )
    ap.add_argument(
        "--pairs-output",
        type=str,
        default=None,
        help="与 --list-pairs 合用：将列表写入该文件（UTF-8），标准输出仍打印统计到 stderr",
    )
    args = ap.parse_args()

    if args.list_pairs:
        if args.list_directed:
            try:
                raw = _fetch_language_pairs_from_hf()
            except Exception as e:
                print(f"拉取语对列表失败: {e}", file=sys.stderr)
                return 1
            lines = _all_configs(raw)
            kind = "有向"
        else:
            try:
                raw = _fetch_language_pairs_from_hf()
            except Exception as e:
                print(f"拉取语对列表失败: {e}", file=sys.stderr)
                return 1
            lines = undirected_pair_ids(raw)
            kind = "无向（OPUS zip）"
        text = "\n".join(lines) + ("\n" if lines else "")
        sys.stdout.write(text)
        if args.pairs_output:
            Path(args.pairs_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.pairs_output).write_text(text, encoding="utf-8")
            print(f"已写入 {args.pairs_output}", file=sys.stderr)
        print(f"# {kind}语对共 {len(lines)} 条（来源: {_pairs_cache_url()}）", file=sys.stderr)
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    if args.config_file:
        try:
            configs = load_download_pair_configs(args.config_file)
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"读取配置文件失败: {e}", file=sys.stderr)
            return 2
        print(f"[ccmatrix] 使用配置文件 {args.config_file}，共 {len(configs)} 个有向语对。")
    else:
        pairs = _fetch_language_pairs_from_hf()
        configs = _all_configs(pairs)
        if args.limit_configs and args.limit_configs > 0:
            configs = configs[: args.limit_configs]

    done_prefixes = set()
    if args.skip_existing:
        try:
            for fn in os.listdir(args.out_dir):
                if "__" in fn:
                    done_prefixes.add(fn.split("__", 1)[0])
        except Exception:
            pass

    failures_path = os.path.join(args.out_dir, "failures.log")
    ok_count = 0
    fail_count = 0

    with open(failures_path, "a", encoding="utf-8") as flog:
        for cfg in tqdm(configs, desc="处理语言对", unit="pair"):
            if args.skip_existing and _safe_filename(cfg) in done_prefixes:
                continue

            ok, msg = preview_one_config(
                config=cfg,
                max_examples=args.max_examples,
                out_dir=args.out_dir,
                split=args.split,
            )
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                flog.write(f"[{datetime.now().isoformat()}] {cfg}\t{msg}\n")
                flog.flush()

    print(f"完成：成功 {ok_count} 个语言对，失败 {fail_count} 个语言对。")
    print(f"输出目录：{os.path.abspath(args.out_dir)}")
    print(f"失败日志：{os.path.abspath(failures_path)}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
