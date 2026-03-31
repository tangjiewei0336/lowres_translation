"""
将 OPUS CCMatrix 指定有向语言对转为 LLaMA-Factory 监督微调常用 **Alpaca** JSONL。

每行: {"instruction": "...", "input": "<源句>", "output": "<目标句>"}
（可选 "system"；可选写入 score 元数据列，见 CLI）

LLaMA-Factory 说明: https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it


from lowres_translation.ccmatrix_download import (
    _ensure_zip_downloaded,
    iter_ccmatrix_parallel,
    opus_archive_pair_id,
)

# CCMatrix / OPUS 使用 ISO 639-1 等短代码；仅用于 instruction 可读性，不全亦可回退为代码本身
CCMATRIX_LANG_NAMES_EN: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "vi": "Vietnamese",
    "th": "Thai",
    "tl": "Tagalog",
    "fil": "Filipino",
    "ind": "Indonesian",
    "id": "Indonesian",
    "ms": "Malay",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "ur": "Urdu",
    "bn": "Bengali",
    "ta": "Tamil",
    "af": "Afrikaans",
}


def _lang_label(code: str) -> str:
    c = code.strip().lower()
    return CCMATRIX_LANG_NAMES_EN.get(c, c)


def _parse_directed_pair(spec: str) -> Tuple[str, str, str]:
    """
    用户输入有向语对，如 en-zh 表示 英译中（源 en，目标 zh）。
    返回 (src_iso, tgt_iso, opus_zip_pair_id)。
    """
    spec = spec.strip().lower()
    if "-" not in spec:
        raise ValueError(f"语对格式应为 src-tgt，例如 en-zh，当前: {spec!r}")
    a, b = spec.split("-", 1)
    a, b = a.strip(), b.strip()
    if not a or not b:
        raise ValueError(f"无效语对: {spec!r}")
    if a == b:
        raise ValueError(f"源与目标不能相同: {spec!r}")
    return a, b, opus_archive_pair_id(a, b)


def _score_value(score_str: str) -> Optional[float]:
    try:
        return float(score_str.strip())
    except Exception:
        return None


def _alpaca_record(
    *,
    src_text: str,
    tgt_text: str,
    src_code: str,
    tgt_code: str,
    instruction_template: str,
    system: Optional[str],
    include_score: bool,
    score_raw: str,
    score_val: Optional[float],
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "instruction": instruction_template.format(
            src_code=src_code,
            tgt_code=tgt_code,
            src_name=_lang_label(src_code),
            tgt_name=_lang_label(tgt_code),
        ),
        "input": src_text,
        "output": tgt_text,
    }
    if system:
        rec["system"] = system
    if include_score:
        rec["score"] = score_val if score_val is not None else score_raw
    return rec


def convert_directed_pair(
    *,
    directed_spec: str,
    zip_cache_dir: Path,
    max_samples: Optional[int],
    min_score: Optional[float],
    instruction_template: str,
    system: Optional[str],
    include_score: bool,
) -> Iterator[Dict[str, Any]]:
    src, tgt, opus_id = _parse_directed_pair(directed_spec)
    zip_path = _ensure_zip_downloaded(opus_id, str(zip_cache_dir))
    n = 0
    for s_line, t_line, sc in iter_ccmatrix_parallel(zip_path, opus_id, src, tgt):
        if not s_line or not t_line:
            continue
        sv = _score_value(sc)
        if min_score is not None and sv is not None and sv < min_score:
            continue
        if min_score is not None and sv is None:
            continue
        yield _alpaca_record(
            src_text=s_line,
            tgt_text=t_line,
            src_code=src,
            tgt_code=tgt,
            instruction_template=instruction_template,
            system=system,
            include_score=include_score,
            score_raw=sc,
            score_val=sv,
        )
        n += 1
        if max_samples is not None and n >= max_samples:
            break


DEFAULT_INSTRUCTION = (
    "Translate the following text from {src_name} to {tgt_name}. "
    "Output only the translation, without notes or explanations."
)


def _safe_dataset_key(pair_spec: str) -> str:
    return pair_spec.replace("-", "_")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="将 CCMatrix（OPUS zip）指定有向语言对导出为 LLaMA-Factory Alpaca JSONL。"
    )
    ap.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="逗号分隔的有向语对，如 en-zh,zh-en（前者为源语，后者为目标语）",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default="llamafactory_data/ccmatrix_mt",
        help="输出目录（将写入 jsonl 与 dataset_info 片段）",
    )
    ap.add_argument(
        "--zip-cache-dir",
        type=str,
        default="ccmatrix_zip_cache",
        help="OPUS zip 缓存目录（默认项目下 ccmatrix_zip_cache）",
    )
    ap.add_argument(
        "--max-samples-per-pair",
        type=int,
        default=None,
        help="每个有向语对最多写入多少条（默认不限制，数据量可能极大）",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="仅保留 score 不低于该值的句对（score 无法解析为浮点的行将被丢弃）",
    )
    ap.add_argument(
        "--instruction-template",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Alpaca instruction 模板，可用占位符 {src_name} {tgt_name} {src_code} {tgt_code}",
    )
    ap.add_argument(
        "--system",
        type=str,
        default=None,
        help='可选 system 字段（如 "You are a professional translator."）',
    )
    ap.add_argument(
        "--include-score",
        action="store_true",
        help="每行额外写入 score 字段；若不在 LLaMA-Factory 的 columns 里映射，一般会被忽略",
    )
    ap.add_argument(
        "--merge",
        action="store_true",
        help="将所有语对合并为一个 merged_alpaca.jsonl（否则每个语对一个文件）",
    )
    ap.add_argument(
        "--root",
        type=str,
        default=None,
        help="项目根目录（用于解析相对 zip-cache / out-dir；默认从本脚本推断）",
    )
    args = ap.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    zip_cache = (
        (root / args.zip_cache_dir).resolve()
        if not Path(args.zip_cache_dir).is_absolute()
        else Path(args.zip_cache_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_cache.mkdir(parents=True, exist_ok=True)

    pair_list = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pair_list:
        print("错误: --pairs 不能为空", file=sys.stderr)
        return 2

    dataset_entries: Dict[str, Dict[str, Any]] = {}
    merge_path = out_dir / "ccmatrix_merged_alpaca.jsonl"
    merge_f = None
    try:
        if args.merge:
            merge_f = open(merge_path, "w", encoding="utf-8")

        for spec in pair_list:
            try:
                _parse_directed_pair(spec)
            except ValueError as e:
                print(f"错误: {e}", file=sys.stderr)
                return 2

            out_name = f"ccmatrix_{_safe_dataset_key(spec)}_alpaca.jsonl"
            out_path = out_dir / out_name
            n_written = 0
            with open(out_path, "w", encoding="utf-8") as f:
                it = convert_directed_pair(
                    directed_spec=spec,
                    zip_cache_dir=zip_cache,
                    max_samples=args.max_samples_per_pair,
                    min_score=args.min_score,
                    instruction_template=args.instruction_template,
                    system=args.system,
                    include_score=args.include_score,
                )
                for rec in tqdm(it, desc=f"转换 {spec}", unit="行"):
                    line = json.dumps(rec, ensure_ascii=False) + "\n"
                    f.write(line)
                    if merge_f is not None:
                        merge_f.write(line)
                    n_written += 1

            key = f"ccmatrix_{_safe_dataset_key(spec)}"
            dataset_entries[key] = {"file_name": out_name, "formatting": "alpaca"}
            print(f"  {spec}: {n_written} 条 -> {out_path}")

        snippet_path = out_dir / "dataset_info.snippet.json"
        with open(snippet_path, "w", encoding="utf-8") as sf:
            json.dump(dataset_entries, sf, ensure_ascii=False, indent=2)
        print(f"已写入 LLaMA-Factory dataset_info 片段: {snippet_path}")
        print("请将片段合并到你的 LLaMA-Factory data/dataset_info.json 中，训练时使用 dataset: <键名>。")

        if args.merge:
            print(f"已合并写入: {merge_path}")
            merged_snip = out_dir / "dataset_info.merged_snippet.json"
            with open(merged_snip, "w", encoding="utf-8") as sf:
                json.dump(
                    {"ccmatrix_merged": {"file_name": merge_path.name, "formatting": "alpaca"}},
                    sf,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"合并数据集的 dataset_info 片段: {merged_snip}")

    except KeyboardInterrupt:
        print("已中断", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"失败: {e}", file=sys.stderr)
        raise
    finally:
        if merge_f is not None and not merge_f.closed:
            merge_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
