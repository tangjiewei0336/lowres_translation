#!/usr/bin/env python3
"""
FLORES-200 两组互相翻译：中文/英文 选一组 × 西/印尼/越南/泰/菲律宾 选一组，双向。
共 2×5×2 = 20 个有向语对。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from flores200_single import run_single_evaluation
from tqdm import tqdm

# 组1：中文/英文 二选一（实际跑两组，即 2）
GROUP_ZH_EN = [
    ("英文", "eng_Latn"),
    ("中文", "zho_Hans"),
]
# 组2：西/印尼/越南/泰/菲律宾 五选一（实际跑五组，即 5）
GROUP_OTHER = [
    ("西班牙语", "spa_Latn"),
    ("印尼语", "ind_Latn"),
    ("越南语", "vie_Latn"),
    ("泰国语", "tha_Thai"),
    ("菲律宾语", "tgl_Latn"),
]


def _build_pairs() -> list[tuple[str, str, str, str]]:
    """两组互相翻译、双向：(组1×组2)×2 方向 = 2×5×2 = 20 个有向语对。"""
    names_a = {code: name for name, code in GROUP_ZH_EN}
    names_b = {code: name for name, code in GROUP_OTHER}
    names = {**names_a, **names_b}
    pairs = []
    for (_n1, c1) in GROUP_ZH_EN:
        for (_n2, c2) in GROUP_OTHER:
            pairs.append((c1, c2, names[c1], names[c2]))
            pairs.append((c2, c1, names[c2], names[c1]))
    return pairs


EVAL_PAIRS = _build_pairs()


def main() -> None:
    parser = argparse.ArgumentParser(description="FLORES-200 两组互相翻译（中/英 × 西/印尼/越/泰/菲，双向 2×5×2=20）")
    parser.add_argument("--base-url", type=str, default="http://localhost:8005/v1", help="vLLM API 地址")
    parser.add_argument("--model", type=str, default=None, help="模型名，不填则自动检测")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "devtest"])
    parser.add_argument("--limit", type=int, default=None, help="每语对评估条数（不填则全量）")
    parser.add_argument("--random-sample", action="store_true", help="与 --limit 同用：随机采样 N 条")
    parser.add_argument("--seed", type=int, default=None, help="随机采样种子，便于复现")
    parser.add_argument("--replicates", type=int, default=1, help="每个有向语对的重复运行次数（默认 1，共 2×5×2=20 组）")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=16, help="翻译请求并发数")
    parser.add_argument("--output-dir", type=str, default="eval_multilingual", help="各语对 JSON 输出目录")
    parser.add_argument("--summary", type=str, default=None, help="汇总 JSON 路径")
    parser.add_argument("--verbose", action="store_true", help="每个语对输出详细调试信息")
    args = parser.parse_args()

    # 2×5×2 语对，可选每对重复 replicates 次
    n_replicates = max(1, int(args.replicates))
    expanded = [
        (src_code, tgt_code, src_name, tgt_name, run_id)
        for (src_code, tgt_code, src_name, tgt_name) in EVAL_PAIRS
        for run_id in range(n_replicates)
    ]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(expanded)

    code_to_name = {code: name for name, code in GROUP_ZH_EN + GROUP_OTHER}
    codes = [code for _, code in GROUP_ZH_EN + GROUP_OTHER]
    checkpoint_path = Path(args.summary) if args.summary else (out_dir / "summary_latest.json")

    print(f"[调试] 两组互相翻译 2×5×2={len(EVAL_PAIRS)} 组 × 重复={n_replicates} → {total}  输出目录={out_dir}  定期保存={checkpoint_path}")

    results = []
    for idx, (src_code, tgt_code, src_name, tgt_name, run_id) in enumerate(tqdm(expanded, desc="语对评估", unit="组")):
        config = f"{src_code}-{tgt_code}"
        safe_name = config.replace("_", "-")
        output_path = out_dir / (f"{safe_name}_run{run_id}.json" if n_replicates > 1 else f"{safe_name}.json")
        t0 = time.perf_counter()
        res = run_single_evaluation(
            config=config,
            base_url=args.base_url,
            model=args.model,
            split=args.split,
            limit=args.limit,
            random_sample=args.random_sample,
            seed=args.seed,
            max_tokens=args.max_tokens,
            output_path=output_path,
            verbose=args.verbose,
            show_progress=True,  # 每个语对都显示“已完成请求数”进度条
            max_workers=args.max_workers,
        )
        elapsed = time.perf_counter() - t0
        one = {
            "config": config,
            "run_id": run_id,
            "bleu_score": res.get("bleu_score"),
            "comet_score": res.get("comet_score"),
            "num_samples": res.get("num_samples"),
            "src_name": src_name,
            "tgt_name": tgt_name,
            "src_code": src_code,
            "tgt_code": tgt_code,
        }
        if res.get("error"):
            one["error"] = res["error"]
        results.append(one)
        score = one.get("bleu_score")
        if not args.verbose:
            if score is not None:
                tqdm.write(f"  [{idx + 1}/{total}] {src_name} → {tgt_name} (run{run_id}): BLEU {score:.2f}  ({elapsed:.1f}s)")
            else:
                tqdm.write(f"  [{idx + 1}/{total}] {src_name} → {tgt_name} (run{run_id}): 失败 {one.get('error', 'unknown')}")

        # 定期保存：每完成一个语对就写入检查点
        table = {src: {tgt: [] for tgt in codes if tgt != src} for src in codes}
        for r in results:
            s, t = r["src_code"], r["tgt_code"]
            if table.get(s) is not None and t in table[s] and r.get("bleu_score") is not None:
                table[s][t].append(r["bleu_score"])
        table_avg = {
            src: {tgt: (sum(v) / len(v) if v else None) for tgt, v in tbl.items()}
            for src, tbl in table.items()
        }
        checkpoint_data = {
            "base_url": args.base_url,
            "split": args.split,
            "limit": args.limit,
            "replicates": n_replicates,
            "completed": len(results),
            "total": total,
            "results": results,
            "table": {
                src: {tgt: (float(v) if v is not None else None) for tgt, v in tbl.items()}
                for src, tbl in table_avg.items()
            },
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    # 最终表格（与检查点中 table_avg 一致，用于打印）
    table = {src: {tgt: [] for tgt in codes if tgt != src} for src in codes}
    for r in results:
        src, tgt = r["src_code"], r["tgt_code"]
        if table.get(src) is not None and tgt in table[src] and r.get("bleu_score") is not None:
            table[src][tgt].append(r["bleu_score"])
    table_avg = {
        src: {tgt: (sum(v) / len(v) if v else None) for tgt, v in tbl.items()}
        for src, tbl in table.items()
    }

    print("\n" + "=" * 80)
    print("BLEU 汇总（行=源语言, 列=目标语言）")
    print("=" * 80)
    col_width = 10
    header = "源 \\ 目标".ljust(14) + "".join(code_to_name[c][:6].ljust(col_width) for c in codes)
    print(header)
    print("-" * len(header))
    for src in codes:
        row = code_to_name[src].ljust(14)
        for tgt in codes:
            if src == tgt:
                row += "-".center(col_width)
            else:
                v = table_avg[src].get(tgt)
                row += (f"{v:.1f}" if v is not None else "ERR").ljust(col_width)
        print(row)

    summary_data = {
        "base_url": args.base_url,
        "split": args.split,
        "limit": args.limit,
        "replicates": n_replicates,
        "completed": len(results),
        "total": total,
        "results": results,
        "table": {
            src: {tgt: (float(v) if v is not None else None) for tgt, v in tbl.items()}
            for src, tbl in table_avg.items()
        },
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {checkpoint_path}")
    print(f"各语对详情目录: {out_dir}")


if __name__ == "__main__":
    main()
