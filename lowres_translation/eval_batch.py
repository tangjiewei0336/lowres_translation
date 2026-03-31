"""Run FLORES-200 evaluation over many directed language pairs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tqdm import tqdm

from lowres_translation.eval_single import run_single_evaluation
from lowres_translation.evaluation_config import limit_for_pair
from lowres_translation.flores_dataset import get_lang_name


def get_lang_name_cached(code: str, pairs: list[tuple[str, str, str, str]]) -> str:
    for s, t, sn, tn in pairs:
        if code == s:
            return sn
        if code == t:
            return tn
    return get_lang_name(code)


def run_batch_evaluation(
    pairs: list[tuple[str, str, str, str]],
    *,
    base_url: str,
    model: str | None,
    split: str,
    limit: int | None,
    random_sample: bool,
    seed: int | None,
    replicates: int,
    max_tokens: int,
    max_workers: int,
    output_dir: Path,
    summary_path: Path,
    verbose: bool,
    metrics: list[str] | None = None,
    pair_sample_limits: dict[str, int | None] | None = None,
) -> list[dict]:
    """
    pairs: (src_code, tgt_code, src_name, tgt_name)
    每完成一个语对写入 summary 检查点；各语对详情 JSON 在 output_dir。
    """
    codes = sorted({c for p in pairs for c in (p[0], p[1])})
    code_to_name = {c: get_lang_name_cached(c, pairs) for c in codes}

    n_rep = max(1, int(replicates))
    expanded: list[tuple[str, str, str, str, int]] = [
        (s, t, sn, tn, rid) for (s, t, sn, tn) in pairs for rid in range(n_rep)
    ]
    total = len(expanded)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[batch] 语对数={len(pairs)} × 重复={n_rep} → {total}  输出={output_dir}  汇总={summary_path}"
    )

    results: list[dict] = []
    for idx, (src_code, tgt_code, src_name, tgt_name, run_id) in enumerate(
        tqdm(expanded, desc="语对评估", unit="组")
    ):
        config = f"{src_code}-{tgt_code}"
        pair_limit = limit_for_pair(config, limit, pair_sample_limits)
        safe_name = config.replace("_", "-")
        output_path = output_dir / (
            f"{safe_name}_run{run_id}.json" if n_rep > 1 else f"{safe_name}.json"
        )
        t0 = time.perf_counter()
        res = run_single_evaluation(
            config=config,
            base_url=base_url,
            model=model,
            split=split,
            limit=pair_limit,
            random_sample=random_sample,
            seed=seed,
            max_tokens=max_tokens,
            output_path=output_path,
            verbose=verbose,
            show_progress=True,
            max_workers=max_workers,
            metrics=metrics,
        )
        elapsed = time.perf_counter() - t0
        one = {
            "config": config,
            "run_id": run_id,
            "limit": pair_limit,
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
        if not verbose:
            if score is not None:
                tqdm.write(
                    f"  [{idx + 1}/{total}] {src_name} → {tgt_name} (run{run_id}): BLEU {score:.2f}  ({elapsed:.1f}s)"
                )
            else:
                tqdm.write(
                    f"  [{idx + 1}/{total}] {src_name} → {tgt_name} (run{run_id}): 失败 {one.get('error', 'unknown')}"
                )

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
            "base_url": base_url,
            "split": split,
            "limit": limit,
            "pair_sample_limits": pair_sample_limits,
            "replicates": n_rep,
            "completed": len(results),
            "total": total,
            "results": results,
            "table": {
                src: {tgt: (float(v) if v is not None else None) for tgt, v in tbl.items()}
                for src, tbl in table_avg.items()
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

    _print_bleu_table(codes, code_to_name, results)
    table = {src: {tgt: [] for tgt in codes if tgt != src} for src in codes}
    for r in results:
        src, tgt = r["src_code"], r["tgt_code"]
        if table.get(src) is not None and tgt in table[src] and r.get("bleu_score") is not None:
            table[src][tgt].append(r["bleu_score"])
    table_avg = {
        src: {tgt: (sum(v) / len(v) if v else None) for tgt, v in tbl.items()}
        for src, tbl in table.items()
    }
    summary_data = {
        "base_url": base_url,
        "split": split,
        "limit": limit,
        "pair_sample_limits": pair_sample_limits,
        "replicates": n_rep,
        "completed": len(results),
        "total": total,
        "results": results,
        "table": {
            src: {tgt: (float(v) if v is not None else None) for tgt, v in tbl.items()}
            for src, tbl in table_avg.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存: {summary_path}")
    print(f"各语对详情目录: {output_dir}")
    return results


def _print_bleu_table(codes: list[str], code_to_name: dict[str, str], results: list[dict]) -> None:
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
