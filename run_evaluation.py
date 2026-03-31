#!/usr/bin/env python3
"""
一键运行：可选启动本地 vLLM，再执行 FLORES-200 评估脚本。
将「本地 vLLM 服务」与「flores200_single / flores200_multilingual」结合起来。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import urllib.request
    URLLIB = True
except Exception:
    URLLIB = False


def wait_for_server(base_url: str, timeout: int = 300, interval: float = 2.0) -> bool:
    """轮询 base_url 的 /v1/models 直到返回 200 或超时。"""
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="启动 vLLM（可选）并运行 FLORES-200 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 先启动 vLLM，再跑 7 语种多向评估（42 个语对）
  python run_evaluation.py --mode multilingual --limit 50

  # 不启动 vLLM，只跑评估（需本机已有服务在 8005）
  python run_evaluation.py --no-serve --mode multilingual

  # 单语对评估（英→中）
  python run_evaluation.py --no-serve --mode single --config eng_Latn-zho_Hans --limit 100
""",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="不启动 vLLM，仅运行评估（本机已在对应端口跑 vLLM 时使用）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="vLLM 模型名或本地路径。不填时用环境变量 VLLM_MODEL，再否则用 Qwen/Qwen3.5-9B",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8005,
        help="vLLM 服务端口（默认 8005）",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM 张量并行 GPU 数（默认 1）",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        metavar="ID[,ID...]",
        help="指定使用的 GPU 编号，逗号分隔（如 0,1 或 4,5）。不填且会启动 vLLM 时会交互询问",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multilingual"],
        default="multilingual",
        help="single=单语对评估，multilingual=7 语种 42 向评估（默认）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="eng_Latn-zho_Hans",
        help="single 模式下的语对，如 eng_Latn-zho_Hans",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["dev", "devtest"],
        help="FLORES 划分",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个语对评估条数（不填则全量）",
    )
    parser.add_argument("--random-sample", action="store_true", help="与 --limit 同用：随机采样 N 条")
    parser.add_argument("--seed", type=int, default=None, help="随机采样种子")
    parser.add_argument("--replicates", type=int, default=1, help="multilingual 模式每语对重复次数（默认 1；2×5×2=20 组）")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_multilingual",
        help="multilingual 模式结果目录",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="multilingual 模式汇总 JSON 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="single 模式单语对结果 JSON 路径",
    )
    parser.add_argument(
        "--runs-csv",
        type=str,
        default=None,
        help="逐条样本运行结果 CSV 路径（不填则写到输出目录的 run_results.csv）",
    )
    parser.add_argument(
        "--lang-csv",
        type=str,
        default=None,
        help="按语言汇总分数 CSV 路径（不填则写到输出目录的 language_scores.csv）",
    )
    parser.add_argument("--max-workers", type=int, default=16, help="翻译请求并发数")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    base_url = f"http://localhost:{args.port}/v1"
    model = args.model or __import__("os").environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")

    vllm_proc = None
    if not args.no_serve:
        if not URLLIB:
            print("需要 urllib 以检测服务就绪，请使用 --no-serve 并手动启动 vLLM。", file=sys.stderr)
            sys.exit(1)
        gpu_str = args.gpus
        if gpu_str is None:
            try:
                gpu_str = input("请输入要使用的 GPU 编号，用逗号分隔（如 0,1 或 4,5），直接回车则使用全部: ").strip()
            except (EOFError, KeyboardInterrupt):
                gpu_str = ""
        env = os.environ.copy()
        if gpu_str:
            env["CUDA_VISIBLE_DEVICES"] = gpu_str
            print(f"使用 GPU: {gpu_str}")
        vllm_bin = script_dir / ".venv" / "bin" / "vllm"
        if not vllm_bin.exists():
            vllm_bin = "vllm"
        cmd = [
            str(vllm_bin),
            "serve",
            model,
            "--port",
            str(args.port),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--reasoning-parser",
            "qwen3",
        ]
        print("启动 vLLM:", " ".join(cmd))
        vllm_proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        print(f"等待 vLLM 就绪 ({base_url}) ...")
        if not wait_for_server(base_url):
            vllm_proc.terminate()
            print("vLLM 启动超时或未就绪。", file=sys.stderr)
            sys.exit(1)
        print("vLLM 已就绪。")

    # 运行评估
    if args.mode == "single":
        # 相对路径按 script_dir 解析；未指定 --output 则默认 eval_single/<config>.json
        output_json = (script_dir / args.output) if args.output and not Path(args.output).is_absolute() else (Path(args.output) if args.output else (script_dir / "eval_single" / f"{args.config}.json"))
        eval_cmd = [
            sys.executable,
            str(script_dir / "flores200_single.py"),
            "--config",
            args.config,
            "--split",
            args.split,
            "--base-url",
            base_url,
            "--max-tokens",
            "512",
        ]
        if args.limit is not None:
            eval_cmd.extend(["--limit", str(args.limit)])
        if args.random_sample:
            eval_cmd.append("--random-sample")
        if args.seed is not None:
            eval_cmd.extend(["--seed", str(args.seed)])
        eval_cmd.extend(["--output", str(output_json)])
        eval_cmd.extend(["--max-workers", str(args.max_workers)])
    else:
        # 相对路径按 script_dir 解析，与子进程 cwd 一致，避免从其他目录运行时找不到文件
        out_dir = (script_dir / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # 为了能导出 CSV：若未指定 --summary，则默认落到输出目录 summary.json
        summary_json = (script_dir / args.summary) if args.summary and not Path(args.summary).is_absolute() else (Path(args.summary) if args.summary else (out_dir / "summary.json"))
        eval_cmd = [
            sys.executable,
            str(script_dir / "flores200_multilingual.py"),
            "--base-url",
            base_url,
            "--split",
            args.split,
            "--output-dir",
            args.output_dir,
        ]
        if args.limit is not None:
            eval_cmd.extend(["--limit", str(args.limit)])
        if args.random_sample:
            eval_cmd.append("--random-sample")
        if args.seed is not None:
            eval_cmd.extend(["--seed", str(args.seed)])
        eval_cmd.extend(["--replicates", str(args.replicates)])
        eval_cmd.extend(["--summary", str(summary_json)])
        eval_cmd.extend(["--max-workers", str(args.max_workers)])

    print("运行评估:", " ".join(eval_cmd))
    rc = subprocess.run(eval_cmd, cwd=str(script_dir))
    if vllm_proc is not None:
        vllm_proc.terminate()
        vllm_proc.wait(timeout=10)

    if rc.returncode == 0:
        # -----------------------------
        # 导出 CSV：逐条样本 + 按语言汇总
        # -----------------------------
        def now_utc_iso() -> str:
            return datetime.now(tz=timezone.utc).isoformat()

        def _parse_pair_config(cfg: str) -> tuple[str, str]:
            parts = (cfg or "").split("-")
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
            return "", ""

        def _load_pair_json(p: Path) -> dict:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        # 收集所有语对 JSON
        pair_jsons: list[Path] = []
        summary_data: dict | None = None
        export_dir: Path
        if args.mode == "single":
            export_dir = output_json.parent
            pair_jsons = [output_json]
        else:
            export_dir = out_dir
            summary_data = _load_pair_json(summary_json)
            # 多语种详情在 output-dir 里，文件名形如 eng-Latn-zho-Hans.json
            pair_jsons = sorted(out_dir.glob("*.json"))
            # 避免把 summary.json 当作语对详情
            pair_jsons = [p for p in pair_jsons if p.name != summary_json.name]

        export_dir.mkdir(parents=True, exist_ok=True)
        runs_csv = Path(args.runs_csv) if args.runs_csv else (export_dir / "run_results.csv")
        lang_csv = Path(args.lang_csv) if args.lang_csv else (export_dir / "language_scores.csv")

        # 逐条样本结果
        run_rows: list[dict] = []
        pair_scores: list[dict] = []
        for pj in pair_jsons:
            data = _load_pair_json(pj)
            cfg = data.get("config", "")
            src_lang, tgt_lang = _parse_pair_config(cfg)
            bleu = data.get("bleu_score")
            comet = data.get("comet_score")
            base_url_j = data.get("base_url")
            model_j = data.get("model")
            split_j = data.get("split")
            pair_scores.append(
                {
                    "config": cfg,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "bleu_score": bleu,
                    "comet_score": comet,
                    "num_samples": data.get("num_samples"),
                }
            )

            samples = data.get("samples")
            if isinstance(samples, list) and samples:
                for s in samples:
                    run_rows.append(
                        {
                            "exported_at": now_utc_iso(),
                            "mode": args.mode,
                            "config": cfg,
                            "src_lang": s.get("src_lang", src_lang),
                            "tgt_lang": s.get("tgt_lang", tgt_lang),
                            "idx": s.get("idx"),
                            "started_at": s.get("started_at"),
                            "elapsed_s": s.get("elapsed_s"),
                            "source": s.get("source"),
                            "hypothesis": s.get("hypothesis"),
                            "reference": s.get("reference"),
                            "pair_bleu_score": bleu,
                            "pair_comet_score": comet,
                            "split": split_j,
                            "base_url": base_url_j,
                            "model": model_j,
                        }
                    )
            else:
                # 兼容旧结构：sources/hypotheses/references 三个数组（无逐条耗时/时间）
                sources = data.get("sources") or []
                hyps = data.get("hypotheses") or []
                refs = data.get("references") or []
                n = min(len(sources), len(hyps), len(refs))
                for i in range(n):
                    run_rows.append(
                        {
                            "exported_at": now_utc_iso(),
                            "mode": args.mode,
                            "config": cfg,
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                            "idx": i,
                            "started_at": None,
                            "elapsed_s": None,
                            "source": sources[i],
                            "hypothesis": hyps[i],
                            "reference": refs[i],
                            "pair_bleu_score": bleu,
                            "pair_comet_score": comet,
                            "split": split_j,
                            "base_url": base_url_j,
                            "model": model_j,
                        }
                    )

        # 写 run_results.csv
        run_fields = [
            "exported_at",
            "mode",
            "config",
            "src_lang",
            "tgt_lang",
            "idx",
            "started_at",
            "elapsed_s",
            "source",
            "hypothesis",
            "reference",
            "pair_bleu_score",
            "pair_comet_score",
            "split",
            "base_url",
            "model",
        ]
        with open(runs_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_fields)
            w.writeheader()
            for r in run_rows:
                w.writerow(r)

        # 按语言汇总（作为源语言/目标语言两种视角）
        as_src_bleu: dict[str, list[float]] = defaultdict(list)
        as_tgt_bleu: dict[str, list[float]] = defaultdict(list)
        as_src_comet: dict[str, list[float]] = defaultdict(list)
        as_tgt_comet: dict[str, list[float]] = defaultdict(list)
        for ps in pair_scores:
            s = ps.get("src_lang") or ""
            t = ps.get("tgt_lang") or ""
            b = ps.get("bleu_score")
            c = ps.get("comet_score")
            if s and isinstance(b, (int, float)):
                as_src_bleu[s].append(float(b))
            if t and isinstance(b, (int, float)):
                as_tgt_bleu[t].append(float(b))
            if s and isinstance(c, (int, float)):
                as_src_comet[s].append(float(c))
            if t and isinstance(c, (int, float)):
                as_tgt_comet[t].append(float(c))

        all_langs = sorted(set(list(as_src_bleu.keys()) + list(as_tgt_bleu.keys()) + list(as_src_comet.keys()) + list(as_tgt_comet.keys())))

        def _avg(xs: list[float]) -> float | None:
            return (sum(xs) / len(xs)) if xs else None

        lang_rows: list[dict] = []
        for lang in all_langs:
            lang_rows.append(
                {
                    "exported_at": now_utc_iso(),
                    "lang": lang,
                    "as_src_num_pairs": len(as_src_bleu.get(lang, [])) or len(as_src_comet.get(lang, [])),
                    "as_tgt_num_pairs": len(as_tgt_bleu.get(lang, [])) or len(as_tgt_comet.get(lang, [])),
                    "as_src_avg_bleu": _avg(as_src_bleu.get(lang, [])),
                    "as_tgt_avg_bleu": _avg(as_tgt_bleu.get(lang, [])),
                    "as_src_avg_comet": _avg(as_src_comet.get(lang, [])),
                    "as_tgt_avg_comet": _avg(as_tgt_comet.get(lang, [])),
                }
            )

        lang_fields = [
            "exported_at",
            "lang",
            "as_src_num_pairs",
            "as_tgt_num_pairs",
            "as_src_avg_bleu",
            "as_tgt_avg_bleu",
            "as_src_avg_comet",
            "as_tgt_avg_comet",
        ]
        with open(lang_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=lang_fields)
            w.writeheader()
            for r in lang_rows:
                w.writerow(r)

        print(f"已导出逐条样本 CSV: {runs_csv}")
        print(f"已导出分语言汇总 CSV: {lang_csv}")
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
