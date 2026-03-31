#!/usr/bin/env python3
"""
入口：读取 JSON 配置（可选启动 vLLM），在进程内调用 src 中的评估逻辑。
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

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowres_translation.eval_batch import run_batch_evaluation
from lowres_translation.eval_single import run_single_evaluation
from lowres_translation.evaluation_config import (
    evaluation_mode,
    limit_for_pair,
    load_evaluation_config,
    normalize_pair_sample_limits,
    resolve_evaluation_pairs,
    single_pair_config_string,
)

try:
    import urllib.request

    URLLIB = True
except Exception:
    URLLIB = False


def wait_for_server(base_url: str, timeout: int = 300, interval: float = 2.0) -> bool:
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


def _default_cfg() -> dict:
    return {
        "mode": "batch",
        "preset": None,
        "model": "Qwen/Qwen3.5-9B",
        "split": "dev",
        "limit": None,
        "pair_sample_limits": {},
        "random_sample": False,
        "seed": None,
        "replicates": 1,
        "metrics": ["bleu", "comet"],
        "bidirectional": True,
        "language_pair_groups": [
            ["eng_Latn", "zho_Hans"],
            ["spa_Latn", "ind_Latn", "vie_Latn", "tha_Thai", "tgl_Latn"],
        ],
        "language_pairs": [],
        "output_dir": "eval_multilingual",
        "summary": None,
        "output": None,
        "max_tokens": 512,
        "max_workers": 16,
        "verbose": False,
        "served_model_name": None,
    }


def _resolve_model_path(spec: str, project_root: Path) -> str:
    """
    若 spec 指向存在的目录或文件，返回绝对路径（供 vLLM 加载本地权重）；
    否则视为 Hugging Face Hub 模型 id 等，原样返回（已做 expanduser）。
    相对路径先按项目根解析，再按当前工作目录解析。
    """
    s = os.path.expanduser((spec or "").strip())
    if not s:
        return s
    p0 = Path(s)
    candidates: list[Path] = []
    if p0.is_absolute():
        candidates.append(p0)
    else:
        candidates.append((project_root / s).resolve())
        try:
            candidates.append(Path(s).resolve())
        except (OSError, RuntimeError):
            pass
    for cand in candidates:
        try:
            rp = cand.resolve()
        except (OSError, RuntimeError):
            continue
        if rp.exists() and (rp.is_dir() or rp.is_file()):
            return str(rp)
    return s


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = v
    return out


def _parse_metrics_arg(s: str | None) -> list[str] | None:
    if s is None:
        return None
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _export_csvs(
    *,
    mode: str,
    export_dir: Path,
    pair_jsons: list[Path],
    summary_json: Path | None,
    runs_csv: Path,
    lang_csv: Path,
) -> None:
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

    export_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict] = []
    pair_scores: list[dict] = []
    for pj in pair_jsons:
        if not pj.exists():
            continue
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
                        "mode": mode,
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
            sources = data.get("sources") or []
            hyps = data.get("hypotheses") or []
            refs = data.get("references") or []
            n = min(len(sources), len(hyps), len(refs))
            for i in range(n):
                run_rows.append(
                    {
                        "exported_at": now_utc_iso(),
                        "mode": mode,
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

    all_langs = sorted(
        set(list(as_src_bleu.keys()) + list(as_tgt_bleu.keys()) + list(as_src_comet.keys()) + list(as_tgt_comet.keys()))
    )

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FLORES-200 评估入口：JSON 配置 + 可选自动启动 vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="详见 README：配置文件字段与可取值。",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=str(ROOT / "evaluation_config.json"),
        help="评估 JSON 配置文件路径（默认项目根目录 evaluation_config.json）",
    )
    parser.add_argument("--no-serve", action="store_true", help="不启动 vLLM，仅评估")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖配置中的 model：可为 Hub id 或本地权重目录/文件的相对（相对项目根）或绝对路径",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="覆盖配置中的 served_model_name；与 vLLM --served-model-name 一致，供 API 请求里的 model 字段使用",
    )
    parser.add_argument("--port", type=int, default=8005, help="vLLM 端口（未指定 base_url 时用）")
    parser.add_argument("--base-url", type=str, default=None, help="覆盖配置中的 OpenAI 兼容 API 根 URL")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpus", type=str, default=None, metavar="ID[,ID...]")
    parser.add_argument("--mode", type=str, default=None, choices=["single", "batch"])
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="single 模式下的语对，如 eng_Latn-zho_Hans（覆盖配置文件中的 config）",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--replicates", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--summary", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="single 模式结果 JSON")
    parser.add_argument("--runs-csv", type=str, default=None)
    parser.add_argument("--lang-csv", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--metrics", type=str, default=None, help="逗号分隔，覆盖配置：bleu,comet")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cfg_path = Path(args.config_file)
    cfg = _default_cfg()
    if cfg_path.is_file():
        cfg = _deep_merge(cfg, load_evaluation_config(cfg_path))
    elif args.config_file != str(ROOT / "evaluation_config.json"):
        print(f"警告: 配置文件不存在: {cfg_path}，使用内置默认。", file=sys.stderr)

    if args.mode is not None:
        cfg["mode"] = args.mode
    if args.config is not None:
        cfg["config"] = args.config
    if args.limit is not None:
        cfg["limit"] = args.limit
    if args.random_sample:
        cfg["random_sample"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.replicates is not None:
        cfg["replicates"] = args.replicates
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.summary is not None:
        cfg["summary"] = args.summary
    if args.output is not None:
        cfg["output"] = args.output
    if args.max_workers is not None:
        cfg["max_workers"] = args.max_workers
    if args.max_tokens is not None:
        cfg["max_tokens"] = args.max_tokens
    if args.model is not None:
        cfg["model"] = args.model
    if args.served_model_name is not None:
        cfg["served_model_name"] = args.served_model_name
    if args.verbose:
        cfg["verbose"] = True

    metrics = _parse_metrics_arg(args.metrics)
    if metrics is None:
        m = cfg.get("metrics", ["bleu", "comet"])
        metrics = [str(x).lower().strip() for x in m] if isinstance(m, list) else ["bleu", "comet"]

    base_url = cfg.get("base_url")
    if args.base_url:
        base_url = args.base_url
    if not base_url:
        base_url = f"http://localhost:{args.port}/v1"

    env_fallback = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")

    def _user_model_spec() -> str | None:
        if args.model is not None:
            t = str(args.model).strip()
            return t if t else None
        m = cfg.get("model")
        if m is None:
            return None
        t = str(m).strip()
        return t if t else None

    user_spec = _user_model_spec()
    raw_spec = user_spec if user_spec is not None else env_fallback
    vllm_model_arg = _resolve_model_path(raw_spec, ROOT)

    served = (cfg.get("served_model_name") or "").strip()
    if args.served_model_name is not None:
        t = str(args.served_model_name).strip()
        served = t if t else ""

    if served:
        effective_model: str | None = served
    elif user_spec is not None:
        effective_model = vllm_model_arg
    elif not args.no_serve:
        effective_model = vllm_model_arg
    else:
        effective_model = None

    mode = evaluation_mode(cfg)
    split = cfg.get("split", "dev")
    if split not in ("dev", "devtest"):
        print("split 必须是 dev 或 devtest", file=sys.stderr)
        sys.exit(1)

    limit = cfg.get("limit")
    if limit is not None:
        limit = int(limit)
    seed = cfg.get("seed")
    if seed is not None:
        seed = int(seed)
    replicates = int(cfg.get("replicates", 1))
    max_tokens = int(cfg.get("max_tokens", 512))
    max_workers = int(cfg.get("max_workers", 16))
    verbose = bool(cfg.get("verbose", False))
    try:
        pair_sample_limits = normalize_pair_sample_limits(cfg.get("pair_sample_limits"))
    except ValueError as e:
        print(f"配置错误: {e}", file=sys.stderr)
        sys.exit(1)

    vllm_proc = None
    if not args.no_serve:
        if not URLLIB:
            print("需要 urllib 以检测服务就绪，请使用 --no-serve 并手动启动 vLLM。", file=sys.stderr)
            sys.exit(1)
        gpu_str = args.gpus
        if gpu_str is None:
            try:
                gpu_str = input("请输入要使用的 GPU 编号，用逗号分隔（如 0,1），直接回车则使用全部: ").strip()
            except (EOFError, KeyboardInterrupt):
                gpu_str = ""
        env = os.environ.copy()
        if gpu_str:
            env["CUDA_VISIBLE_DEVICES"] = gpu_str
            print(f"使用 GPU: {gpu_str}")
        vllm_bin = ROOT / ".venv" / "bin" / "vllm"
        if not vllm_bin.exists():
            vllm_bin = "vllm"
        cmd = [
            str(vllm_bin),
            "serve",
            vllm_model_arg,
            "--port",
            str(args.port),
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--reasoning-parser",
            "qwen3",
        ]
        if served:
            cmd.extend(["--served-model-name", served])
        print("启动 vLLM:", " ".join(cmd))
        vllm_proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
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

    rc = 0
    try:
        if mode == "single":
            pair_s = single_pair_config_string(cfg)
            out_rel = cfg.get("output")
            if args.output:
                out_rel = args.output
            output_json = (
                ROOT / out_rel
                if out_rel and not Path(out_rel).is_absolute()
                else Path(out_rel or (ROOT / "eval_single" / f"{pair_s}.json"))
            )
            output_json.parent.mkdir(parents=True, exist_ok=True)
            print(f"运行 single: {pair_s} → {output_json}")
            single_limit = limit_for_pair(pair_s, limit, pair_sample_limits)
            res = run_single_evaluation(
                config=pair_s,
                base_url=base_url,
                model=effective_model,
                split=split,
                limit=single_limit,
                random_sample=bool(cfg.get("random_sample", False)),
                seed=seed,
                max_tokens=max_tokens,
                output_path=output_json,
                verbose=True,
                max_workers=max_workers,
                metrics=metrics,
            )
            rc = 0 if not res.get("error") else 1
            export_dir = output_json.parent
            pair_jsons = [output_json]
            summary_json = None
        else:
            pairs = resolve_evaluation_pairs(cfg)
            out_dir = Path(cfg.get("output_dir", "eval_multilingual"))
            if not out_dir.is_absolute():
                out_dir = ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            summ = cfg.get("summary")
            summary_json = (
                ROOT / summ
                if summ and not Path(summ).is_absolute()
                else Path(summ or (out_dir / "summary.json"))
            )
            summary_json.parent.mkdir(parents=True, exist_ok=True)
            print(f"运行 batch: {len(pairs)} 个有向语对 → {out_dir}")
            run_batch_evaluation(
                pairs,
                base_url=base_url,
                model=effective_model,
                split=split,
                limit=limit,
                random_sample=bool(cfg.get("random_sample", False)),
                seed=seed,
                replicates=replicates,
                max_tokens=max_tokens,
                max_workers=max_workers,
                output_dir=out_dir,
                summary_path=summary_json,
                verbose=verbose,
                metrics=metrics,
                pair_sample_limits=pair_sample_limits,
            )
            export_dir = out_dir
            pair_jsons = sorted(out_dir.glob("*.json"))
            pair_jsons = [p for p in pair_jsons if p.name != summary_json.name]
            rc = 0

        if rc == 0:
            runs_csv = Path(args.runs_csv) if args.runs_csv else (export_dir / "run_results.csv")
            lang_csv = Path(args.lang_csv) if args.lang_csv else (export_dir / "language_scores.csv")
            _export_csvs(
                mode=mode,
                export_dir=export_dir,
                pair_jsons=pair_jsons,
                summary_json=summary_json,
                runs_csv=runs_csv,
                lang_csv=lang_csv,
            )
    finally:
        if vllm_proc is not None:
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
            except Exception:
                pass

    sys.exit(rc)


if __name__ == "__main__":
    main()
