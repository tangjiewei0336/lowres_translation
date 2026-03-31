"""FLORES-200 single pair evaluation (vLLM OpenAI-compatible API)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from lowres_translation.flores_dataset import (
    get_lang_name,
    load_sources_references,
    parse_config,
)
from lowres_translation.lang_names import LANG_NAMES

for _name in ("urllib3", "httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _translate_one(
    client,
    model: str,
    text: str,
    system: str,
    max_tokens: int,
) -> str:
    """单条翻译，供并发调用。"""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text.strip()},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


@dataclass(frozen=True)
class SampleRun:
    idx: int
    started_at: str | None
    elapsed_s: float | None
    hypothesis: str


def translate_batch(
    client,
    model: str,
    sources: list[str],
    src_lang: str,
    tgt_lang: str,
    max_tokens: int = 512,
    max_workers: int = 16,
    show_progress: bool = True,
    return_sample_runs: bool = False,
) -> list[str] | tuple[list[str], list[SampleRun]]:
    """调用 vLLM API 批量翻译，并发发送请求（线程池）。"""
    src_name = get_lang_name(src_lang)
    tgt_name = get_lang_name(tgt_lang)
    system = (
        f"You are a professional translator. Translate from {src_name} to {tgt_name}. "
        "Output only the translation, no explanation or preamble."
    )
    workers = min(max_workers, len(sources)) or 1
    results = [""] * len(sources)
    sample_runs: list[SampleRun] | None = (
        [SampleRun(i, None, None, "") for i in range(len(sources))] if return_sample_runs else None
    )
    errors = []

    def do_one(i: int, text: str) -> tuple[int, str]:
        started_ts = time.time()
        started_at = datetime.fromtimestamp(started_ts, tz=timezone.utc).isoformat()
        t0 = time.perf_counter()
        out = _translate_one(client, model, text, system, max_tokens)
        elapsed_s = time.perf_counter() - t0
        if sample_runs is not None:
            sample_runs[i] = SampleRun(idx=i, started_at=started_at, elapsed_s=elapsed_s, hypothesis=out)
        return (i, out)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(do_one, i, text): i for i, text in enumerate(sources)}
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(sources), desc="  翻译", unit="句", leave=True) if show_progress else None
        except ImportError:
            pbar = None
        for fut in as_completed(futures):
            try:
                idx, out = fut.result()
                results[idx] = out
            except Exception as e:
                errors.append(str(e))
                results[futures[fut]] = ""
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    for e in errors:
        print(f"  [API 错误] {e}")
    if return_sample_runs:
        return results, (sample_runs or [])
    return results


_COMET_MODEL_CACHE: tuple | None = None
_COMET_MODEL_ERROR: str | None = None


def _get_comet_model():
    """懒加载并缓存 COMET 模型，仅首次调用时下载/加载。"""
    global _COMET_MODEL_CACHE, _COMET_MODEL_ERROR
    if _COMET_MODEL_ERROR is not None:
        return None, _COMET_MODEL_ERROR
    if _COMET_MODEL_CACHE is not None:
        return _COMET_MODEL_CACHE, None
    try:
        from comet import download_model, load_from_checkpoint  # type: ignore
    except Exception as e:
        _COMET_MODEL_ERROR = f"COMET 未安装: {e}. 请安装: pip install unbabel-comet"
        return None, _COMET_MODEL_ERROR
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        _COMET_MODEL_CACHE = model
        return model, None
    except Exception as e:
        _COMET_MODEL_ERROR = f"COMET 加载失败: {e}"
        return None, _COMET_MODEL_ERROR


def _compute_comet_system_score(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    *,
    batch_size: int = 8,
) -> tuple[float | None, str | None]:
    """计算 COMET 系统分（越大越好）。"""
    model, err = _get_comet_model()
    if model is None:
        return None, err
    try:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hypotheses, references)]
        pred = model.predict(data, batch_size=batch_size, gpus=1, progress_bar=False)
        return float(pred.system_score), None
    except Exception as e:
        return None, f"COMET 计算失败: {e}"


def run_single_evaluation(
    config: str,
    base_url: str,
    model: str | None = None,
    split: str = "dev",
    limit: int | None = None,
    random_sample: bool = False,
    seed: int | None = None,
    max_tokens: int = 512,
    output_path: Path | str | None = None,
    verbose: bool = True,
    show_progress: bool = True,
    max_workers: int = 16,
    metrics: list[str] | None = None,
) -> dict:
    """
    运行单语对 FLORES-200 评估。
    metrics: 例如 ["bleu", "comet"]；未列出的指标将跳过（省 COMET 时间）。
    """
    import time as _time

    from openai import OpenAI
    from sacrebleu import BLEU

    mset = {m.lower().strip() for m in (metrics or ["bleu", "comet"])}
    src_lang, tgt_lang, src_col, tgt_col = parse_config(config)
    result: dict = {
        "config": config,
        "bleu_score": None,
        "comet_score": None,
        "num_samples": None,
        "hypotheses": [],
        "references": [],
        "sources": [],
        "samples": [],
        "model": None,
        "error": None,
        "metrics": sorted(mset),
    }

    try:
        if verbose:
            print(f"  [1/4] 语对: {src_lang} → {tgt_lang}  (列: {src_col} / {tgt_col}, split={split})")
        sources, references = load_sources_references(
            config, split, limit=limit, random_sample=random_sample, seed=seed, verbose=verbose
        )
        result["sources"] = sources
        result["references"] = references
        result["num_samples"] = len(sources)
        if verbose:
            print(f"  [2/4] 已加载 {len(sources)} 条样本")

        api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = OpenAI(api_key=api_key, base_url=base_url)
        if model is None:
            try:
                models = client.models.list()
                model = models.data[0].id
            except Exception as e:
                err = str(e).lower()
                if "401" in err or "authentication" in err or "invalid" in err and verbose:
                    print("错误: 无法连接模型列表（401）。请确认 vLLM 已启动且 --base-url 正确，或使用 --model 指定模型名。")
                result["error"] = str(e)
                return result
        result["model"] = model
        if verbose:
            print(f"  [3/4] API: {base_url}  模型: {model}")

        if verbose:
            print(f"  [4/4] 翻译中 (max_tokens={max_tokens}, 并发={max_workers}) ...")
        t0 = _time.perf_counter()
        hypotheses, sample_runs = translate_batch(
            client,
            model,
            sources,
            src_lang,
            tgt_lang,
            max_tokens=max_tokens,
            max_workers=max_workers,
            show_progress=show_progress,
            return_sample_runs=True,
        )
        result["hypotheses"] = hypotheses
        result["samples"] = [
            {
                "idx": sr.idx,
                "started_at": sr.started_at,
                "elapsed_s": sr.elapsed_s,
                "source": sources[sr.idx],
                "reference": references[sr.idx],
                "hypothesis": sr.hypothesis,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
            for sr in sample_runs
        ]
        elapsed = _time.perf_counter() - t0
        if verbose:
            print(f"  翻译耗时: {elapsed:.1f}s")

        score = None
        if "bleu" in mset:
            bleu = BLEU()
            score = bleu.corpus_score(hypotheses, [references])
            result["bleu_score"] = score.score
            if verbose:
                print(f"  BLEU: {score.score:.2f}  ({score})")
        elif verbose:
            print("  BLEU: 已跳过（metrics 未包含 bleu）")

        comet_score = None
        comet_error = None
        if "comet" in mset:
            comet_score, comet_error = _compute_comet_system_score(sources, hypotheses, references)
            result["comet_score"] = comet_score
            if comet_error and verbose:
                print(f"  COMET: 跳过/失败 ({comet_error})")
            elif comet_score is not None and verbose:
                print(f"  COMET(system): {comet_score:.4f}")
        else:
            result["comet_score"] = None
            if verbose:
                print("  COMET: 已跳过（metrics 未包含 comet）")

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "config": config,
                "split": split,
                "base_url": base_url,
                "model": model,
                "num_samples": len(sources),
                "bleu_score": result["bleu_score"],
                "comet_score": comet_score,
                "bleu_str": str(score) if score is not None else None,
                "hypotheses": hypotheses,
                "references": references,
                "sources": sources,
                "samples": result["samples"],
                "metrics": sorted(mset),
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"结果已写入: {out}")
    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"评估失败: {e}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FLORES-200 单语对评估（vLLM 翻译模型）")
    parser.add_argument("--config", type=str, default="eng_Latn-zho_Hans", help="语对，如 eng_Latn-zho_Hans")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "devtest"])
    parser.add_argument("--base-url", type=str, default="http://localhost:8005/v1", help="vLLM API 地址")
    parser.add_argument("--model", type=str, default=None, help="模型名，不填则自动检测")
    parser.add_argument("--limit", type=int, default=None, help="评估条数（不填则全量）")
    parser.add_argument("--random-sample", action="store_true", help="与 --limit 同用：从全量中随机采样 N 条，否则取前 N 条")
    parser.add_argument("--seed", type=int, default=None, help="随机采样时的种子，便于复现")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径")
    parser.add_argument("--max-workers", type=int, default=16, help="翻译请求并发数（默认 16）")
    parser.add_argument(
        "--metrics",
        type=str,
        default="bleu,comet",
        help="逗号分隔：bleu, comet（默认两者都算）",
    )
    parser.add_argument("--list-languages", action="store_true", help="列出全部语言代码后退出")
    args = parser.parse_args()

    if args.list_languages:
        print("FLORES-200 支持以下 200 种语言（--config 格式：源语-目标语，如 eng_Latn-zho_Hans）：\n")
        for code, name in sorted(LANG_NAMES.items()):
            print(f"  {code:14}  {name}")
        print("\n示例: --config eng_Latn-fra_Latn  英→法  --config zho_Hans-eng_Latn  简中→英")
        return

    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    print(f"[调试] base_url={args.base_url}  config={args.config}  split={args.split}  limit={args.limit}")
    res = run_single_evaluation(
        config=args.config,
        base_url=args.base_url,
        model=args.model,
        split=args.split,
        limit=args.limit,
        random_sample=args.random_sample,
        seed=args.seed,
        max_tokens=args.max_tokens,
        output_path=args.output,
        verbose=True,
        max_workers=args.max_workers,
        metrics=metrics,
    )
    if res.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
