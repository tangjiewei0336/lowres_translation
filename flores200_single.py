#!/usr/bin/env python3
"""
FLORES-200 单语对评估模块。
可被直接运行（CLI）或由 flores200_multilingual 导入调用。
Dataset: https://huggingface.co/datasets/Muennighoff/flores200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import tarfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# 关闭 HTTP 请求的 INFO 日志，避免刷屏
for _name in ("urllib3", "httpx", "httpcore", "openai"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# FLORES-200 全部 200 种语言代码与显示名（用于 prompt）
# 来源: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
LANG_NAMES = {
    "ace_Arab": "Acehnese (Arabic script)",
    "ace_Latn": "Acehnese (Latin script)",
    "acm_Arab": "Mesopotamian Arabic",
    "acq_Arab": "Ta'izzi-Adeni Arabic",
    "aeb_Arab": "Tunisian Arabic",
    "afr_Latn": "Afrikaans",
    "ajp_Arab": "South Levantine Arabic",
    "aka_Latn": "Akan",
    "amh_Ethi": "Amharic",
    "apc_Arab": "North Levantine Arabic",
    "arb_Arab": "Modern Standard Arabic",
    "arb_Latn": "Modern Standard Arabic (Romanized)",
    "ars_Arab": "Najdi Arabic",
    "ary_Arab": "Moroccan Arabic",
    "arz_Arab": "Egyptian Arabic",
    "asm_Beng": "Assamese",
    "ast_Latn": "Asturian",
    "awa_Deva": "Awadhi",
    "ayr_Latn": "Central Aymara",
    "azb_Arab": "South Azerbaijani",
    "azj_Latn": "North Azerbaijani",
    "bak_Cyrl": "Bashkir",
    "bam_Latn": "Bambara",
    "ban_Latn": "Balinese",
    "bel_Cyrl": "Belarusian",
    "bem_Latn": "Bemba",
    "ben_Beng": "Bengali",
    "bho_Deva": "Bhojpuri",
    "bjn_Arab": "Banjar (Arabic script)",
    "bjn_Latn": "Banjar (Latin script)",
    "bod_Tibt": "Standard Tibetan",
    "bos_Latn": "Bosnian",
    "bug_Latn": "Buginese",
    "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan",
    "ceb_Latn": "Cebuano",
    "ces_Latn": "Czech",
    "cjk_Latn": "Chokwe",
    "ckb_Arab": "Central Kurdish",
    "crh_Latn": "Crimean Tatar",
    "cym_Latn": "Welsh",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "dik_Latn": "Southwestern Dinka",
    "dyu_Latn": "Dyula",
    "dzo_Tibt": "Dzongkha",
    "ell_Grek": "Greek",
    "eng_Latn": "English",
    "epo_Latn": "Esperanto",
    "est_Latn": "Estonian",
    "eus_Latn": "Basque",
    "ewe_Latn": "Ewe",
    "fao_Latn": "Faroese",
    "fij_Latn": "Fijian",
    "fin_Latn": "Finnish",
    "fon_Latn": "Fon",
    "fra_Latn": "French",
    "fur_Latn": "Friulian",
    "fuv_Latn": "Nigerian Fulfulde",
    "gla_Latn": "Scottish Gaelic",
    "gle_Latn": "Irish",
    "glg_Latn": "Galician",
    "grn_Latn": "Guarani",
    "guj_Gujr": "Gujarati",
    "hat_Latn": "Haitian Creole",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "hne_Deva": "Chhattisgarhi",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo",
    "ilo_Latn": "Ilocano",
    "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic",
    "ita_Latn": "Italian",
    "jav_Latn": "Javanese",
    "jpn_Jpan": "Japanese",
    "kab_Latn": "Kabyle",
    "kac_Latn": "Jingpho",
    "kam_Latn": "Kamba",
    "kan_Knda": "Kannada",
    "kas_Arab": "Kashmiri (Arabic script)",
    "kas_Deva": "Kashmiri (Devanagari script)",
    "kat_Geor": "Georgian",
    "knc_Arab": "Central Kanuri (Arabic script)",
    "knc_Latn": "Central Kanuri (Latin script)",
    "kaz_Cyrl": "Kazakh",
    "kbp_Latn": "Kabiyè",
    "kea_Latn": "Kabuverdianu",
    "khk_Cyrl": "Halh Mongolian",
    "khm_Khmr": "Khmer",
    "kik_Latn": "Kikuyu",
    "kin_Latn": "Kinyarwanda",
    "kir_Cyrl": "Kyrgyz",
    "kmb_Latn": "Kimbundu",
    "kmr_Latn": "Northern Kurdish",
    "kon_Latn": "Kikongo",
    "kor_Hang": "Korean",
    "lao_Laoo": "Lao",
    "lij_Latn": "Ligurian",
    "lim_Latn": "Limburgish",
    "lin_Latn": "Lingala",
    "lit_Latn": "Lithuanian",
    "lmo_Latn": "Lombard",
    "ltg_Latn": "Latgalian",
    "ltz_Latn": "Luxembourgish",
    "lua_Latn": "Luba-Kasai",
    "lug_Latn": "Ganda",
    "luo_Latn": "Luo",
    "lus_Latn": "Mizo",
    "lvs_Latn": "Standard Latvian",
    "mag_Deva": "Magahi",
    "mai_Deva": "Maithili",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "min_Arab": "Minangkabau (Arabic script)",
    "min_Latn": "Minangkabau (Latin script)",
    "mkd_Cyrl": "Macedonian",
    "plt_Latn": "Plateau Malagasy",
    "mlt_Latn": "Maltese",
    "mni_Beng": "Meitei (Bengali script)",
    "mos_Latn": "Mossi",
    "mri_Latn": "Maori",
    "mya_Mymr": "Burmese",
    "nld_Latn": "Dutch",
    "nno_Latn": "Norwegian Nynorsk",
    "nob_Latn": "Norwegian Bokmål",
    "npi_Deva": "Nepali",
    "nso_Latn": "Northern Sotho",
    "nus_Latn": "Nuer",
    "nya_Latn": "Nyanja",
    "oci_Latn": "Occitan",
    "gaz_Latn": "West Central Oromo",
    "ory_Orya": "Odia",
    "pag_Latn": "Pangasinan",
    "pan_Guru": "Eastern Panjabi",
    "pap_Latn": "Papiamento",
    "pes_Arab": "Western Persian",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "prs_Arab": "Dari",
    "pbt_Arab": "Southern Pashto",
    "quy_Latn": "Ayacucho Quechua",
    "ron_Latn": "Romanian",
    "run_Latn": "Rundi",
    "rus_Cyrl": "Russian",
    "sag_Latn": "Sango",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "scn_Latn": "Sicilian",
    "shn_Mymr": "Shan",
    "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "smo_Latn": "Samoan",
    "sna_Latn": "Shona",
    "snd_Arab": "Sindhi",
    "som_Latn": "Somali",
    "sot_Latn": "Southern Sotho",
    "spa_Latn": "Spanish",
    "als_Latn": "Tosk Albanian",
    "srd_Latn": "Sardinian",
    "srp_Cyrl": "Serbian",
    "ssw_Latn": "Swati",
    "sun_Latn": "Sundanese",
    "swe_Latn": "Swedish",
    "swh_Latn": "Swahili",
    "szl_Latn": "Silesian",
    "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar",
    "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik",
    "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai",
    "tir_Ethi": "Tigrinya",
    "taq_Latn": "Tamasheq (Latin script)",
    "taq_Tfng": "Tamasheq (Tifinagh script)",
    "tpi_Latn": "Tok Pisin",
    "tsn_Latn": "Tswana",
    "tso_Latn": "Tsonga",
    "tuk_Latn": "Turkmen",
    "tum_Latn": "Tumbuka",
    "tur_Latn": "Turkish",
    "twi_Latn": "Twi",
    "tzm_Tfng": "Central Atlas Tamazight",
    "uig_Arab": "Uyghur",
    "ukr_Cyrl": "Ukrainian",
    "umb_Latn": "Umbundu",
    "urd_Arab": "Urdu",
    "uzn_Latn": "Northern Uzbek",
    "vec_Latn": "Venetian",
    "vie_Latn": "Vietnamese",
    "war_Latn": "Waray",
    "wol_Latn": "Wolof",
    "xho_Latn": "Xhosa",
    "ydd_Hebr": "Eastern Yiddish",
    "yor_Latn": "Yoruba",
    "yue_Hant": "Yue Chinese",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "zsm_Latn": "Standard Malay",
    "zul_Latn": "Zulu",
}

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
    返回 (sources, references)。
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
            temperature=0.0,extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
    }
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return ""  # 错误时在调用处统一打印


@dataclass(frozen=True)
class SampleRun:
    idx: int
    started_at: str | None
    elapsed_s: float | None
    hypothesis: str


def translate_batch(
    client,  # OpenAI
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
    results = [""] * len(sources)  # 预分配，保证顺序
    sample_runs: list[SampleRun] | None = [SampleRun(i, None, None, "") for i in range(len(sources))] if return_sample_runs else None
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


# COMET 模型只加载一次，多语对复用，避免反复下载/加载
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
    """
    计算 COMET 系统分（越大越好）。
    使用模块级缓存，权重只加载一次，多语对复用。
    """
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
) -> dict:
    """
    运行单语对 FLORES-200 评估，可被 flores200_multilingual 直接调用。
    返回 dict：config, bleu_score, num_samples, hypotheses, references, sources, model, error(若失败)。
    """
    import time as _time
    from openai import OpenAI
    from sacrebleu import BLEU

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
            client, model, sources, src_lang, tgt_lang,
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

        bleu = BLEU()
        score = bleu.corpus_score(hypotheses, [references])
        result["bleu_score"] = score.score
        if verbose:
            print(f"  BLEU: {score.score:.2f}  ({score})")

        comet_score, comet_error = _compute_comet_system_score(sources, hypotheses, references)
        result["comet_score"] = comet_score
        if comet_error and verbose:
            print(f"  COMET: 跳过/失败 ({comet_error})")
        elif comet_score is not None and verbose:
            print(f"  COMET(system): {comet_score:.4f}")

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "config": config,
                "split": split,
                "base_url": base_url,
                "model": model,
                "num_samples": len(sources),
                "bleu_score": score.score,
                "comet_score": comet_score,
                "bleu_str": str(score),
                "hypotheses": hypotheses,
                "references": references,
                "sources": sources,
                "samples": result["samples"],
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
    parser.add_argument("--list-languages", action="store_true", help="列出全部语言代码后退出")
    args = parser.parse_args()

    if args.list_languages:
        print("FLORES-200 支持以下 200 种语言（--config 格式：源语-目标语，如 eng_Latn-zho_Hans）：\n")
        for code, name in sorted(LANG_NAMES.items()):
            print(f"  {code:14}  {name}")
        print("\n示例: --config eng_Latn-fra_Latn  英→法  --config zho_Hans-eng_Latn  简中→英")
        return

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
    )
    if res.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
