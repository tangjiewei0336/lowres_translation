"""Load evaluation JSON config: presets, language pairs, bidirectional expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lowres_translation.flores_dataset import get_lang_name

# 与旧版 flores200_multilingual 一致：中/英 × 西/印尼/越/泰/菲，双向
_PRESET_ZH_EN_CROSS_LOWRES_GROUP_A = [("英文", "eng_Latn"), ("中文", "zho_Hans")]
_PRESET_ZH_EN_CROSS_LOWRES_GROUP_B = [
    ("西班牙语", "spa_Latn"),
    ("印尼语", "ind_Latn"),
    ("越南语", "vie_Latn"),
    ("泰国语", "tha_Thai"),
    ("菲律宾语", "tgl_Latn"),
]


def _build_preset_zh_en_cross_lowres() -> list[tuple[str, str, str, str]]:
    names_a = {code: name for name, code in _PRESET_ZH_EN_CROSS_LOWRES_GROUP_A}
    names_b = {code: name for name, code in _PRESET_ZH_EN_CROSS_LOWRES_GROUP_B}
    names = {**names_a, **names_b}
    pairs = []
    for _n1, c1 in _PRESET_ZH_EN_CROSS_LOWRES_GROUP_A:
        for _n2, c2 in _PRESET_ZH_EN_CROSS_LOWRES_GROUP_B:
            pairs.append((c1, c2, names[c1], names[c2]))
            pairs.append((c2, c1, names[c2], names[c1]))
    return pairs


PRESETS: dict[str, list[tuple[str, str, str, str]]] = {
    "zh_en_cross_lowres": _build_preset_zh_en_cross_lowres(),
}


def _as_lang_codes(raw: Any, *, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} 必须是字符串数组")
    out = [str(x).strip() for x in raw if str(x).strip()]
    if not out:
        raise ValueError(f"{field} 至少包含一个语言代码")
    return out


def _expand_language_pair_groups(
    groups_raw: Any,
    *,
    bidirectional: bool,
) -> list[tuple[str, str, str, str]]:
    """
    两组语言代码的笛卡尔积：第一组中每个 src × 第二组中每个 tgt。
    bidirectional 为 True 时，对每个 (a,b) 再增加 (b,a)（去重）。
    """
    if not isinstance(groups_raw, list):
        raise ValueError("language_pair_groups 必须是数组，且恰好包含两个子数组")
    if len(groups_raw) != 2:
        raise ValueError(
            "language_pair_groups 必须恰好包含两个子数组，例如 "
            '[["eng_Latn","zho_Hans"], ["spa_Latn","ind_Latn"]]'
        )
    g0 = _as_lang_codes(groups_raw[0], field="language_pair_groups[0]")
    g1 = _as_lang_codes(groups_raw[1], field="language_pair_groups[1]")

    expanded: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a in g0:
        for b in g1:
            if a == b:
                continue
            directed: tuple[tuple[str, str], ...]
            if bidirectional:
                directed = ((a, b), (b, a))
            else:
                directed = ((a, b),)
            for x, y in directed:
                if x == y:
                    continue
                key = (x, y)
                if key in seen:
                    continue
                seen.add(key)
                expanded.append((x, y, get_lang_name(x), get_lang_name(y)))
    return expanded


def _as_str_pairs(raw: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not raw:
        return out
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((str(item[0]).strip(), str(item[1]).strip()))
        elif isinstance(item, dict) and "src" in item and "tgt" in item:
            out.append((str(item["src"]).strip(), str(item["tgt"]).strip()))
        else:
            raise ValueError(f"无效的 language_pairs 项: {item!r}")
    return out


def resolve_evaluation_pairs(cfg: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    """
    返回 [(src_code, tgt_code, src_name, tgt_name), ...]
    """
    preset = (cfg.get("preset") or "").strip()
    bidirectional = bool(cfg.get("bidirectional", True))
    raw_pairs = cfg.get("language_pairs")
    groups_raw = cfg.get("language_pair_groups")

    if preset:
        if preset not in PRESETS:
            raise ValueError(f"未知 preset={preset!r}，可选: {sorted(PRESETS.keys())}")
        base = PRESETS[preset]
        return list(base)

    if groups_raw is not None and groups_raw != []:
        return _expand_language_pair_groups(groups_raw, bidirectional=bidirectional)

    pairs = _as_str_pairs(raw_pairs)
    if not pairs:
        raise ValueError(
            "配置中需提供 preset、非空的 language_pair_groups，或非空的 language_pairs"
        )

    expanded: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for s, t in pairs:
        if not s or not t:
            raise ValueError(f"语言代码不能为空: {(s, t)}")
        for a, b in ((s, t), (t, s)) if bidirectional else ((s, t),):
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            expanded.append((a, b, get_lang_name(a), get_lang_name(b)))
    return expanded


def normalize_pair_sample_limits(raw: Any) -> dict[str, int | None] | None:
    """
    pair_sample_limits: 语对字符串 -> 该语对使用的样本条数。
    键格式与评估 config 一致，如 eng_Latn-zho_Hans（下划线在语言代码内，中间一个连字符）。
    值为 null 表示该语对使用**全量**句子（与顶层 limit:null 同义）；未出现的语对使用顶层 limit。
    """
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "pair_sample_limits 必须是 JSON 对象，例如 {\"eng_Latn-spa_Latn\": 100, \"spa_Latn-eng_Latn\": null}"
        )
    out: dict[str, int | None] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        if v is None:
            out[key] = None
        else:
            out[key] = int(v)
    return out or None


def limit_for_pair(
    config: str,
    default_limit: int | None,
    pair_limits: dict[str, int | None] | None,
) -> int | None:
    """返回该有向语对应使用的 limit（条数）；None 表示全量。"""
    if not pair_limits:
        return default_limit
    if config not in pair_limits:
        return default_limit
    return pair_limits[config]


def load_evaluation_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    return data


def evaluation_mode(cfg: dict[str, Any]) -> str:
    m = (cfg.get("mode") or "batch").strip().lower()
    if m not in ("single", "batch"):
        raise ValueError("mode 必须是 single 或 batch")
    return m


def single_pair_config_string(cfg: dict[str, Any]) -> str:
    """
    single 模式下的语对字符串，如 eng_Latn-zho_Hans。
    优先使用顶层 config；否则取 language_pairs 的第一项。
    """
    raw = (cfg.get("config") or "").strip()
    if raw:
        return raw
    pairs = _as_str_pairs(cfg.get("language_pairs"))
    if len(pairs) >= 1:
        s, t = pairs[0]
        return f"{s}-{t}"
    raise ValueError('single 模式需要 "config": "src-tgt" 或非空的 language_pairs')
