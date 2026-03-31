#!/usr/bin/env python3
"""
从 eval_multilingual/summary.json 读取 BLEU/COMET 汇总并绘制热力图。
用法: python plot_bleu_heatmap.py [--summary path] [--out dir] [--metric bleu|comet|both]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# 与 flores200_multilingual 一致的语种顺序与显示名
CODE_TO_NAME = {
    "eng_Latn": "英文",
    "zho_Hans": "中文",
    "spa_Latn": "西班牙语",
    "ind_Latn": "印尼语",
    "vie_Latn": "越南语",
    "tha_Thai": "泰国语",
    "tgl_Latn": "菲律宾语",
}
LANG_ORDER = list(CODE_TO_NAME.keys())


def _code_to_idx(code: str) -> int | None:
    try:
        return LANG_ORDER.index(code)
    except ValueError:
        return None


def _setup_cjk_font(plt, font_manager) -> None:
    found = None
    cjk_candidates = [
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Heiti SC",
    ]
    for name in cjk_candidates:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            found = name
            break
    if found:
        plt.rcParams["font.sans-serif"] = [found] + list(plt.rcParams["font.sans-serif"])
    else:
        for f in font_manager.fontManager.ttflist:
            if "CJK" in f.name or "Chinese" in f.name or "WenQuanYi" in f.name:
                plt.rcParams["font.sans-serif"] = [f.name] + list(plt.rcParams["font.sans-serif"])
                found = f.name
                break
        if not found:
            print("未检测到中文字体，中文可能显示为方框。可安装: sudo apt install fonts-wqy-microhei 或 fonts-noto-cjk")
    plt.rcParams["axes.unicode_minus"] = False


def _draw_heatmap(ax, matrix: np.ndarray, labels: list[str], title: str, vmin: float, vmax: float, fmt: str = ".1f"):
    n = len(labels)
    plot_data = np.ma.masked_invalid(matrix)
    im = ax.imshow(plot_data, cmap="YlOrRd", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    import matplotlib.pyplot as plt
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            if i == j:
                text = "—"
            elif np.isnan(matrix[i, j]):
                text = "ERR"
            else:
                text = f"{matrix[i, j]:{fmt}}"
            color = "white" if not np.isnan(matrix[i, j]) and matrix[i, j] > (vmin + vmax) / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
    ax.set_title(title)
    return im


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 BLEU/COMET 汇总热力图")
    parser.add_argument("--summary", type=str, default=None, help="summary.json 路径，默认 eval_multilingual/summary.json")
    parser.add_argument("--out", type=str, default=None, help="输出目录，默认 summary 同目录")
    parser.add_argument("--metric", type=str, choices=["bleu", "comet", "both"], default="both", help="绘制 BLEU / COMET / 两者")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    summary_path = Path(args.summary) if args.summary else (script_dir / "eval_multilingual" / "summary.json")
    if not summary_path.is_absolute():
        summary_path = script_dir / summary_path
    out_dir = Path(args.out) if args.out else summary_path.parent

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    table = data.get("table") or {}
    results = data.get("results") or []

    n = len(LANG_ORDER)
    labels = [CODE_TO_NAME[c] for c in LANG_ORDER]

    # BLEU 矩阵（来自 table）
    bleu_matrix = np.full((n, n), np.nan)
    for i, src in enumerate(LANG_ORDER):
        row = table.get(src) or {}
        for j, tgt in enumerate(LANG_ORDER):
            if src == tgt:
                continue
            v = row.get(tgt)
            if v is not None:
                bleu_matrix[i, j] = float(v)

    # COMET 矩阵（来自 results）
    comet_matrix = np.full((n, n), np.nan)
    for r in results:
        src, tgt = r.get("src_code"), r.get("tgt_code")
        val = r.get("comet_score")
        if src is None or tgt is None or val is None:
            continue
        i, j = _code_to_idx(src), _code_to_idx(tgt)
        if i is not None and j is not None:
            comet_matrix[i, j] = float(val)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ImportError:
        raise SystemExit("请安装 matplotlib: pip install matplotlib")

    _setup_cjk_font(plt, font_manager)

    def save_bleu():
        fig, ax = plt.subplots(figsize=(8, 7))
        vmin = 0
        vmax = 50
        if not np.all(np.isnan(bleu_matrix)):
            vmax = max(50, np.nanmax(bleu_matrix) * 1.05)
        im = _draw_heatmap(ax, bleu_matrix, labels, "BLEU 汇总（行=源语言, 列=目标语言）", vmin, vmax)
        fig.colorbar(im, ax=ax, label="BLEU")
        fig.tight_layout()
        path = out_dir / "bleu_heatmap.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"已保存: {path}")

    def save_comet():
        fig, ax = plt.subplots(figsize=(8, 7))
        valid = comet_matrix[~np.isnan(comet_matrix)]
        if len(valid) == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.nanmin(comet_matrix))
            vmax = float(np.nanmax(comet_matrix))
            if vmax - vmin < 1e-6:
                vmin, vmax = vmin - 0.1, vmax + 0.1
        im = _draw_heatmap(ax, comet_matrix, labels, "COMET 汇总（行=源语言, 列=目标语言）", vmin, vmax, fmt=".3f")
        fig.colorbar(im, ax=ax, label="COMET")
        fig.tight_layout()
        path = out_dir / "comet_heatmap.png"
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        plt.close()
        print(f"已保存: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.metric in ("bleu", "both"):
        save_bleu()
    if args.metric in ("comet", "both"):
        save_comet()


if __name__ == "__main__":
    main()
