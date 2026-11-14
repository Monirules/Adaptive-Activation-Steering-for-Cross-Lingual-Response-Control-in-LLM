#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import result_sets as R

COLORS = {
    "prompting": "#8c8c88",
    "diffmean_l20": "#2a78d6",
    "diffmean_l16": "#1baf7a",
    "tuned": "#eda100",
    "adaptive": "#eb6834",
    "tuned_adaptive": "#6250d6",
}
INK = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e4e4e0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": MUTED,
    "xtick.major.size": 0,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})


def style_axes(ax, ylabel: str, ymax: float):
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def grouped_bars(ax, groups: list, series: list, values: dict, fmt: str, label_size: float,
                 rotate: bool, width: float = 0.8):
    for gi, g in enumerate(groups):
        present = [s for s in series if values.get((g, s)) is not None]
        if not present:
            continue
        w = width / len(present)
        for si, s in enumerate(present):
            x = gi - width / 2 + w * (si + 0.5)
            v = values[(g, s)]
            ax.bar(x, v, w, color=COLORS.get(s, "#2a78d6"), edgecolor="white", linewidth=0.8)
            ax.text(x, v + (0.012 if not rotate else 0.015) * ax.get_ylim()[1], format(v, fmt),
                    ha="center", va="bottom", fontsize=label_size, color=INK,
                    rotation=90 if rotate else 0)
    ax.set_xticks(range(len(groups)))


def legend_above(ax, series: list, labels: dict, ncol: int):
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[s]) for s in series]
    ax.legend(handles, [labels[s] for s in series], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=ncol, handlelength=1.0, handleheight=1.0, columnspacing=1.2)


def save(fig, out: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def overall(model: str, track: str, method_dir: str):
    s = R.load_scores(model, track, method_dir)
    return None if s is None else s["overall"]


def fig_llama_methods(out: Path):
    series = list(R.METHODS)
    tracks = list(R.TRACKS)
    vals = {}
    for t in tracks:
        for m in series:
            o = overall(R.LLAMA, t, R.METHODS[m])
            vals[(t, m)] = None if o is None else o["lss"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style_axes(ax, "LSS (test)", 1.12)
    grouped_bars(ax, tracks, series, vals, ".3f", 7, rotate=True, width=0.84)
    ax.set_xticklabels([R.TRACK_LABELS[t] for t in tracks])
    legend_above(ax, series, R.METHOD_LABELS, ncol=3)
    save(fig, out, "llama_lss_by_method")


def fig_models(out: Path):
    models = list(R.MODEL_STEERING)
    series = ["prompting", "diffmean_l20"]
    labels = {"prompting": "Prompting", "diffmean_l20": "DiffMean steering (best global setting)"}
    vals = {}
    for m in models:
        p = overall(m, "t1", "english")
        s = overall(m, "t1", R.MODEL_STEERING[m])
        vals[(m, "prompting")] = None if p is None else p["lss"]
        vals[(m, "diffmean_l20")] = None if s is None else s["lss"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    style_axes(ax, "LSS (test, Track 1)", 1.1)
    grouped_bars(ax, models, series, vals, ".3f", 7.5, rotate=False, width=0.7)
    ax.set_xticklabels([R.MODEL_LABELS[m] for m in models])
    legend_above(ax, series, labels, ncol=2)
    save(fig, out, "models_prompting_vs_steering")


def fig_languages(out: Path, track: str, series: list, name: str):
    stores = {m: R.load_scores(R.LLAMA, track, R.METHODS[m]) for m in series}
    series = [m for m in series if stores[m] is not None]
    langs = sorted({t for m in series for t in stores[m]["by_target_language"]},
                   key=lambda c: R.LANG_NAMES.get(c, c))
    vals = {(l, m): stores[m]["by_target_language"].get(l, {}).get("lss") for l in langs for m in series}
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style_axes(ax, f"LSS (test, {R.TRACK_LABELS[track]})", 1.16)
    grouped_bars(ax, langs, series, vals, ".2f", 6.2, rotate=True, width=0.84)
    ax.set_xticklabels([R.LANG_NAMES.get(l, l) for l in langs], fontsize=8, rotation=25, ha="right",
                       rotation_mode="anchor")
    legend_above(ax, series, R.METHOD_LABELS, ncol=len(series))
    save(fig, out, name)


def fig_repetition(out: Path):
    series = list(R.METHODS)
    tracks = list(R.TRACKS)
    vals = {(t, m): R.loop_rate(t, m) for t in tracks for m in series}
    series = [m for m in series if any(vals[(t, m)] is not None for t in tracks)]
    top = max(v for v in vals.values() if v is not None)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    style_axes(ax, "Answers with repeated text (%)", top * 1.22)
    grouped_bars(ax, tracks, series, vals, ".1f", 7, rotate=True, width=0.84)
    ax.set_xticklabels([R.TRACK_LABELS[t] for t in tracks])
    legend_above(ax, series, R.METHOD_LABELS, ncol=3)
    save(fig, out, "llama_repetition_rate")


def fig_lfs_or(out: Path):
    series = list(R.METHODS)
    tracks = list(R.TRACKS)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, t in zip(axes, tracks):
        for m in series:
            o = overall(R.LLAMA, t, R.METHODS[m])
            if o is None:
                continue
            ax.scatter(o["lfs"], o["or"], s=42, color=COLORS[m], edgecolor="white", linewidth=1.2, zorder=3)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0.65, 1.02)
        ax.set_xlabel("LFS")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.text(0.04, 0.05, R.TRACK_LABELS[t], transform=ax.transAxes, fontsize=8.5, color=INK,
                fontweight="bold", va="bottom")
    axes[0].set_ylabel("OR")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=6.5, color=COLORS[m]) for m in series]
    fig.legend(handles, [R.METHOD_LABELS[m] for m in series], loc="lower center",
               bbox_to_anchor=(0.5, 0.97), ncol=3, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    save(fig, out, "llama_lfs_vs_or")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=R.RESULTS / "figures")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {args.out}")
    fig_llama_methods(args.out)
    fig_models(args.out)
    fig_languages(args.out, "t2", ["prompting", "diffmean_l20", "tuned", "tuned_adaptive"],
                  "llama_track2_by_language")
    fig_languages(args.out, "t1", ["prompting", "diffmean_l20", "adaptive"], "llama_track1_by_language")
    fig_repetition(args.out)
    fig_lfs_or(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
