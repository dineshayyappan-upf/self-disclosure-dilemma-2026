"""Paper condition radars."""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import behavioral as base

CATEGORIES = ["type1", "type2"]


CATEGORY_TITLES = {
    "type1": "(a) Protected (non-disability)",
    "type2": "(b) Protected (disability)",
}


DISPLAY = {**base.TAG_DISPLAY, "task_completion": "Task addressing"}


def panel_scale(summary: pd.DataFrame, category: str, tag: str):
    upper, ticks = base.task_radar_scale(category, tag)
    if upper is None:
        rows = summary[(summary.category == category) & (summary.tag == tag)]
        upper = base.radar_ylim(float(rows.pooled_rate.max()))
        ticks = list(np.linspace(0, upper, 5)[1:])
    return float(upper), ticks


def draw_figure(summary: pd.DataFrame, tag: str):
    # Seven inches matches a typical ACM two-column text width. Fixed bounds
    # retain consistent label sizes and figure dimensions across all six tags.
    fig = plt.figure(figsize=(7.0, 3.5))
    fig.suptitle(DISPLAY[tag], x=0.5, y=0.985, fontsize=10, fontweight="bold")
    styles = color_model_styles(base.COMPARISON_MODELS)
    handles = []
    for idx, category in enumerate(CATEGORIES):
        ax = fig.add_axes([0.095 + idx * 0.515, 0.215, 0.285, 0.57], polar=True)
        category_rows = summary[summary.category == category]
        conditions = base.condition_order(sorted(category_rows.condition.unique()), category)
        angles = base.radar_angles(len(conditions))
        upper, ticks = panel_scale(summary, category, tag)
        handles = []
        for model in base.COMPARISON_MODELS:
            values, _ = base.radar_values(summary, category, tag, model, conditions)
            values = values + values[:1]
            style = styles[model]
            (line,) = ax.plot(
                angles, values, color=style["color"], linewidth=1.25,
                linestyle=style["linestyle"], marker=style["marker"],
                markersize=3.6, markerfacecolor="white", markeredgewidth=0.85,
                label=base.MODEL_DISPLAY[model], clip_on=False,
            )
            ax.fill(angles, values, color=style["color"], alpha=0.075)
            handles.append(line)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([base.short_condition_label(c) for c in conditions], fontsize=7.5)
        base.align_radar_labels(ax, angles[:-1])
        ax.tick_params(axis="x", pad=7)
        ax.set_ylim(0, upper)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in ticks], fontsize=6.2)
        ax.set_rlabel_position(22)
        ax.tick_params(axis="y", pad=2)
        ax.grid(color="0.76", linewidth=0.5)
        ax.spines["polar"].set_linewidth(0.6)
        fig.text(0.2375 + idx * 0.515, 0.89, CATEGORY_TITLES[category],
                 ha="center", va="center", fontsize=8.4, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=3, frameon=False, fontsize=8.5, handlelength=2.6,
               columnspacing=2.0)
    return fig



MODEL_COLORS = {
    "haiku45": "#7B3294",
    "sonnet46": "#E69F00",
    "opus46": "#0072B2",
    "qwen3_vl_235b": "#D55E00",
    "gpt_oss_safeguard_120b": "#009E73",
}


def color_model_styles(models: list[str]) -> dict[str, dict[str, object]]:
    styles = {}
    for idx, model in enumerate(models):
        base_style = base.BAR_STYLES[idx % len(base.BAR_STYLES)]
        styles[model] = {
            "color": MODEL_COLORS.get(model, "#4C4C4C"),
            "hatch": base_style["hatch"],
            "marker": base_style["marker"],
            "linestyle": base_style["linestyle"],
        }
    return styles
