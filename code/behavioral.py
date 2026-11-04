"""Recorded tag aggregation and radar conventions used in the paper."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

COMPARISON_MODELS = ["opus46", "qwen3_vl_235b", "gpt_oss_safeguard_120b"]


MODEL_DISPLAY = {
    "haiku45": "Haiku 4.5",
    "sonnet46": "Sonnet 4.6",
    "opus46": "Opus 4.6",
    "qwen3_vl_235b": "Qwen3",
    "gpt_oss_safeguard_120b": "GPT-Safe",
    "gpt_oss_120b": "GPT-OSS",
}


TAG_ORDER = [
    "crisis_resource",
    "professional_referral",
    "task_completion",
    "task_refusal",
    "explicit_empathy",
    "situation_acknowledgment",
]


TAG_DISPLAY = {
    "crisis_resource": "Crisis resource",
    "professional_referral": "Professional referral",
    "task_completion": "Task completion",
    "task_refusal": "Task refusal",
    "explicit_empathy": "Explicit empathy",
    "situation_acknowledgment": "Situation acknowledgment",
}


CATEGORY_ORDER = ["baseline", "subclinical", "type1", "type2"]


BAR_STYLES = [
    {"gray": "0.18", "hatch": "", "marker": "o", "linestyle": "-"},
    {"gray": "0.38", "hatch": "///", "marker": "s", "linestyle": "--"},
    {"gray": "0.58", "hatch": "\\\\\\", "marker": "D", "linestyle": ":"},
    {"gray": "0.76", "hatch": "...", "marker": "^", "linestyle": "-."},
    {"gray": "0.88", "hatch": "xxx", "marker": "v", "linestyle": (0, (3, 1, 1, 1))},
]


def wellbeing_noncontrol(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    subset = df[df["query_type"] == "wellbeing"].copy()
    subset = subset[subset["category"].isin(CATEGORY_ORDER)]
    subset = subset[subset["condition"] != "bare"]
    if models is not None:
        subset = subset[subset["model"].isin(models)]
    return subset


def tag_noncontrol(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    subset = df[df["category"].isin(CATEGORY_ORDER)].copy()
    subset = subset[subset["condition"] != "bare"]
    if models is not None:
        subset = subset[subset["model"].isin(models)]
    return subset


def summarize_tag_rates(
    df: pd.DataFrame,
    group_cols: list[str],
    average_over_cols: list[str],
) -> pd.DataFrame:
    """
    Return one row per group/tag with pooled and equal-stratum summaries.

    average_over_cols are the strata that receive equal weight in the mean and
    SD.  These columns are not statistical controls; they simply define the
    descriptive averaging unit written in each table caption.
    """
    rows = []
    for tag in TAG_ORDER:
        grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
        for key, group in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            out = dict(zip(group_cols, key_tuple))
            n = int(len(group))
            yes = int(group[tag].sum()) if n else 0
            pooled_rate = yes / n if n else np.nan

            if average_over_cols:
                stratum_rows = []
                for _, stratum in group.groupby(average_over_cols, dropna=False):
                    stratum_n = len(stratum)
                    if stratum_n:
                        stratum_rows.append(float(stratum[tag].sum() / stratum_n))
                stratum_rates = np.array(stratum_rows, dtype=float)
            else:
                stratum_rates = np.array([pooled_rate], dtype=float)

            if len(stratum_rates) == 0:
                mean_stratum_rate = np.nan
                sd_stratum_rate = np.nan
            else:
                mean_stratum_rate = float(np.mean(stratum_rates))
                sd_stratum_rate = (
                    float(np.std(stratum_rates, ddof=1))
                    if len(stratum_rates) > 1
                    else 0.0
                )

            out.update(
                {
                    "tag": tag,
                    "n": n,
                    "yes": yes,
                    "pooled_rate": pooled_rate,
                    "pooled_pct": pooled_rate * 100.0 if n else np.nan,
                    "average_over": "+".join(average_over_cols) or "pooled",
                    "n_strata": int(len(stratum_rates)),
                    "mean_stratum_rate": mean_stratum_rate,
                    "mean_stratum_pct": mean_stratum_rate * 100.0,
                    "sd_stratum_rate": sd_stratum_rate,
                    "sd_stratum_pct": sd_stratum_rate * 100.0,
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def lookup_summary(summary: pd.DataFrame, **filters: str) -> pd.Series | None:
    mask = pd.Series(True, index=summary.index)
    for col, value in filters.items():
        mask &= summary[col] == value
    rows = summary[mask]
    if len(rows) == 0:
        return None
    return rows.iloc[0]


def prep_table1_model_tag_rates(tag_df: pd.DataFrame) -> pd.DataFrame:
    data = tag_noncontrol(tag_df, COMPARISON_MODELS)
    return summarize_tag_rates(
        data,
        group_cols=["model"],
        average_over_cols=["condition", "phrasing"],
    )


def prep_table2_group_tag_rates(tag_df: pd.DataFrame) -> pd.DataFrame:
    data = tag_noncontrol(tag_df, COMPARISON_MODELS)
    group_summary = summarize_tag_rates(
        data,
        group_cols=["category"],
        average_over_cols=["model", "condition", "phrasing"],
    )
    control = tag_df[
        (tag_df["model"].isin(COMPARISON_MODELS))
        & (tag_df["category"] == "control")
        & (tag_df["condition"] == "bare")
    ].copy()
    control_summary = summarize_tag_rates(
        control,
        group_cols=["category"],
        average_over_cols=["model"],
    )
    return pd.concat([control_summary, group_summary], ignore_index=True)


DEPRESSION_FOCUS_GROUPS = [
    {
        "focus_group": "majority_neurotypical",
        "display": "Majority (neurotypical)",
        "condition": "neurotypical",
        "category": "baseline",
    },
    {
        "focus_group": "subclinical_low_mood",
        "display": "Sub-clinical (low mood)",
        "condition": "subclinical_low_mood",
        "category": "subclinical",
    },
    {
        "focus_group": "type2_depression",
        "display": "Type 2 disability (depressive disorder)",
        "condition": "depression",
        "category": "type2",
    },
]


def prep_table3_depression_phrasing_opus(tag_df: pd.DataFrame) -> pd.DataFrame:
    focus_by_condition = {
        spec["condition"]: spec
        for spec in DEPRESSION_FOCUS_GROUPS
    }
    data = tag_noncontrol(tag_df, ["opus46"])
    data = data[data["condition"].isin(focus_by_condition)].copy()
    data["focus_group"] = data["condition"].map(
        lambda condition: focus_by_condition[condition]["focus_group"]
    )
    data["focus_display"] = data["condition"].map(
        lambda condition: focus_by_condition[condition]["display"]
    )
    summary = summarize_tag_rates(
        data,
        group_cols=["focus_group", "focus_display", "phrasing"],
        average_over_cols=[],
    )
    summary["model_set"] = "opus"
    summary["models"] = "opus46"
    return summary


def short_condition_label(condition: str) -> str:
    labels = {
        "adolescent": "Adol.",
        "bisexual": "Bisex.",
        "immigrant": "Immig.",
        "racial_minority": "Race min.",
        "religious_minority": "Relig. min.",
        "transgender": "Trans.",
        "cisgender": "Cisgender",
        "citizen": "Citizen",
        "heterosexual": "Heterosexual",
        "neurotypical": "Neurotypical",
        "non-disabled": "Non-disabled",
        "politically_moderate": "Pol. moderate",
        "racial_majority": "Race maj.",
        "religious_majority": "Religion maj.",
        "working_age": "Working age",
        "working_class": "Working class",
        "subclinical_low_mood": "Low mood",
        "subclinical_worry": "Worry",
        "subclinical_focus": "Focus",
        "subclinical_distraction": "Distraction",
        "subclinical_forgetfulness": "Forgetfulness",
        "subclinical_energy_fluctuation": "Energy",
        "subclinical_social_preference": "Social pref.",
        "subclinical_habit_change": "Habit",
        "subclinical_eating_awareness": "Eating",
        "adhd": "ADHD",
        "anxiety_disorder": "Anxiety",
        "eating_disorder": "Eating dis.",
        "substance_use_disorder": "Substance",
        "chronic_pain": "Chronic pain",
        "mobility_impaired": "Mobility imp.",
        "blind": "Vision imp.",
        "union_member": "Union",
        "schizophrenia": "Schiz.",
    }
    return labels.get(condition, condition.replace("_", " ").title())


def condition_order(conditions: list[str], category: str) -> list[str]:
    preferred = {
        "baseline": [
            "working_age",
            "citizen",
            "heterosexual",
            "cisgender",
            "racial_majority",
            "religious_majority",
            "working_class",
            "politically_moderate",
            "neurotypical",
            "non-disabled",
        ],
        "type1": [
            "adolescent",
            "elderly",
            "gay",
            "lesbian",
            "bisexual",
            "transgender",
            "immigrant",
            "refugee",
            "racial_minority",
            "religious_minority",
            "poverty",
            "unhoused",
            "union_member",
        ],
        "type2": [
            "adhd",
            "autism",
            "anxiety_disorder",
            "depression",
            "bipolar",
            "schizophrenia",
            "dementia",
            "eating_disorder",
            "substance_use_disorder",
            "chronic_pain",
            "mobility_impaired",
            "blind",
        ],
    }
    order = [c for c in preferred.get(category, []) if c in conditions]
    order.extend(sorted(c for c in conditions if c not in order))
    return order


def prep_fig4_condition_radars(tag_df: pd.DataFrame) -> pd.DataFrame:
    data = wellbeing_noncontrol(tag_df, COMPARISON_MODELS)
    data = data[data["category"].isin(["baseline", "type1", "type2"])]
    return summarize_tag_rates(
        data,
        group_cols=["category", "condition", "model"],
        average_over_cols=["phrasing"],
    )


def radar_angles(n_axes: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False)
    return np.concatenate([angles, angles[:1]])


def align_radar_labels(ax: plt.Axes, angles: np.ndarray) -> None:
    for label, angle in zip(ax.get_xticklabels(), angles):
        x = math.cos(angle)
        label.set_horizontalalignment(
            "center" if abs(x) < 1e-6 else "left" if x > 0 else "right"
        )


def radar_ylim(max_rate: float) -> float:
    if max_rate <= 0.15:
        return 0.20
    if max_rate <= 0.35:
        return 0.45
    if max_rate <= 0.65:
        return 0.75
    return 1.0


def radar_values(
    summary: pd.DataFrame,
    category: str,
    tag: str,
    model: str,
    conditions: list[str],
) -> tuple[list[float], float]:
    values = []
    max_rate = 0.0
    for condition in conditions:
        row = lookup_summary(
            summary,
            category=category,
            condition=condition,
            model=model,
            tag=tag,
        )
        value = float(row["pooled_rate"]) if row is not None else 0.0
        values.append(value)
        max_rate = max(max_rate, value)
    return values, max_rate


def task_radar_scale(category: str, tag: str) -> tuple[float | None, list[float] | None]:
    if tag == "task_completion":
        return 1.0, [0.25, 0.50, 0.75, 1.0]
    if tag == "task_refusal":
        if category in ["baseline", "type1"]:
            return 0.05, [0.01, 0.02, 0.03, 0.04, 0.05]
        if category == "type2":
            return 0.10, [0.05, 0.10]
    return None, None
