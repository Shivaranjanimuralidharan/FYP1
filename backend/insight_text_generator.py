# insight_text_generator.py
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Templates for description generation
# -------------------------------------------------------------------
TEMPLATES = {
    "trend": "Between {start} and {end}, {measure} shows a {trend_dir} trend with slope {slope:.3f}.",
    "seasonality": "A seasonal repeating pattern appears between {start} and {end}, with dominant period about {period} days.",
    "extreme": "{measure} reaches an extreme value of {extreme_value} on {extreme_time}.",
    "outlier": "{measure} contains outliers at time points: {outlier_times}.",
    "correlation": "{measure} shows correlation with {correlates} in this interval.",
    "similarity": "The segment resembles past subsequences with a similarity window of ~{similarity_period} days.",
    "autocorrelation": "Strong autocorrelation is observed at lag {lag}.",
    "distribution": "Within this range, {measure} has mean {mean:.2f} and variance {variance:.2f}."
}


# -------------------------------------------------------------------
# MAIN FUNCTION – generate human-readable text descriptions
# -------------------------------------------------------------------
def generate_text_descriptions(eva_sequence):
    descriptions = []

    for seg in eva_sequence:
        insight_type = seg.get("insight_type")
        if not insight_type:
            insight_type = infer_insight_type(seg)
            seg["insight_type"] = insight_type  # standardize so renderer can reuse it

        template = TEMPLATES.get(insight_type, "{measure} shows interesting behaviour between {start} and {end}.")

        facts = seg.get("facts", {})
        # ---------------------------------------------------------
        # AUTO-FILL missing facts using segment raw data
        # ---------------------------------------------------------
        measure = seg["measure"]
        start = seg["start"]
        end = seg["end"]

        # try loading from df if provided in seg
        series = None
        df = seg.get("_df")   # optional injection
        if df is not None:
            try:
                series = df.loc[start:end, measure].dropna()
            except Exception:
                series = None

        # ========== COMPUTE SLOPE ==========
        if "slope" not in facts:
            if series is not None and len(series) > 1:
                x = np.arange(len(series))
                y = series.values
                slope = np.polyfit(x, y, 1)[0]
                facts["slope"] = float(slope)
            else:
                facts["slope"] = 0.0

        # ========== EXTREME VALUE ==========
        if "extreme_value" not in facts or "extreme_time" not in facts:
            if series is not None and len(series) > 0:
                idx = series.idxmax()
                facts["extreme_value"] = float(series.max())
                facts["extreme_time"] = str(idx)
            else:
                facts["extreme_value"] = "N/A"
                facts["extreme_time"] = "N/A"

        # ========== MEAN & VARIANCE ==========
        if "mean" not in facts:
            facts["mean"] = float(series.mean()) if series is not None else 0.0
        if "variance" not in facts:
            facts["variance"] = float(series.var()) if series is not None else 0.0

        # ========== PERIOD (fallback for seasonality) ==========
        if "period" not in facts:
            facts["period"] = "N/A"

        # save updated facts back into segment
        seg["facts"] = facts


        desc = template.format(
            start=seg["start"],
            end=seg["end"],
            measure=seg["measure"],
            trend_dir="increasing" if facts.get("slope", 0) > 0 else "decreasing",
            slope=facts.get("slope", 0),
            period=facts.get("period", "N/A"),
            extreme_value=facts.get("extreme_value", "N/A"),
            extreme_time=facts.get("extreme_time", "N/A"),
            outlier_times=facts.get("outliers", []),
            correlates=facts.get("correlates", "other measures"),
            similarity_period=facts.get("similarity_period", "N/A"),
            lag=facts.get("autocorr_lag", "N/A"),
            mean=facts.get("mean", 0),
            variance=facts.get("variance", 0)
        )

        descriptions.append(desc)

    return descriptions

def generate_single_description(insight, df):
    """
    Minimal text generator for single insight (interactive mode).
    Replace this later with full template‑based Module 3.2 logic.
    """
    measure = insight["measure"]
    ins_type = insight["insight_type"]
    subspace = insight.get("subspace")

    return f"{ins_type} insight detected on {measure} at range {subspace}."

def infer_insight_type(seg):
    scores = seg.get("insight_scores", {})
    if not scores:
        return "trend"  # safe fallback

    # Find max-scoring insight(s)
    max_score = max(scores.values())
    best = [k for k, v in scores.items() if v == max_score]

    # Resolve ties deterministically
    priority = [
        "extreme",
        "trend",
        "outlier",
        "seasonality",
        "correlation",
        "similarity",
        "autocorrelation",
        "distribution"
    ]

    for p in priority:
        if p in best:
            return p

    return best[0]  # final fallback

