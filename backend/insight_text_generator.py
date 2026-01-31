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
    Full template-based description generator
    used by:
    - Generate Current
    - RL suggestions
    """

    # Normalize insight_type
    insight_type = insight.get("insight_type", "trend")
    measure = insight["measure"]
    start = insight.get("start")
    end = insight.get("end")

    # Extract series
    try:
        series = df.loc[pd.to_datetime(start):pd.to_datetime(end), measure].dropna()
    except Exception:
        series = None

    facts = {}

    # ---------------- TREND ----------------
    if insight_type == "trend" and series is not None and len(series) > 1:
        slope = np.polyfit(np.arange(len(series)), series.values, 1)[0]
        facts["slope"] = slope
        trend_dir = "increasing" if slope > 0 else "decreasing"

        return (
            f"An {trend_dir} trend is revealed in the {measure} "
            f"between {start} and {end}."
        )

    # ---------------- DISTRIBUTION ----------------
    if insight_type == "distribution" and series is not None:
        return (
            f"A distribution of {measure} between {start} and {end} "
            f"can be seen, with a mean value of {series.mean():.2f}."
        )

    # ---------------- EXTREME ----------------
    if insight_type == "extreme" and series is not None:
        idx = series.idxmax()
        return (
            f"Among these variables, one in particular deserves attention. "
            f"The {measure} in the range {start} to {end} "
            f"is highlighted by its maximum value of {series.max():.2f}."
        )

    # ---------------- SEASONALITY ----------------
    if insight_type == "seasonality":
        num_periods = "multiple"
        if series is not None and len(series) > 10:
            diffs = series.diff().dropna()
            num_periods = int(((diffs.shift(1) > 0) & (diffs <= 0)).sum())

        return (
            f"A seasonal pattern in observed in the {measure} measure which can be observed between {start} and {end} "
            f"with {num_periods} periods present in the whole time range."
        )

    # ---------------- OUTLIER ----------------
    if insight_type == "outlier" and series is not None:
        ts = series.idxmax()
        return (
            f"Taking multiple variables into account, the value at {ts} "
            f"exhibits an outlier within the range {start} to {end} "
            f"in comparison to the rest."
        )

    # ---------------- CORRELATION ----------------
    if insight_type == "correlation":
        return (
            f"Taking multiple variables as a whole, the correlation "
            f"between {start} and {end} with respect to multiple measures is shown."
        )

    # ---------------- SIMILARITY ----------------
    if insight_type == "similarity":
        return (
            f"The {measure} between {start} and {end} "
            f"is highly similar to other measures across subspaces."
        )

    # ---------------- AUTOCORRELATION (NEW) ----------------
    if insight_type == "autocorrelation":
        return (
            f"A strong autocorrelation pattern is observed in {measure} "
            f"between {start} and {end}, indicating dependency on past values."
        )

    # ---------------- FALLBACK ----------------
    return f"A notable pattern is observed in {measure} between {start} and {end}."


