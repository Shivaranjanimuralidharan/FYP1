# sequence_chart_renderer.py
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Any

plt.switch_backend("Agg")   # enable server-side rendering


def infer_insight_type(seg):
    """
    Pick insight type from insight_scores using a tie‑break priority.
    """
    scores = seg.get("insight_scores", {})
    if not scores:
        return "trend"  # safe fallback

    # 1. Get max score value
    max_score = max(scores.values())

    # 2. Collect all insight types that have this score
    tied = [k for k, v in scores.items() if v == max_score]

    # 3. Tie‑break priority (you can adjust if needed)
    priority = [
        "trend",
        "seasonality",
        "extreme",
        "correlation",
        "similarity",
        "distribution",
        "autocorrelation",
        "outlier",
    ]

    # pick first matching by priority
    for p in priority:
        if p in tied:
            return p

    return tied[0]  # fallback

# -------------------------------------------------------------------
#  Insight-specific overlay drawing
# -------------------------------------------------------------------
def overlay_insight(ax, sub, insight_type, facts):
    """Draw annotations on chart depending on insight type."""

    if insight_type == "trend":
        slope = facts.get("slope", 0)
        intercept = facts.get("intercept", sub.iloc[0] if len(sub) > 0 else 0)
        t = np.arange(len(sub))
        trend_line = slope * t + intercept
        ax.plot(sub.index, trend_line, color="orange", linestyle="--", linewidth=2)

    elif insight_type == "seasonality":
        peaks = facts.get("season_peaks", [])
        if len(peaks):
            ax.scatter(peaks, sub.loc[peaks], color="purple", s=20)

    elif insight_type == "extreme":
        t = facts.get("extreme_time", None)
        v = facts.get("extreme_value", None)
        if t is not None:
            ax.scatter([t], [v], color="red", s=40)
            ax.axvline(t, linestyle="--", color="red")

    elif insight_type == "outlier":
        pts = facts.get("outliers", [])
        if len(pts):
            ax.scatter(pts, sub.loc[pts], color="pink", edgecolors="black", s=40)

    elif insight_type == "correlation":
        ax.axvspan(sub.index[0], sub.index[-1], color="lightblue", alpha=0.1)

    elif insight_type == "similarity":
        ax.axvspan(sub.index[0], sub.index[-1], color="yellow", alpha=0.15)

    elif insight_type == "autocorrelation":
        ax.axvspan(sub.index[0], sub.index[-1], color="lightgreen", alpha=0.12)

    elif insight_type == "distribution":
        mean = sub.mean()
        std = sub.std()
        ax.axhspan(mean - std, mean + std, color="gray", alpha=0.15)


# -------------------------------------------------------------------
#  MAIN FUNCTION – GENERATE CHARTS
# -------------------------------------------------------------------
def generate_annotated_charts(eva_sequence, raw_df, out_dir):
    """
    eva_sequence = ranked_segments.json (top-k)
    raw_df = cleaned_df.parquet
    out_dir = preproc_runs/<run_id>/annotated_charts/
    """
    os.makedirs(out_dir, exist_ok=True)

    chart_paths = []

    for idx, seg in enumerate(eva_sequence):
        measure = seg["measure"]

    # --- FIX: derive insight type dynamically ---
        insight_type = seg.get("insight_type")
        if insight_type is None:
            insight_type = infer_insight_type(seg)

        facts = seg.get("facts", {})

        start = pd.to_datetime(seg["start"])
        end = pd.to_datetime(seg["end"])

        sub = raw_df.loc[start:end, measure].dropna()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, 2))

        ax.plot(sub.index, sub.values, color="#446FA5", linewidth=1.6)

        overlay_insight(ax, sub, insight_type, facts)

        ax.set_title(f"{measure} — {insight_type.capitalize()}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.xticks(rotation=30)
        plt.tight_layout()

        filename = f"chart_{idx}_{insight_type}.png"
        fpath = os.path.join(out_dir, filename)
        fig.savefig(fpath, dpi=150)
        plt.close(fig)

        chart_paths.append(fpath)

    return chart_paths

def render_single_chart(insight, df, folder, filename="single.png"):
    """
    Minimal renderer for interactive Generate Current button.
    Uses the same logic as generate_annotated_charts but for 1 chart only.
    """
    path = os.path.join(folder, filename)

    # TODO: Replace with real annotated chart rendering
    import matplotlib.pyplot as plt

    measure = insight["measure"]
    series = df[measure]

    plt.figure(figsize=(6,3))
    plt.plot(series.index, series.values)
    plt.title(f"{insight['insight_type']} ‑ {measure}")
    plt.savefig(path)
    plt.close()

    return path
