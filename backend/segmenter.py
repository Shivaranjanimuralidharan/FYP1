# segmenter.py
"""
Segmentation engine for Visail (Algorithm 3) with 8 insight-type scoring.
Usage:
    from segmenter import segment_from_run_id
    segments, segments_path = segment_from_run_id(run_id, base_dir, config=None)
Artifacts written:
    preproc_runs/<run_id>/segments.json
    preproc_runs/<run_id>/<thumbnail_folder>/*.png
"""

import os
import math
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy.signal import find_peaks, periodogram
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# try import statsmodels ACF for better autocorr; fallback is provided
try:
    from statsmodels.tsa.stattools import acf as sm_acf
except Exception:
    sm_acf = None

# ---------------------------
# Utility helpers
# ---------------------------

def time_iou(start_a, end_a, start_b, end_b):
    """Compute IoU in time (works with pd.Timestamp)."""
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    try:
        inter = max(0.0, (inter_end - inter_start).total_seconds())
        union_start = min(start_a, start_b)
        union_end = max(end_a, union_end := end_b)  # preserve naming for clarity
        union = max(1e-9, (union_end - union_start).total_seconds())
        return inter / union
    except Exception:
        # fallback to index numeric difference if timestamps incompatible
        try:
            inter = max(0.0, float(inter_end) - float(inter_start))
            union = max(1e-9, float(union_end) - float(union_start))
            return inter / union
        except Exception:
            return 0.0

def save_thumbnail(series: pd.Series, peaks_positions: List[int], outpath: str, width=200, height=80):
    """Create a small thumbnail plotting the series and peaks."""
    try:
        plt.ioff()
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        x = np.arange(len(series))
        ax.plot(x, series.values, linewidth=1)
        if peaks_positions:
            px = peaks_positions
            py = series.iloc[peaks_positions].values
            ax.scatter(px, py, s=8, c="red")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout(pad=0)
        fig.savefig(outpath, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
    except Exception:
        try:
            if os.path.exists(outpath):
                os.remove(outpath)
        except Exception:
            pass

# ---------------------------
# Core segmentation functions
# ---------------------------

def sliding_window_candidates(series: pd.Series,
                              measure: str,
                              scales: List[int],
                              step_ratio: float,
                              min_points: int,
                              max_seg_missing: float) -> List[Dict[str, Any]]:
    """Generate candidate segments via multi-scale sliding windows."""
    n = len(series)
    candidates = []
    for w in scales:
        if w <= 0 or w > n:
            continue
        step = max(1, int(math.floor(w * step_ratio)))
        start = 0
        while start + w <= n:
            seg = series.iloc[start:start + w]
            nan_ratio = float(seg.isna().mean())
            non_na = int(seg.count())
            if nan_ratio <= max_seg_missing and non_na >= min_points:
                candidate = {
                    "measure": measure,
                    "start": seg.index[0],
                    "end": seg.index[-1],
                    "window": w,
                    "raw_series": seg.copy(),  # kept temporarily for feature computation
                    "priority_score": float(seg.var()) if seg.var() is not None else 0.0
                }
                candidates.append(candidate)
            start += step
    return candidates

def compute_segment_features(seg: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute features for a single candidate segment and attach to seg['features']."""
    s = seg["raw_series"]
    sub = s.dropna()
    features = {}
    features["n_points"] = int(sub.shape[0])
    features["missing_ratio"] = float(s.isna().mean())
    features["variance"] = float(sub.var(ddof=0)) if sub.size > 0 else 0.0

    # OLS slope (use integer time indices)
    if len(sub) >= 2:
        t = np.arange(len(sub))
        try:
            a, b = np.polyfit(t, sub.values, 1)
            features["trend_slope"] = float(a)
        except Exception:
            features["trend_slope"] = 0.0
    else:
        features["trend_slope"] = 0.0

    # peaks (on smoothed series)
    try:
        window_for_smooth = min(5, max(1, int(len(s)//10)))
        smoothed = s.rolling(window=window_for_smooth, min_periods=1, center=True).median().fillna(method="ffill").fillna(method="bfill")
        pk_prom = cfg.get("peak_prominence", None)
        arr = smoothed.values
        if arr.size > 0:
            if pk_prom is None:
                pk_prom_calc = max(1e-9, (np.nanstd(arr) * 0.5))
            else:
                pk_prom_calc = pk_prom
            peaks_idx, _ = find_peaks(arr, prominence=pk_prom_calc, distance=1)
            features["n_peaks"] = int(len(peaks_idx))
            features["peaks_positions"] = [int(int(p)) for p in peaks_idx]
        else:
            features["n_peaks"] = 0
            features["peaks_positions"] = []
    except Exception:
        features["n_peaks"] = 0
        features["peaks_positions"] = []

    # outliers via MAD
    try:
        med = np.nanmedian(sub.values)
        mad = np.nanmedian(np.abs(sub.values - med)) if sub.size > 0 else 0.0
        if mad == 0:
            outlier_mask = np.zeros_like(sub.values, dtype=bool)
        else:
            z = (sub.values - med) / (mad * 1.4826)
            outlier_mask = np.abs(z) > cfg.get("outlier_mad_thresh", 3.5)
        features["outlier_count"] = int(np.sum(outlier_mask))
    except Exception:
        features["outlier_count"] = 0

    # placeholder for seasonality strength (may be filled by profiler later)
    features["seasonality_strength"] = None
    features["novelty_score"] = None

    return features

def merge_segments(segments: List[Dict[str, Any]], iou_thresh: float, merge_rule: str = "union") -> List[Dict[str, Any]]:
    """Merge overlapping segments based on time IoU. merge_rule: 'union' or 'keep_best'."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda s: s.get("priority_score", 0.0), reverse=True)
    merged: List[Dict[str, Any]] = []
    for seg in segs:
        merged_flag = False
        for i, m in enumerate(merged):
            iou = time_iou(seg["start"], seg["end"], m["start"], m["end"])
            if iou > iou_thresh:
                merged_flag = True
                if merge_rule == "union":
                    new_start = min(seg["start"], m["start"])
                    new_end = max(seg["end"], m["end"])
                    try:
                        if "raw_series" in seg and "raw_series" in m and seg["raw_series"] is not None and m["raw_series"] is not None:
                            s_concat = pd.concat([m["raw_series"], seg["raw_series"]])
                            s_concat = s_concat[~s_concat.index.duplicated(keep='first')]
                            s_concat = s_concat.sort_index()
                            new_raw = s_concat
                        else:
                            new_raw = seg.get("raw_series", m.get("raw_series"))
                    except Exception:
                        new_raw = seg.get("raw_series", m.get("raw_series"))
                    new_priority = max(seg.get("priority_score",0), m.get("priority_score",0))
                    merged[i] = {
                        "measure": seg.get("measure", m.get("measure")),
                        "start": new_start,
                        "end": new_end,
                        "window": int(((seg.get("window",0) or 0) + (m.get("window",0) or 0)) // 2),
                        "raw_series": new_raw,
                        "priority_score": new_priority
                    }
                elif merge_rule == "keep_best":
                    # keep existing, nothing to do (we process by priority)
                    pass
                break
        if not merged_flag:
            merged.append(seg.copy())
    return merged

def normalize_segment_features(segments: List[Dict[str, Any]], feature_keys: List[str]) -> None:
    """Normalize selected features across segments in-place (min-max)."""
    for fk in feature_keys:
        vals = np.array([seg.get("features", {}).get(fk, 0.0) or 0.0 for seg in segments], dtype=float)
        if vals.size == 0:
            continue
        vmin = np.nanmin(vals)
        vmax = np.nanmax(vals)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        for seg in segments:
            raw = seg.get("features", {}).get(fk, 0.0) or 0.0
            seg.setdefault("features", {})[f"{fk}_norm"] = float((raw - vmin) / denom)

# ---------------------------
# 8 Insight-type scoring helpers
# ---------------------------

def safe_div(x, y, eps=1e-12):
    return x / (y + eps)

def clip01(x):
    return float(max(0.0, min(1.0, x)))

def compute_insight_scores(seg: Dict[str, Any], df: pd.DataFrame, profiler: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute 8 insight-type scores for a candidate segment and return dict:
      distribution, extreme, trend, correlation, similarity, outlier, seasonality, autocorrelation
    Scores are bounded [0,1].
    """
    measure = seg.get("measure")
    start = seg.get("start")
    end = seg.get("end")
    raw = seg.get("raw_series") if "raw_series" in seg else None
    if raw is None or raw.size == 0:
        return {k: 0.0 for k in ["distribution","extreme","trend","correlation","similarity","outlier","seasonality","autocorrelation"]}

    series = raw.astype(float)
    sub = series.dropna()
    n = max(1, len(sub))

    # global stats (prefer profiler info)
    global_var = None
    global_std = None
    global_mean = None
    if profiler:
        pm = profiler.get("per_measure", {}).get(measure, {})
        basic = pm.get("basic_stats", {}) if isinstance(pm, dict) else {}
        global_var = basic.get("var")
        global_std = basic.get("std")
        global_mean = basic.get("mean")
    if global_var in (None, 0):
        if measure in df.columns:
            try:
                global_var = float(df[measure].var(ddof=0))
            except Exception:
                global_var = float(np.nanvar(series))
        else:
            global_var = float(np.nanvar(series))
    if global_std in (None, 0):
        if measure in df.columns:
            try:
                global_std = float(np.nanstd(df[measure].dropna().values))
            except Exception:
                global_std = float(np.nanstd(series))
        else:
            global_std = float(np.nanstd(series))
    if global_mean is None:
        try:
            global_mean = float(df[measure].dropna().mean()) if measure in df.columns else float(np.nanmean(series))
        except Exception:
            global_mean = float(np.nanmean(series))

    # 1) Distribution: relative variance (soft mapping)
    seg_var = float(np.nanvar(sub.values)) if sub.size > 0 else 0.0
    dist_score = clip01(safe_div(seg_var, (global_var if global_var else 1.0)))
    distribution = clip01(np.sqrt(dist_score))

    # 2) Extreme: max deviation from median normalized by global std
    med = float(np.nanmedian(sub.values)) if sub.size > 0 else 0.0
    max_dev = float(np.max(np.abs(sub.values - med))) if sub.size > 0 else 0.0
    extreme = clip01(safe_div(max_dev, (3.0 * (global_std if global_std and global_std>0 else 1.0))))

    # 3) Trend: slope magnitude * length normalized
    trend_slope = seg.get("features", {}).get("trend_slope", 0.0)
    trend_raw = abs(trend_slope) * n
    trend = clip01(safe_div(trend_raw, (3.0 * (global_std if global_std and global_std>0 else 1.0))))

    # 4) Correlation: max abs Pearson corr with other measures over same time window
    corr_score = 0.0
    if measure in df.columns:
        try:
            window_df = df.loc[start:end]
            for other in window_df.columns:
                if other == measure:
                    continue
                a = window_df[measure].dropna()
                b = window_df[other].dropna()
                idx = a.index.intersection(b.index)
                if len(idx) >= 3:
                    try:
                        r = pearsonr(a.loc[idx].values, b.loc[idx].values)[0]
                        corr_score = max(corr_score, abs(float(r)))
                    except Exception:
                        pass
        except Exception:
            corr_score = 0.0
    correlation = clip01(corr_score)

    # 5) Similarity: self-similarity by splitting segment halves
    similarity = 0.0
    if len(sub) >= 4:
        half = len(sub) // 2
        left = sub.iloc[:half]
        right = sub.iloc[-half:]
        L = min(len(left), len(right))
        if L >= 3:
            try:
                rl = left.iloc[-L:].values
                rr = right.iloc[:L].values
                r = pearsonr(rl, rr)[0]
                similarity = clip01(abs(float(r)))
            except Exception:
                similarity = 0.0

    # 6) Outlier: fraction of outliers (scaled)
    outlier_count = seg.get("features", {}).get("outlier_count", 0)
    outlier = clip01(safe_div(outlier_count, max(1, n)) * 2.0)

    # 7) Seasonality: prefer profiler's STL strength else periodogram top/total
    seasonality = 0.0
    if profiler:
        pm = profiler.get("per_measure", {}).get(measure, {})
        stl = pm.get("stl", {}) if isinstance(pm, dict) else {}
        if stl and stl.get("seasonality_strength") is not None:
            try:
                seasonality = clip01(float(stl.get("seasonality_strength")))
            except Exception:
                seasonality = 0.0
    if seasonality == 0.0:
        try:
            f, Pxx = periodogram(sub.values, detrend="linear", scaling="spectrum")
            mask = f > 0
            P = Pxx[mask] if mask.any() else Pxx
            if P.size > 0:
                top = float(np.max(P))
                total = float(np.sum(P))
                if total > 0:
                    seasonality = clip01(top / total)
        except Exception:
            seasonality = 0.0

    # 8) Autocorrelation: use statsmodels acf if available else lag-1 corr
    acf_score = 0.0
    try:
        if sm_acf is not None:
            L = min(cfg.get("acf_max_lag", 20), max(1, int(len(sub)//2)))
            acfs = sm_acf(sub.values, nlags=L, fft=True)
            if len(acfs) > 1:
                acf_score = float(np.nanmax(np.abs(acfs[1:])))
        else:
            if len(sub) >= 2:
                a = sub.values[:-1]
                b = sub.values[1:]
                r = np.corrcoef(a, b)[0,1]
                acf_score = float(abs(r))
    except Exception:
        acf_score = 0.0
    autocorrelation = clip01(acf_score)

    return {
        "distribution": distribution,
        "extreme": extreme,
        "trend": trend,
        "correlation": correlation,
        "similarity": similarity,
        "outlier": outlier,
        "seasonality": seasonality,
        "autocorrelation": autocorrelation
    }

# ---------------------------
# High-level orchestration
# ---------------------------

DEFAULT_CONFIG = {
    "scales": [30, 60, 120],       # window sizes in points
    "step_ratio": 0.5,
    "min_points": 10,
    "max_seg_missing": 0.2,
    "use_changepoint": False,
    "cp_penalty": None,
    "min_cp_length": 10,
    "peak_prominence": None,
    "merge_iou": 0.5,
    "merge_rule": "union",
    "outlier_mad_thresh": 3.5,
    "thumbnail_folder": "thumbnails",
    "acf_max_lag": 20
}

def segment_from_dataframe(df: pd.DataFrame, profiler: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Main entry: accept cleaned dataframe and profiler and return list of segment metadata dicts."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    scales = cfg["scales"]
    step_ratio = cfg["step_ratio"]
    min_points = cfg["min_points"]
    max_seg_missing = cfg["max_seg_missing"]

    all_candidates: List[Dict[str, Any]] = []

    for measure in df.columns:
        series = df[measure]
        if len(series.dropna()) < min_points:
            continue
        candidates = sliding_window_candidates(series, measure, scales, step_ratio, min_points, max_seg_missing)
        all_candidates.extend(candidates)

    # optional change-point branch: left as hook (user can supply CPD implementation)
    if cfg.get("use_changepoint", False):
        # Placeholder for CPD-based segments (e.g., using ruptures). Not implemented by default.
        pass

    # Compute per-segment features, insight scores & prepare thumbnails later
    segments: List[Dict[str, Any]] = []
    for i, c in enumerate(all_candidates):
        seg = c.copy()
        seg["features"] = compute_segment_features(seg, cfg)
        seg["insight_scores"] = compute_insight_scores(seg, df, profiler, cfg)
        seg["_thumbnail_temp"] = None
        segments.append(seg)

    # Merge overlapping segments
    merged = merge_segments(segments, iou_thresh=cfg["merge_iou"], merge_rule=cfg["merge_rule"])

    # Recompute features & insight scores for merged segments where raw_series may have changed
    for seg in merged:
        if "raw_series" in seg and seg["raw_series"] is not None:
            seg["features"] = compute_segment_features(seg, cfg)
            seg["insight_scores"] = compute_insight_scores(seg, df, profiler, cfg)

    # Normalize selected features across segments
    feature_keys = ["variance", "n_points", "n_peaks", "outlier_count", "trend_slope"]
    normalize_segment_features(merged, feature_keys)

    # Build final metadata list (drop heavy raw_series, include features & insight_scores)
    final_segments = []
    for idx, seg in enumerate(merged):
        meta = {
            "id": idx,
            "measure": seg.get("measure"),
            "start": str(seg.get("start")),
            "end": str(seg.get("end")),
            "window": int(seg.get("window") or 0),
            "priority_score": float(seg.get("priority_score") or 0.0),
            "features": seg.get("features", {}),
            "insight_scores": seg.get("insight_scores", {}),
            "thumbnail": None
        }
        final_segments.append({"meta": meta, "_raw_series": seg.get("raw_series")})

    return final_segments

def segment_from_run_id(run_id: str, base_dir: str, config: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load cleaned_df.parquet and profiler.json from base_dir/run_id and compute segments.
    Saves segments.json and thumbnails into run folder. Returns (segments_list, segments_json_path).
    """
    run_out = os.path.join(base_dir, run_id)
    cleaned_path = os.path.join(run_out, "cleaned_df.parquet")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError("cleaned_df.parquet not found for run_id")

    # load dataframe
    df = pd.read_parquet(cleaned_path)

    # load profiler if available
    profiler = None
    profiler_path = os.path.join(run_out, "profiler.json")
    if os.path.exists(profiler_path):
        try:
            with open(profiler_path, "r", encoding="utf-8") as fh:
                profiler = json.load(fh)
        except Exception:
            profiler = None

    segments_wrapped = segment_from_dataframe(df, profiler=profiler, config=config)

    # create thumbnails folder
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    thumb_dir = os.path.join(run_out, cfg.get("thumbnail_folder", DEFAULT_CONFIG["thumbnail_folder"]))
    os.makedirs(thumb_dir, exist_ok=True)

    # save thumbnails and prepare final list
    final_meta_list = []
    for seg in segments_wrapped:
        meta = seg["meta"]
        raw = seg.get("_raw_series")
        if isinstance(raw, pd.Series):
            peaks = meta.get("features", {}).get("peaks_positions", []) if "features" in meta else []
            safe_start = meta["start"].replace(":", "_").replace(" ", "_")
            safe_end = meta["end"].replace(":", "_").replace(" ", "_")
            fname = f"segment_{meta['measure']}_{safe_start}_{safe_end}.png"
            outpath = os.path.join(thumb_dir, fname)
            try:
                save_thumbnail(raw, peaks, outpath)
                meta["thumbnail"] = os.path.relpath(outpath, run_out)
            except Exception:
                meta["thumbnail"] = None
        else:
            meta["thumbnail"] = None
        final_meta_list.append(meta)

    # save segments.json
    segments_path = os.path.join(run_out, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as fh:
        json.dump(final_meta_list, fh, indent=2, default=str)

    return final_meta_list, segments_path

# If executed directly for debugging
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--base_dir", default="preproc_runs")
    args = parser.parse_args()
    segs, p = segment_from_run_id(args.run_id, args.base_dir)
    print("Saved segments to", p)
