# insight_scorer.py
"""
Module 1.4: Insight Scoring & User Override Handler.

Primary functions:
- compute_composite_scores(segments_meta, cfg) -> sorted list (highest first)
- run_scoring_for_run(run_id, base_dir, cfg) -> writes ranked_segments.json and topk.json
- log_user_feedback(run_id, feedback_entry, base_dir) -> appends to feedback.json and lightly updates weights

Segments input expected: list of dicts (each has .meta with 'features' and 'insight_scores')
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional

# default configuration & weights
DEFAULT_CFG = {
    "alpha_features": 0.6,
    "beta_insights": 0.4,
    "gamma_quality_penalty": 0.15,
    "min_points_for_quality": 5,
    "max_missing_allowed": 0.5,
    "top_k": 10,
    # structural feature weights (must sum to 1)
    "wF": {
        "trend_abs": 0.20,
        "seasonality": 0.15,
        "n_peaks": 0.10,
        "variance": 0.10,
        "len_pref": 0.15,
        "missing_inv": 0.10,
        "novelty": 0.20
    },
    # insight weights (8 insights; must sum to 1)
    "wI": {
        "distribution": 0.12,
        "extreme": 0.08,
        "trend": 0.12,
        "correlation": 0.10,
        "similarity": 0.10,
        "outlier": 0.10,
        "seasonality": 0.24,
        "autocorrelation": 0.14
    },
    # online update parameters
    "feedback_update_rate": 0.05,  # how much to nudge weights on user accept/reject
    "feedback_file": "feedback.json",
    # filename outputs
    "ranked_fname": "ranked_segments.json",
    "topk_fname": "topk.json"
}

# ----------------- helpers -----------------

def _normalize(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return arr
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isnan(mn) or np.isnan(mx) or (mx - mn) == 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)

def _safe_get(d, *keys, default=0.0):
    x = d
    try:
        for k in keys:
            x = x[k]
        return x if x is not None else default
    except Exception:
        return default

# ----------------- main scoring -----------------

def compute_composite_scores(segments: List[Dict[str, Any]], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Input: segments is list of metadata dicts (each has 'features' and 'insight_scores').
    Returns: same list but each dict gets 'composite_score' and is sorted descending.
    """
    C = {**DEFAULT_CFG, **(cfg or {})}
    n = len(segments)
    if n == 0:
        return []

    # Collect arrays
    trend_arr = np.zeros(n); season_arr = np.zeros(n); peaks_arr = np.zeros(n)
    var_arr = np.zeros(n); len_arr = np.zeros(n); missing_arr = np.zeros(n); novelty_arr = np.zeros(n)
    insight_keys = ["distribution","extreme","trend","correlation","similarity","outlier","seasonality","autocorrelation"]
    insight_arrs = {k: np.zeros(n) for k in insight_keys}

    for i, seg in enumerate(segments):
        f = seg.get("features", {})
        trend_arr[i] = abs(f.get("trend_slope", 0.0))
        season_arr[i] = f.get("seasonality_strength", 0.0) or 0.0
        peaks_arr[i] = f.get("n_peaks", 0.0)
        var_arr[i] = f.get("variance", 0.0)
        len_arr[i] = f.get("n_points", 0.0)
        missing_arr[i] = f.get("missing_ratio", 0.0)
        novelty_arr[i] = f.get("novelty_score", 0.0) or 0.0

        ins = seg.get("insight_scores", {})
        for k in insight_keys:
            insight_arrs[k][i] = float(ins.get(k, 0.0))

    # normalize structural features
    trend_n = _normalize(trend_arr)
    season_n = _normalize(season_arr)
    peaks_n = _normalize(peaks_arr)
    var_n = _normalize(var_arr)
    len_n = _normalize(len_arr)
    missing_n = _normalize(missing_arr)
    missing_inv_n = 1.0 - missing_n
    novelty_n = _normalize(novelty_arr)

    # normalized insight arrays (they may already be 0..1 but normalize for safety)
    insight_n = {k: _normalize(v) for k, v in insight_arrs.items()}

    # structural weighted sum
    wF = C["wF"]
    struct_score = (wF["trend_abs"] * trend_n
                    + wF["seasonality"] * season_n
                    + wF["n_peaks"] * peaks_n
                    + wF["variance"] * var_n
                    + wF["len_pref"] * len_n
                    + wF["missing_inv"] * missing_inv_n
                    + wF["novelty"] * novelty_n)

    # insight weighted sum
    wI = C["wI"]
    insight_score = np.zeros(n)
    for k in insight_keys:
        insight_score += wI.get(k, 0.0) * insight_n[k]

    # quality penalty
    quality_penalty = np.zeros(n)
    for i, seg in enumerate(segments):
        f = seg.get("features", {})
        if (f.get("n_points", 0) < C["min_points_for_quality"]) or (f.get("missing_ratio", 0.0) > C["max_missing_allowed"]):
            quality_penalty[i] = 1.0

    composite = (C["alpha_features"] * struct_score) + (C["beta_insights"] * insight_score) - (C["gamma_quality_penalty"] * quality_penalty)
    composite = np.clip(composite, 0.0, 1.0)

    # attach score and return sorted
    for i, seg in enumerate(segments):
        seg["composite_score"] = float(composite[i])

    segments_sorted = sorted(segments, key=lambda s: s.get("composite_score", 0.0), reverse=True)
    return segments_sorted

# ----------------- run-for-run helper -----------------

def run_scoring_for_run(run_id: str, base_dir: str, cfg: Optional[Dict[str, Any]] = None, wait_timeout: int = 300, poll_interval: float = 2.0) -> Dict[str, Any]:
    """
    Waits for segments.json in run folder, loads it, computes scores, writes outputs:
      - ranked_segments.json (full list with composite scores)
      - topk.json (top-K entries)
    returns dict with paths written.
    """
    C = {**DEFAULT_CFG, **(cfg or {})}
    run_folder = os.path.join(base_dir, run_id)
    segments_path = os.path.join(run_folder, "segments.json")

    start = time.time()
    # wait until segments.json exists or timeout
    while True:
        if os.path.exists(segments_path) or (time.time() - start) > wait_timeout:
            break
        time.sleep(poll_interval)

    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"{segments_path} not found after waiting")

    with open(segments_path, "r", encoding="utf-8") as fh:
        segments_list = json.load(fh)

    # segments_list is list of meta dicts; keep as-is but compute scores
    # compute_composite_scores expects list of dicts (we keep meta-level)
    scored = compute_composite_scores(segments_list, C)

    ranked_path = os.path.join(run_folder, C["ranked_fname"])
    topk_path = os.path.join(run_folder, C["topk_fname"])
    with open(ranked_path, "w", encoding="utf-8") as fh:
        json.dump(scored, fh, indent=2, default=str)
    topk = scored[: C.get("top_k", 10)]
    with open(topk_path, "w", encoding="utf-8") as fh:
        json.dump(topk, fh, indent=2, default=str)

    return {"ranked_path": ranked_path, "topk_path": topk_path, "topk_count": len(topk)}

# ----------------- feedback handler (simple) -----------------

def log_user_feedback(run_id: str, feedback: Dict[str, Any], base_dir: str, cfg: Optional[Dict[str, Any]] = None) -> str:
    """
    feedback: {
       "segment_id": int,
       "action": "accept"|"reject"|"edit",
       "notes": "...",          # optional
       "timestamp": "..."
    }
    Appends to feedback.json and performs a tiny weight adaptation if action is "accept" or "reject".
    Returns path to feedback file.
    """
    C = {**DEFAULT_CFG, **(cfg or {})}
    run_folder = os.path.join(base_dir, run_id)
    os.makedirs(run_folder, exist_ok=True)
    feedback_fp = os.path.join(run_folder, C["feedback_file"])

    # append entry
    entry = dict(feedback)
    entry["ts"] = entry.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    # write
    existing = []
    if os.path.exists(feedback_fp):
        try:
            with open(feedback_fp, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = []

    existing.append(entry)
    with open(feedback_fp, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2, default=str)

    # tiny online adaptation: nudge insight weights if accepted
    if entry.get("action") == "accept":
        # Load ranked segments to see the segment's insight_scores
        ranked_fp = os.path.join(run_folder, C["ranked_fname"])
        if os.path.exists(ranked_fp):
            try:
                with open(ranked_fp, "r", encoding="utf-8") as fh:
                    ranked = json.load(fh)
                # find segment by id
                seg = next((s for s in ranked if s.get("id") == entry.get("segment_id")), None)
                if seg:
                    ins = seg.get("insight_scores", {})
                    wI = C.get("wI", {})
                    rate = C.get("feedback_update_rate", 0.05)
                    # increase weights proportional to insight score
                    for k, v in ins.items():
                        if k in wI:
                            wI[k] = float(wI.get(k, 0.0)) + rate * float(v)
                    # renormalize
                    total = sum(wI.values()) or 1.0
                    for k in list(wI.keys()):
                        wI[k] = float(wI[k]) / total
                    # persist the adjusted weights into run folder (so run-specific tuning exists)
                    weights_fp = os.path.join(run_folder, "insight_weights.json")
                    with open(weights_fp, "w", encoding="utf-8") as fh:
                        json.dump(wI, fh, indent=2)
            except Exception:
                pass

    return feedback_fp

# ----------------- end of module -----------------
