# time_series_profiler.py
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from typing import Dict, Any, List, Optional
import math

# ---------- Helpers ----------

def robust_mad_zscore(series: pd.Series):
    """Return MAD-based z-scores for a series (NaNs preserved)."""
    s = series.dropna()
    if s.empty:
        return pd.Series(index=series.index, dtype=float)
    med = np.median(s)
    mad = np.median(np.abs(s - med))
    if mad == 0:
        # fallback to std zscore
        std = s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0
        z = (series - med) / std
        return z
    else:
        return (series - med) / (mad * 1.4826)  # approx conversion to std

def detect_peaks(series: pd.Series, distance: int = 1, prominence: float = None):
    """Return indices of peaks and prominences using scipy.find_peaks."""
    arr = series.fillna(method="ffill").fillna(method="bfill").values
    if len(arr) == 0:
        return []
    # choose prominence if not given
    if prominence is None:
        prominence = (np.nanstd(arr) + 1e-9) * 0.5
    peaks, props = find_peaks(arr, distance=distance, prominence=prominence)
    prominences = props.get("prominences", np.repeat(np.nan, len(peaks)))
    return [{"pos": int(p), "value": float(arr[p]), "prominence": float(pr)} for p, pr in zip(peaks, prominences)]

def ols_trend(series: pd.Series):
    """Fit linear OLS trend (a * t + b) on non-null values; return (a, b) and trend series aligned to original index."""
    s = series.dropna()
    if len(s) < 2:
        return (0.0, float(s.mean() if not s.empty else 0.0)), pd.Series(index=series.index, data=np.nan)
    # use integer time steps from 0..n-1 for stability
    t = np.arange(len(s))
    a, b = np.polyfit(t, s.values, 1)  # slope, intercept in value space vs index
    # create a trend aligned to original index (interpolated for missing points)
    full_t = pd.Series(np.nan, index=series.index)
    # map index positions for non-nulls
    idxs = np.where(~series.isna())[0]
    trend_vals = a * np.arange(len(idxs)) + b
    full_t.iloc[idxs] = trend_vals
    return (float(a), float(b)), full_t

def strongest_periods(series: pd.Series, fs=1.0, top_n=3):
    """
    Compute period candidates via periodogram.
    fs: sampling frequency (1 / sample_interval) — for regularly sampled data assume 1.
    Returns list of (period, power) sorted by power desc.
    """
    s = series.dropna()
    if len(s) < 3:
        return []
    f, Pxx = signal.periodogram(s.values, fs=fs, detrend="linear", scaling="spectrum")
    # ignore DC (f==0)
    mask = f > 0
    f = f[mask]
    Pxx = Pxx[mask]
    if len(f) == 0:
        return []
    # convert frequency -> period (1/f)
    # avoid zero-frequency; sort by power
    idx = np.argsort(Pxx)[::-1]
    res = []
    for i in idx[:top_n]:
        if Pxx[i] <= 0 or f[i] == 0:
            continue
        period = 1.0 / f[i]
        res.append((float(period), float(Pxx[i])))
    return res

def seasonality_strength(stl_resid, stl_trend, stl_seasonal):
    """
    A simple measure: var(seasonal) / (var(resid) + var(seasonal) + var(trend))
    (Higher → stronger seasonality).
    """
    v_season = np.nanvar(stl_seasonal)
    v_resid = np.nanvar(stl_resid)
    v_trend = np.nanvar(stl_trend)
    denom = v_season + v_resid + v_trend
    if denom == 0:
        return 0.0
    return float(v_season / denom)

def compute_acf_and_adf(series: pd.Series, nlags: int = 40):
    """Compute ACF up to nlags and run Augmented Dickey-Fuller test."""
    s = series.dropna()
    result = {"acf": [], "adf": {"stat": None, "pvalue": None, "usedlag": None, "nobs": None}}
    if len(s) < 3:
        return result
    try:
        acf_vals = acf(s.values, nlags=min(nlags, len(s)-1), fft=True)
        result["acf"] = [float(x) for x in acf_vals]
    except Exception:
        result["acf"] = []
    try:
        adf_res = adfuller(s.values, autolag="AIC")
        result["adf"] = {"stat": float(adf_res[0]), "pvalue": float(adf_res[1]), "usedlag": int(adf_res[2]), "nobs": int(adf_res[3])}
    except Exception:
        result["adf"] = {"stat": None, "pvalue": None, "usedlag": None, "nobs": None}
    return result

# ---------- Main profiler ----------

def profile_timeseries(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Profile a cleaned, time-indexed DataFrame.
    Input: df (index must be datetime-like), config (dict)
    Output: profiler dict P
    """
    # Default configuration
    default_C = {
        "min_points": 20,
        "stl_period": None,         # If None, STL will infer (or we'll feed seasonal=13 for weekly-like)
        "period_candidates_topn": 3,
        "acf_max_lag": 40,
        "peak_min_distance": 3,
        "peak_prominence": None,    # if None, computed adaptively per series
        "outlier_mad_thresh": 3.5,
        "downsample_for_periodogram": 0,  # 0 means no downsample
    }
    C = default_C if config is None else {**default_C, **config}

    P: Dict[str, Any] = {"per_measure": {}, "summary": {}}

    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # Can't ensure time index — still allow profiling but warn in summary
            time_index_ok = False
        else:
            time_index_ok = True
    else:
        time_index_ok = True

    measures = list(df.columns)
    global_stats = {
        "measures_count": len(measures),
        "skipped_measures": 0,
        "total_points": int(df.size),
        "time_index_ok": time_index_ok,
    }

    for m in measures:
        s: pd.Series = df[m]
        snz = s.dropna()
        if len(snz) < C["min_points"]:
            P["per_measure"][m] = {"skipped": True, "reason": "too_short", "n": int(len(snz))}
            global_stats["skipped_measures"] += 1
            continue

        # Basic stats
        basic = {}
        basic["n"] = int(len(snz))
        basic["mean"] = float(snz.mean())
        basic["median"] = float(snz.median())
        basic["std"] = float(snz.std(ddof=0))
        basic["var"] = float(snz.var(ddof=0))
        basic["min"] = float(snz.min())
        basic["max"] = float(snz.max())
        basic["skew"] = float(stats.skew(snz.values)) if len(snz) > 2 else None
        basic["kurtosis"] = float(stats.kurtosis(snz.values)) if len(snz) > 3 else None
        basic["missing_ratio"] = float(s.isna().mean())

        # Trend via OLS
        (slope_int, intercept), trend_series = ols_trend(s)
        # detrended (only where trend exists)
        sdetr = s.copy()
        mask_nonnull = ~s.isna()
        sdetr[mask_nonnull] = s[mask_nonnull] - trend_series[mask_nonnull]

        # Period candidates via periodogram (regular sampling assumed)
        # optionally downsample for speed
        period_candidates = strongest_periods(s, fs=1.0, top_n=C["period_candidates_topn"])

        # STL decomposition (if enough points). Use candidate period if provided, else fallback.
        stl_res = {"trend": [], "seasonal": [], "resid": [], "seasonality_strength": None}
        try:
            # choose period for STL: if provided in config use it; else try first candidate period rounded.
            stl_period = C["stl_period"]
            if stl_period is None and len(period_candidates) > 0:
                # use nearest integer > 1
                candidate = int(round(period_candidates[0][0]))
                if candidate >= 2:
                    stl_period = candidate
            # if still None and index frequency exists and is seasonal choose 7 or 12 heuristics could be used,
            # but we proceed only if stl_period is valid integer >=2
            if stl_period is not None and stl_period >= 2 and len(snz) >= max(2*stl_period, 10):
                stl = STL(snz, period=stl_period, robust=True)
                stlres = stl.fit()
                stl_trend = stlres.trend
                stl_seasonal = stlres.seasonal
                stl_resid = stlres.resid
                stl_res = {
                    "trend": stl_trend.tolist(),
                    "seasonal": stl_seasonal.tolist(),
                    "resid": stl_resid.tolist(),
                    "seasonality_strength": seasonality_strength(stl_resid, stl_trend, stl_seasonal),
                    "period": stl_period,
                }
        except Exception:
            # safe fallback: leave stl_res as default
            stl_res = stl_res

        # ACF and ADF
        acf_adf = compute_acf_and_adf(s, nlags=C["acf_max_lag"])

        # Smoothing and peak detection
        try:
            ssmooth = s.rolling(window=3, min_periods=1, center=True).median()
        except Exception:
            ssmooth = s.copy()
        peaks = detect_peaks(ssmooth.fillna(method="ffill").fillna(method="bfill"),
                             distance=C["peak_min_distance"],
                             prominence=C["peak_prominence"])

        # Outliers via MAD z-score
        zscore = robust_mad_zscore(s)
        mad_thresh = C["outlier_mad_thresh"]
        outlier_mask = (zscore.abs() > mad_thresh)
        outliers = []
        for idx in np.where(outlier_mask)[0]:
            # idx is positional index, map to label
            try:
                label = s.index[idx]
                outliers.append({"pos": int(idx), "index": str(label), "value": float(s.iloc[idx]), "z": float(zscore.iloc[idx])})
            except Exception:
                pass

        # Volatility, peak density, SNR
        volatility = float(sdetr.std(ddof=0))
        peak_density = float(len(peaks) / max(1.0, len(snz)))
        signal_power = float(np.nanvar(snz.values))
        noise_power = float(np.nanvar(sdetr.dropna().values)) if len(sdetr.dropna()) > 0 else 0.0
        snr = float(signal_power / (noise_power + 1e-12))

        # Trend direction
        trend_direction = "up" if slope_int > 0 else ("down" if slope_int < 0 else "flat")

        # Data quality heuristics
        quality = {}
        quality["missing_ratio"] = basic["missing_ratio"]
        quality["n_points"] = basic["n"]
        quality["var_zero"] = (basic["var"] == 0.0)

        # Assemble per-measure profile
        measure_profile = {
            "skipped": False,
            "basic_stats": basic,
            "ols_trend": {"slope": float(slope_int), "intercept": float(intercept)},
            "detrended_std": float(volatility),
            "detrended_sample_count": int(sdetr.dropna().shape[0]),
            "period_candidates": [{"period": p, "power": pw} for (p, pw) in period_candidates],
            "stl": stl_res,
            "acf_adf": acf_adf,
            "peaks": peaks,
            "outliers": outliers,
            "volatility": volatility,
            "peak_density": peak_density,
            "snr": snr,
            "trend_direction": trend_direction,
            "quality": quality,
        }

        P["per_measure"][m] = measure_profile

    # Global summary across measures
    # compute simple aggregated metrics
    n_profiles = len(P["per_measure"])
    vars_zero = sum(1 for mv in P["per_measure"].values() if (not mv.get("skipped", True)) and mv["basic_stats"]["var"] == 0.0)
    avg_missing = np.mean([mv["basic_stats"]["missing_ratio"] for mv in P["per_measure"].values() if (not mv.get("skipped", True))]) if n_profiles > 0 else 0.0
    avg_snr = np.mean([mv["snr"] for mv in P["per_measure"].values() if (not mv.get("skipped", True))]) if n_profiles > 0 else 0.0

    P["summary"] = {
        "time_index_ok": time_index_ok,
        "measures_analyzed": n_profiles,
        "measures_skipped": global_stats["skipped_measures"],
        "measures_zero_variance": int(vars_zero),
        "avg_missing_ratio": float(avg_missing),
        "avg_snr": float(avg_snr),
    }

    return P

# ---------- Integrated Usage (NO standalone running) ----------

import os

def profile_from_run_id(run_id: str, base_dir: str) -> dict:
    """
    Automatically fetch cleaned_df.parquet from:
        <base_dir>/<run_id>/cleaned_df.parquet
    and run the profiler on it.

    Parameters:
        run_id: str - ID created during preprocessing
        base_dir: str - path to preproc_runs directory

    Returns:
        profiler dict (same structure as profile_timeseries)
    """
    cleaned_path = os.path.join(base_dir, run_id, "cleaned_df.parquet")

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"cleaned_df.parquet not found for run_id={run_id}")

    # Load cleaned dataset
    df = pd.read_parquet(cleaned_path)

    # Run profiling
    profiler = profile_timeseries(df)

    return profiler

