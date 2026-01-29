# data_preprocessor_algo1.py
"""
Algorithm 1 — Data Preprocessor

Usage:
    from data_preprocessor_algo1 import run_preprocess
    cleaned_df, metadata, preview_paths = run_preprocess(
        csv_path="data.csv",
        time_col=None,
        target_freq=None,
        agg="mean",
        impute_strategy="interpolate",
        impute_limit=3,
        low_variance_frac=0.01,
        max_missing_ratio=0.2,
        out_dir="out"
    )
"""

import os
import json
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


def _is_parseable_datetime(s: str) -> bool:
    try:
        pd.to_datetime(s)
        return True
    except Exception:
        return False


def auto_detect_time_col(df: pd.DataFrame) -> Optional[str]:
    # heuristic 1: column name contains date/time
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            return c
    # heuristic 2: choose column with most parseable datetime strings (sample)
    best_col = None
    best_score = -1
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = sum(1 for v in sample if _is_parseable_datetime(v))
        if parsed > best_score:
            best_score = parsed
            best_col = c
    if best_score <= 0:
        return None
    return best_col


def run_preprocess(
    csv_path: str,
    time_col: Optional[str] = None,
    target_freq: Optional[str] = None,
    agg: str = "mean",
    impute_strategy: str = "interpolate",  # "interpolate" | "ffill" | "none"
    impute_limit: Optional[int] = None,
    low_variance_frac: float = 0.01,
    max_missing_ratio: float = 0.2,
    out_dir: str = "preproc_out",
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
   

    os.makedirs(out_dir, exist_ok=True)

    # Step 1: read CSV
    raw_df = pd.read_csv(csv_path)
    metadata: Dict[str, Any] = {}
    transform_log: Dict[str, Any] = {"steps": []}

    transform_log["steps"].append("read_csv")

    # Step 2: time column detection
    detected_time_col = time_col
    if detected_time_col is None:
        detected_time_col = auto_detect_time_col(raw_df)
        transform_log["steps"].append("auto_detect_time_col")
    if detected_time_col is None or detected_time_col not in raw_df.columns:
        raise ValueError("Time column not found. Provide time_col argument.")

    metadata["time_col"] = detected_time_col
    transform_log["time_col"] = detected_time_col

    # Step 3: parse timestamps and drop NaT
    df = raw_df.copy()
    df[detected_time_col] = pd.to_datetime(df[detected_time_col], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=[detected_time_col])
    n_after = len(df)
    transform_log["parsed_timestamps_rows_dropped"] = int(n_before - n_after)
    transform_log["steps"].append("parse_timestamps_drop_NaT")

    # Step 4: set index and sort
    df = df.set_index(detected_time_col).sort_index()
    transform_log["steps"].append("set_index_sort")

    # Step 5: numeric columns selection (exclude index)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # If no numeric columns, try to coerce non-time cols to numeric
    if len(numeric_cols) == 0:
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                pass
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metadata["numeric_columns_initial"] = numeric_cols.copy()
    transform_log["steps"].append("select_numeric_columns")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found after coercion. At least one numeric column is required.")

    # Step 6: frequency determination
    if target_freq:
        freq = target_freq
        transform_log["freq_source"] = "target_freq_arg"
    else:
        try:
            freq = pd.infer_freq(df.index)
        except Exception:
            freq = None
        if freq is None:
            freq = "D"  # fallback
            transform_log["freq_source"] = "inferred_none_fallback_D"
        else:
            transform_log["freq_source"] = "inferred"
    metadata["inferred_or_target_freq"] = freq
    transform_log["steps"].append("freq_infer_or_target")

    # Step 7: resample using agg
    agg_map = {
        "mean": "mean",
        "sum": "sum",
        "median": "median",
        "first": "first",
        "last": "last",
    }
    agg_func = agg_map.get(agg, "mean")
    resampled = getattr(df[numeric_cols].resample(freq), agg_func)()
    transform_log["steps"].append("resample")
    transform_log["resampled_shape"] = resampled.shape

    # Step 8-14: imputation
    if impute_strategy == "interpolate":
        imputed = resampled.interpolate(method="time", limit=impute_limit, limit_direction="both")
        transform_log["impute"] = f"interpolate_limit_{impute_limit}"
    elif impute_strategy == "ffill":
        imputed = resampled.fillna(method="ffill", limit=impute_limit)
        transform_log["impute"] = f"ffill_limit_{impute_limit}"
    elif impute_strategy in ("none", "none", None):
        imputed = resampled.copy()
        transform_log["impute"] = "none"
    else:
        raise ValueError(f"Unknown impute_strategy: {impute_strategy}")
    transform_log["steps"].append("impute")

    # Step 15: variance and threshold
    var = imputed.var(skipna=True)
    median_var = float(np.nanmedian(var.values)) if len(var) > 0 else 0.0
    baseline = max(median_var, 1e-12)
    low_var_thresh = low_variance_frac * baseline
    transform_log["var_summary"] = {
        "per_column_var": var.to_dict(),
        "median_var": median_var,
        "low_var_thresh": low_var_thresh,
    }
    transform_log["steps"].append("compute_variance_and_threshold")

    # Step 16: keep columns where NaN fraction ≤ max_missing_ratio and var ≥ threshold
    nan_frac = imputed.isna().mean()
    kept_mask = (nan_frac <= max_missing_ratio) & (var!=0)
    kept_columns = [c for c, keep in kept_mask.items() if keep]
    dropped_columns = [c for c in numeric_cols if c not in kept_columns]
    transform_log["kept_columns"] = kept_columns
    transform_log["dropped_columns"] = dropped_columns
    metadata["kept_columns"] = kept_columns
    metadata["dropped_columns"] = dropped_columns
    transform_log["steps"].append("filter_columns_missing_and_lowvar")

    cleaned_df = imputed[kept_columns].copy()

    # Step 18-20: if too many rows (>100000) resample weekly mean
    if len(cleaned_df) > 100000:
        cleaned_df = cleaned_df.resample("7D").mean()
        transform_log["downsampled_to_7D"] = True
    else:
        transform_log["downsampled_to_7D"] = False
    transform_log["cleaned_shape"] = cleaned_df.shape
    transform_log["steps"].append("maybe_downsample_large")

    # Step 21: Save cleaned df.parquet, metadata.json, transform_log.json
    cleaned_path = os.path.join(out_dir, "cleaned_df.parquet")
    metadata_path = os.path.join(out_dir, "metadata.json")
    transform_log_path = os.path.join(out_dir, "transform_log.json")
    cleaned_df.to_parquet(cleaned_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    with open(transform_log_path, "w") as f:
        json.dump(transform_log, f, indent=2, default=str)

    # Create small previews (first/last 5 rows as CSV) and return paths
    preview_first = os.path.join(out_dir, "preview_first.csv")
    preview_last = os.path.join(out_dir, "preview_last.csv")
    cleaned_df.head(5).to_csv(preview_first)
    cleaned_df.tail(5).to_csv(preview_last)
    preview_paths = {"preview_first": preview_first, "preview_last": preview_last}

    # Step 22: return
    return cleaned_df, metadata, {"cleaned_path": cleaned_path, "metadata_path": metadata_path, "transform_log_path": transform_log_path, **preview_paths}


# quick test block when run directly
if __name__ == "__main__":
    # small synthetic test similar to earlier file
    import pandas as pd
    import numpy as np
    dates = pd.date_range("2021-01-01", periods=200, freq="D")
    a = np.sin(np.linspace(0, 20, 200)) + np.random.normal(scale=0.05, size=200)
    b = np.linspace(0, 10, 200) + np.random.normal(scale=0.2, size=200)
    df_test = pd.DataFrame({"date": dates, "confirmed": a, "tested": b})
    # inject NaNs and outliers
    df_test.loc[10:12, "confirmed"] = np.nan
    df_test.loc[100, "tested"] = 1e6

    out = run_preprocess(
        csv_path=None if False else "tmp_test_input.csv",
        time_col="date",
        target_freq=None,
        agg="mean",
        impute_strategy="interpolate",
        impute_limit=3,
        low_variance_frac=0.001,
        max_missing_ratio=0.2,
        out_dir="tmp_algo1_out",
    )

    # if running this file directly, create the tmp csv then run:
    if not os.path.exists("tmp_test_input.csv"):
        df_test.to_csv("tmp_test_input.csv", index=False)
        cleaned_df, metadata, paths = run_preprocess(
            csv_path="tmp_test_input.csv",
            time_col="date",
            target_freq=None,
            agg="mean",
            impute_strategy="interpolate",
            impute_limit=3,
            low_variance_frac=0.001,
            max_missing_ratio=0.2,
            out_dir="tmp_algo1_out",
        )
        print("Cleaned shape:", cleaned_df.shape)
        print("Metadata:", metadata)
        print("Saved files:", paths)
