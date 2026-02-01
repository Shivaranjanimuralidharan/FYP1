# app_preprocess.py
# FastAPI service to accept CSV uploads and run Algorithm 1 (data preprocessor).
# Run with: uvicorn app_preprocess:app --reload --port 8000

import os
import shutil
import traceback
from typing import Optional

from typing import List, Dict, Any, Optional, Tuple


from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

#Profiler
from fastapi import BackgroundTasks
from time_series_profiler import profile_from_run_id    # function we added earlier
import json

#Segmenter
import time
from segmenter import segment_from_run_id 

#insight scorer
from fastapi import BackgroundTasks
from insight_scorer import run_scoring_for_run, log_user_feedback

#Charts


from sequence_chart_renderer import (
    generate_annotated_charts,
    render_single_chart,          # add this
)

from insight_text_generator import (
    generate_text_descriptions,
    generate_single_description,  # add this
)

from eva_sequence_generator import generate_eva_sequence  # add this



from data_preprocessor_algo1 import run_preprocess 

# For /api/data endpoint
import pandas as pd
import numpy as np

#Mod2
from rl.inference import PPOInferenceEngine
from rl.eva_env import ACTION_TYPES


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(APP_ROOT, "uploads")
OUT_BASE = os.path.join(APP_ROOT, "preproc_runs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_BASE, exist_ok=True)

app = FastAPI(title="Visail Preprocessing API")

# Allow local frontend (vite default 5173) and other dev hosts
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/preprocess", response_model=dict)
async def preprocess_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    time_col: Optional[str] = Form(None),
    target_freq: Optional[str] = Form(None),
    agg: str = Form("mean"),
    impute_strategy: str = Form("interpolate"),
    impute_limit: Optional[int] = Form(None),
    low_variance_frac: float = Form(0.01),
    max_missing_ratio: float = Form(0.2),
):
   
    # validate file
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    # create unique run dir
    run_id = str(abs(hash(file.filename + str(os.urandom(8)))))[:12]
    run_out = os.path.join(OUT_BASE, run_id)
    os.makedirs(run_out, exist_ok=True)

    # save uploaded file
    uploaded_path = os.path.join(UPLOAD_DIR, f"{run_id}__{file.filename}")
    try:
        with open(uploaded_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Calling preprocessing function 
    try:
        _ = run_preprocess(
            csv_path=uploaded_path,
            time_col=time_col,
            target_freq=target_freq,
            agg=agg,
            impute_strategy=impute_strategy,
            impute_limit=impute_limit,
            low_variance_frac=low_variance_frac,
            max_missing_ratio=max_missing_ratio,
            out_dir=run_out,
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {e}\n\n{tb}")

    # Build artifact paths 
    cleaned_fp = os.path.join(run_out, "cleaned_df.parquet")
    metadata_fp = os.path.join(run_out, "metadata.json")
    transform_fp = os.path.join(run_out, "transform_log.json")
    preview_first_fp = os.path.join(run_out, "preview_first.csv")
    preview_last_fp = os.path.join(run_out, "preview_last.csv")

    # Enqueue profiler as background task (non-blocking)
    # This will create preproc_runs/<run_id>/profiler.json when done
    background_tasks.add_task(run_profiler_task, run_id, OUT_BASE)
        # Enqueue segmenter task as well: it will wait for profiler.json (or timeout) then run segmentation
    background_tasks.add_task(run_segmenter_task, run_id, OUT_BASE, 300, 2.0, None)
        # Enqueue scoring task (will wait for segments.json)
    background_tasks.add_task(run_scoring_task, run_id, OUT_BASE, 300, 2.0)
    background_tasks.add_task(run_sequence_renderer_task, run_id, OUT_BASE)




    # Check which artifacts exist
    artifacts = {
        "cleaned_path": f"/api/download/{run_id}/cleaned_df.parquet" if os.path.exists(cleaned_fp) else None,
        "metadata_path": f"/api/download/{run_id}/metadata.json" if os.path.exists(metadata_fp) else None,
        "transform_log_path": f"/api/download/{run_id}/transform_log.json" if os.path.exists(transform_fp) else None,
        "preview_first": f"/api/download/{run_id}/preview_first.csv" if os.path.exists(preview_first_fp) else None,
        "preview_last": f"/api/download/{run_id}/preview_last.csv" if os.path.exists(preview_last_fp) else None,
    }

    missing = [name for name, path in artifacts.items() if path is None]
    resp = {
        "run_id": run_id,
        **artifacts,
    }
    if missing:
        resp["warning"] = f"Preprocessing completed but these expected artifacts are missing: {missing}"

    # Try to include kept/dropped columns from metadata.json if it exists
    if os.path.exists(metadata_fp):
        try:
            with open(metadata_fp, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            kept_cols = meta.get("kept_columns")
            dropped_cols = meta.get("dropped_columns")
            if kept_cols is not None:
                resp["kept_columns"] = kept_cols
            if dropped_cols is not None:
                resp["dropped_columns"] = dropped_cols
        except Exception:
            # ignore metadata parsing errors (metadata is optional)
            pass

    # Always return JSON (prevents empty response body)
    return JSONResponse(content=resp)



@app.get("/api/download/{run_id}/{filename}")
def download_artifact(run_id: str, filename: str):
    run_out = os.path.join(OUT_BASE, run_id)
    file_path = os.path.join(run_out, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)


@app.get("/api/runs")
def list_runs():
    """List recent run ids for convenience."""
    runs = []
    for name in os.listdir(OUT_BASE):
        p = os.path.join(OUT_BASE, name)
        if os.path.isdir(p):
            runs.append(name)
    return {"runs": runs}

def run_profiler_task(run_id: str, base_dir: str):
    """
    Worker function to run the profiler for run_id and save profiler.json under the run folder.
    Meant to be used with BackgroundTasks (non-blocking for the request).
    """
    try:
        profiler = profile_from_run_id(run_id, base_dir)   # loads cleaned_df.parquet and profiles
        out_folder = os.path.join(base_dir, run_id)
        os.makedirs(out_folder, exist_ok=True)
        profiler_path = os.path.join(out_folder, "profiler.json")
        with open(profiler_path, "w", encoding="utf-8") as fh:
            json.dump(profiler, fh, indent=2, default=lambda o: o if isinstance(o, (str,int,float,bool)) else str(o))
        # optionally: log success
        print(f"[profiler] saved profiler for run {run_id} -> {profiler_path}")
    except Exception as e:
        # log the error; do not raise (background tasks lost otherwise)
        print(f"[profiler] failed for run {run_id}: {e}")

def run_segmenter_task(run_id: str, base_dir: str, wait_timeout: int = 300, poll_interval: float = 2.0, config: Optional[Dict[str, Any]] = None):
    """
    Background worker that waits for profiler.json (produced by profiler) and then
    runs the segmenter. It writes preproc_runs/<run_id>/segments.json and thumbnails.
    - wait_timeout: how many seconds to wait for profiler.json before running anyway
    - poll_interval: seconds between checks
    """
    run_out = os.path.join(base_dir, run_id)
    profiler_path = os.path.join(run_out, "profiler.json")
    cleaned_path = os.path.join(run_out, "cleaned_df.parquet")
    start = time.time()

    # Wait until cleaned data exists (should already) and profiler (preferable)
    while True:
        if os.path.exists(cleaned_path) and (os.path.exists(profiler_path) or (time.time() - start) > wait_timeout):
            break
        time.sleep(poll_interval)

    # Now run segmentation (load cleaned_df inside function)
    try:
        segs, segs_path = segment_from_run_id(run_id, base_dir, config=config)
        print(f"[segmenter] saved segments for run {run_id} -> {segs_path}")
    except Exception as e:
        # log error, don't raise (background)
        print(f"[segmenter] failed for run {run_id}: {e}")

# ----------------------------
# Endpoint: return cleaned data as JSON time series
# ----------------------------
def read_cleaned_df_to_timeseries(run_id: str, max_points: int = 5000):
    """
    Loads cleaned_df.parquet for run_id and returns JSON-friendly structure:
    {
      "timestamps": [t1_iso, t2_iso, ...],
      "columns": ["colA", "colB", ...],
      "data": { "colA": [v1, v2, ...], "colB": [...] }
    }
    Downsamples to max_points if necessary (simple index sampling).
    """
    run_out = os.path.join(OUT_BASE, run_id)
    cleaned_path = os.path.join(run_out, "cleaned_df.parquet")
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError("cleaned_df.parquet not found for run_id")

    # Read parquet (requires pyarrow or fastparquet installed)
    df = pd.read_parquet(cleaned_path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            # if conversion fails keep original index but convert timestamps to strings later
            pass
    df = df.sort_index()

    # If too many points, pick evenly spaced indices
    n = len(df)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        df = df.iloc[idx]

    # Convert timestamps to ISO strings
    try:
        timestamps = [ts.isoformat() for ts in df.index.to_pydatetime()]
    except Exception:
        # fallback: str()
        timestamps = [str(ts) for ts in df.index.tolist()]

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    data = {}
    for c in numeric_cols:
        # convert numpy types to native python types; fill NaN with None for JSON
        col_vals = df[c].where(df[c].notna(), None).tolist()
        data[c] = col_vals

    return {"timestamps": timestamps, "columns": numeric_cols, "data": data}

def run_scoring_task(run_id: str, base_dir: str, wait_timeout: int = 300, poll_interval: float = 2.0):
    """
    Background worker: waits for segments.json (created by segmenter) and runs scoring.
    Writes ranked_segments.json and topk.json into run folder.
    """
    try:
        out = run_scoring_for_run(run_id, base_dir, cfg=None, wait_timeout=wait_timeout, poll_interval=poll_interval)
        print(f"[scoring] saved ranked results for run {run_id} -> {out}")
    except Exception as e:
        print(f"[scoring] failed for run {run_id}: {e}")


def run_sequence_renderer_task(run_id: str, base_dir: str):
    """
    Generates annotated charts + descriptions when user clicks Start.
    """
    run_folder = os.path.join(base_dir, run_id)

    cleaned = os.path.join(run_folder, "cleaned_df.parquet")
    topk = os.path.join(run_folder, "topk.json")

    if not os.path.exists(cleaned) or not os.path.exists(topk):
        print("[renderer] Missing inputs.")
        return

    import pandas as pd, json
    df = pd.read_parquet(cleaned)

    with open(topk, "r") as f:
        eva_seq = json.load(f)

    charts_dir = os.path.join(run_folder, "annotated_charts")
    os.makedirs(charts_dir, exist_ok=True)
    for seg in eva_seq:
     if "insight_type" not in seg:
        seg["insight_type"] = "trend"

    generate_annotated_charts(eva_seq, df, charts_dir)

    # inject df so text generator can compute facts
    for seg in eva_seq:
        seg["_df"] = df  

    descriptions = generate_text_descriptions(eva_seq)

    with open(os.path.join(run_folder, "descriptions.json"), "w") as f:
        json.dump(descriptions, f, indent=2)

    print(f"[renderer] Completed for run {run_id}")


@app.post("/api/render_sequence/{run_id}")
def render_sequence(run_id: str, background_tasks: BackgroundTasks):
    """
    Trigger Module 3 (Annotated Charts + Insight Descriptions).
    Called ONLY when user clicks START in UI.
    """
    background_tasks.add_task(run_sequence_renderer_task, run_id, OUT_BASE)
    return {"status": "started", "run_id": run_id}


@app.get("/api/data/{run_id}")
def get_timeseries_data(run_id: str, max_points: int = 5000):
    """
    Returns cleaned time series data for plotting on the timeline view.
    Example: GET /api/data/<run_id>?max_points=2000
    """
    try:
        payload = read_cleaned_df_to_timeseries(run_id, max_points=int(max_points))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Run not found or cleaned data not available yet")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Failed to load cleaned data: {e}\n\n{tb}")
    return JSONResponse(content=payload)

@app.get("/api/profile/{run_id}")
def get_profiler_json(run_id: str):
    path = os.path.join(OUT_BASE, run_id, "profiler.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Profiler results not ready")
    return FileResponse(path, filename="profiler.json")

@app.get("/api/{run_id}/score_segments")
def score_segments(run_id: str):
    """
    Return top‑K scored segments produced by Module 1.4.
    """
    run_folder = os.path.join(OUT_BASE, run_id)
    scores_path = os.path.join(run_folder, "scores.json")
    topk_path = os.path.join(run_folder, "topk.json")

    if not os.path.exists(scores_path):
        raise HTTPException(404, "Scores not computed for this run")

    import json
    with open(scores_path, "r") as f:
        scores = json.load(f)

    # If topk not saved, return all sorted scores
    if os.path.exists(topk_path):
        with open(topk_path, "r") as f:
            topk = json.load(f)
    else:
        topk = scores[:50]   # default top‑50

    return {"top_k": topk}

@app.post("/api/{run_id}/sequence/generate_one")
async def generate_one_insight(run_id: str, payload: dict):
    import uuid
    import pandas as pd

    run_folder = os.path.join(OUT_BASE, run_id)
    df = pd.read_parquet(os.path.join(run_folder, "cleaned_df.parquet"))

    measure = payload["measure"]
    ins_type = payload["type"]
    breakdown = payload["breakdown"]
    subspace = payload.get("subspace")

    # Convert subspace indices → timestamps
    if subspace:
        timestamps = df.index.tolist()
        start = timestamps[subspace["start"]]
        end = timestamps[subspace["end"]]
    else:
        start, end = df.index.min(), df.index.max()

    # Build EVA-style segment (THIS IS KEY)
    segment = {
        "measure": measure,
        "insight_type": ins_type,
        "breakdown": breakdown,
        "start": start,
        "end": end,
    }

    charts_folder = os.path.join(run_folder, "annotated_charts")
    os.makedirs(charts_folder, exist_ok=True)

    # ✅ UNIQUE filename per request
    uid = uuid.uuid4().hex[:8]
    filename = f"interactive_{uid}_{measure}_{ins_type}.png"

    # ✅ Render EXACTLY ONE annotated EVA chart
    generate_annotated_charts([segment], df, charts_folder, filename_override=filename)

    # Text description matches the same segment
    description = generate_single_description(segment, df)

    return {
        **segment,
        "thumbnail_path": f"/api/{run_id}/sequence/charts/{filename}",
        "description": description
    }



    # Module 3.1 — annotated chart
    charts_folder = os.path.join(run_folder, "annotated_charts")
    os.makedirs(charts_folder, exist_ok=True)

    import uuid

    filename = f"single_{uuid.uuid4().hex}.png"

    chart_path = render_single_chart(
        insight, df, charts_folder, filename=filename
    )


    # Module 3.2 — description
    description = generate_single_description(insight, df)

    card = {
    **insight,
    "thumbnail_path": f"/api/{run_id}/sequence/charts/{filename}",
    "description": description
   }


    return card

@app.post("/api/{run_id}/sequence/generate_subsequent")
async def generate_subsequent(run_id: str, payload: dict):
    """
    Regenerates FULL sequence starting from modified insight definition.
    """
    run_folder = os.path.join(OUT_BASE, run_id)
    import pandas as pd, json

    df = pd.read_parquet(os.path.join(run_folder, "cleaned_df.parquet"))

    measure = payload["measure"]
    ins_type = payload["type"]
    breakdown = payload["breakdown"]
    subspace = payload.get("subspace")

    modified_template = {
        "measure": measure,
        "insight_type": ins_type,
        "breakdown": breakdown,
        "subspace": subspace
    }

    # Module 3 — generate full EVA sequence
    new_sequence = generate_eva_sequence(df, modified_template)

    charts_folder = os.path.join(run_folder, "annotated_charts")
    os.makedirs(charts_folder, exist_ok=True)

    # render charts
    for idx, ins in enumerate(new_sequence):
        render_single_chart(ins, df, charts_folder, filename=f"{idx}.png")

    # descriptions
    descriptions = [generate_single_description(ins, df) for ins in new_sequence]
    desc_path = os.path.join(run_folder, "descriptions.json")
    with open(desc_path, "w") as f:
        json.dump(descriptions, f, indent=2)

    return {"sequence": new_sequence}

@app.get("/api/{run_id}/sequence/charts/{filename}")
def serve_chart(run_id: str, filename: str):
    fp = os.path.join(OUT_BASE, run_id, "annotated_charts", filename)
    if not os.path.exists(fp):
        raise HTTPException(404, "Chart not found")
    return FileResponse(fp)

@app.get("/api/{run_id}/sequence/descriptions")
def serve_descriptions(run_id: str):
    fp = os.path.join(OUT_BASE, run_id, "descriptions.json")
    if not os.path.exists(fp):
        raise HTTPException(404, "Descriptions not found")
    import json
    with open(fp, "r") as f:
        desc = json.load(f)
    return {"descriptions": desc}

@app.get("/api/{run_id}/sequence/combined")
def get_combined_sequence(run_id: str):
    """
    Combines:
    - topk.json (segment metadata)
    - annotated_charts/*.png
    - descriptions.json
    into a single clean list for UI to render sequence cards.
    """

    run_folder = os.path.join(OUT_BASE, run_id)
    topk_path = os.path.join(run_folder, "topk.json")
    desc_path = os.path.join(run_folder, "descriptions.json")
    charts_folder = os.path.join(run_folder, "annotated_charts")

    if not os.path.exists(topk_path):
        raise HTTPException(404, "topk.json not found")

    with open(topk_path, "r") as f:
        segments = json.load(f)

    # descriptions may not exist yet
    descriptions = []
    if os.path.exists(desc_path):
        with open(desc_path, "r") as f:
            descriptions = json.load(f)

    combined = []

    for idx, seg in enumerate(segments):

        # Ensure insight_type exists (your inference logic already sets it)
        insight_type = seg.get("insight_type", "unknown")

        # Find chart
        expected_file = f"chart_{idx}_{insight_type}.png"
        chart_path = os.path.join(charts_folder, expected_file)

        # If missing, fallback to *any* file for that index
        if not os.path.exists(chart_path):
            # find something like chart_0_*.png
            candidates = [
                f for f in os.listdir(charts_folder)
                if f.startswith(f"chart_{idx}_")
            ]
            if candidates:
                chart_path = os.path.join(charts_folder, candidates[0])
                expected_file = candidates[0]
            else:
                expected_file = None

        # Get description
        description = descriptions[idx] if idx < len(descriptions) else "Description not available."

        combined.append({
            "id": idx,
            "measure": seg["measure"],
            "insight_type": insight_type,
            "start": seg["start"],
            "end": seg["end"],
            "thumbnail_path": f"/api/{run_id}/sequence/charts/{expected_file}" if expected_file else None,
            "description": description
        })

    return {"sequence": combined}


@app.post("/get_alternatives")
def get_alternatives(payload: dict):

    state_vector = payload["state_vector"]
    clicked_insight = payload["clicked_insight"]
    run_id = payload["run_id"]

    run_folder = os.path.join(OUT_BASE, run_id)
    df = pd.read_parquet(os.path.join(run_folder, "cleaned_df.parquet"))

    metadata_path = os.path.join(OUT_BASE, run_id, "metadata.json")

    # ✅ create RL engine
    ppo_engine = PPOInferenceEngine(
        state_dim=128,
        action_dim=len(ACTION_TYPES),
        metadata_path=metadata_path
    )

    charts_folder = os.path.join(run_folder, "annotated_charts")
    os.makedirs(charts_folder, exist_ok=True)

    # RL propose alternatives
    rl_alternatives = ppo_engine.get_alternate_insights(
        state_vector,
        clicked_insight
    )

    rendered_alternatives = []

    for idx, alt in enumerate(rl_alternatives):

        insight = alt["insight"]

        segment = {
            "measure": insight["measure"],
            "insight_type": insight["insight_type"],
            "breakdown": insight["breakdown"],
            "start": insight["start"],
            "end": insight["end"],
        }

        # ----------------------------------
        # APPLY SHIFT ACTION (if any)
        # ----------------------------------
        shift = insight.get("shift")

        if shift and segment["start"] and segment["end"]:
            start = pd.to_datetime(segment["start"])
            end = pd.to_datetime(segment["end"])
            window = end - start

            if shift == "forward":
                new_start = start + window
                new_end = end + window
            elif shift == "backward":
                new_start = start - window
                new_end = end - window
            else:
                new_start, new_end = start, end

            # clamp to dataset bounds
            min_t = df.index.min()
            max_t = df.index.max()

            if new_start >= min_t and new_end <= max_t:
                segment["start"] = new_start
                segment["end"] = new_end


        filename = f"alt_{idx}_{segment['insight_type']}.png"

        generate_annotated_charts(
            [segment],
            df,
            charts_folder,
            filename_override=filename
        )

        description = generate_single_description(segment, df)

        rendered_alternatives.append({
            "insight": {
                "measure": segment["measure"],
                "insight_type": segment["insight_type"],
                "breakdown": segment["breakdown"],
                "start": segment["start"],
                "end": segment["end"],
                "thumbnail_path": f"/api/{run_id}/sequence/charts/alt_{idx}_{segment['insight_type']}.png",
                "description": description,
            },
            "meta": alt["meta"],
        })

    import json 
    print(
        "\n[RL alternatives – formatted]\n"
        + json.dumps(rendered_alternatives, indent=2, default=str)
    )

    rl_out_path = os.path.join(run_folder, "rl_alternatives.json")

    with open(rl_out_path, "w", encoding="utf-8") as f:
        json.dump(rendered_alternatives, f, indent=2, default=str)

    print(f"[RL] Saved alternatives → {rl_out_path}")

    return {"alternatives": rendered_alternatives}

from fastapi import HTTPException
from module4.module4_main import run_module4
import os


@app.post("/api/{run_id}/export_narrative")
def export_narrative(run_id: str):

    try:
        print(f"[Export] Starting export for run_id={run_id}")

        BASE_DIR = os.path.join(os.getcwd(), "preproc_runs")
        run_dir = os.path.join(BASE_DIR, run_id)

        if not os.path.exists(run_dir):
            raise Exception(f"Run folder not found: {run_id}")

        result = run_module4(run_id, BASE_DIR)

        print("[Export] Module4 completed")

        narrative_path = os.path.join(run_dir, "narrative.txt")

        if not os.path.exists(narrative_path):
            raise Exception("Narrative file was not generated")

        return {
        "status": "ok",
        "narrative_path": f"/api/{run_id}/narrative.txt"
    }

    except Exception as e:
        print("❌ EXPORT ERROR:", str(e))
        import traceback
        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import FileResponse

@app.get("/api/{run_id}/narrative.txt")
def download_narrative(run_id: str):
    BASE_DIR = os.path.join(os.getcwd(), "preproc_runs")
    narrative_path = os.path.join(BASE_DIR, run_id, "narrative.txt")

    if not os.path.exists(narrative_path):
        raise HTTPException(status_code=404, detail="Narrative not found")

    return FileResponse(
        narrative_path,
        media_type="text/plain",
        filename="narrative.txt"
    )

def load_dataset_metadata(run_id):
    path = os.path.join(OUT_BASE, run_id, "metadata.json")
    with open(path) as f:
        meta = json.load(f)
    return {
        "time_col": meta["time_col"],
        "measures": meta["kept_columns"]
    }



