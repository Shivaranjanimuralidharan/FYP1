# module4/narrative_aggregator.py

import os
import json
from datetime import datetime


def _safe_round(v, d=3):
    try:
        return round(float(v), d)
    except Exception:
        return v


def aggregate_narrative_inputs(run_id, base_dir):
    run_folder = os.path.join(base_dir, run_id)

    # ---------- LOAD INPUTS ----------
    with open(os.path.join(run_folder, "topk.json")) as f:
        insights = json.load(f)

    desc_path = os.path.join(run_folder, "descriptions.json")
    descriptions = json.load(open(desc_path)) if os.path.exists(desc_path) else []

    rl_path = os.path.join(run_folder, "rl_alternatives.json")
    rl_logs = json.load(open(rl_path)) if os.path.exists(rl_path) else []

    meta_path = os.path.join(run_folder, "metadata.json")
    dataset_meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # ---------- SORT INSIGHTS (chronological) ----------
    insights = sorted(insights, key=lambda x: x.get("start", ""))

    SummaryBlocks = []
    ChartDescriptors = []
    UserLogic = []

    # ---------- BUILD BLOCKS ----------
    for idx, ins in enumerate(insights):

        facts = ins.get("facts", {})
        insight_type = (
            ins.get("insight_type")
            or ins.get("type")
            or "trend"
        )
        summary = {
            "id": idx,
            "claim": descriptions[idx] if idx < len(descriptions) else "",
            "measure": ins["measure"],
            "insight_type": insight_type,
            "time_window": {
                "start": ins["start"],
                "end": ins["end"]
            },

            # 🔹 numeric evidence (LLM grounding)
            "key_values": {
                "slope": _safe_round(facts.get("slope")),
                "mean": _safe_round(facts.get("mean")),
                "variance": _safe_round(facts.get("variance")),
                "extreme_value": _safe_round(facts.get("extreme_value"))
            },

            # 🔹 quality flags
            "quality": {
                "has_facts": bool(facts),
                "description_present": idx < len(descriptions)
            }
        }

        SummaryBlocks.append(summary)

        ChartDescriptors.append({
            "id": idx,
            "measure": ins["measure"],
            "type": insight_type,
            "start": ins["start"],
            "end": ins["end"]
        })

    # ---------- USER LOGIC ----------
    for rl in rl_logs:
        UserLogic.append({
            "insight_id": rl.get("id"),
            "action": rl.get("meta", {}).get("action"),
            "probability": _safe_round(
                rl.get("meta", {}).get("probability")
            ),
            "timestamp": datetime.utcnow().isoformat()
        })

    # ---------- CONTEXT ----------
    actions = [u["action"] for u in UserLogic if u["action"]]

    Context = {
        "ordering": "chronological",
        "cross_links": "temporal",
        "preferences": {
            "frequent_actions": list(set(actions))
        },
        "dataset": {
            "kept_columns": dataset_meta.get("kept_columns"),
        }
    }

    # ---------- VALIDATION ----------
    if not SummaryBlocks:
        raise ValueError("No insights found for narrative aggregation")

    # ---------- FINAL BUNDLE ----------
    narrative_bundle = {
        "SummaryBlocks": SummaryBlocks,
        "ChartDescriptors": ChartDescriptors,
        "UserLogic": UserLogic,
        "Context": Context,
        "generated_at": datetime.utcnow().isoformat()
    }

    # ---------- SAVE ----------
    out_path = os.path.join(run_folder, "narrative_bundle.json")

    with open(out_path, "w") as f:
        json.dump(narrative_bundle, f, indent=2)

    print(f"[Module4] Narrative bundle saved → {out_path}")

    return narrative_bundle
