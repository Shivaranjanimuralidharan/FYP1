# module4/module4_main.py

import os
from module4.narrative_aggregator import aggregate_narrative_inputs
from module4.prompt_builder import build_prompt
from module4.llm_narrative_generator import generate_narrative
from llm_clients.ollama_client import OllamaClient


def run_module4(run_id, base_dir):
    print("[Module4] Starting pipeline")

    # -------------------------
    # Step 1 — Aggregate inputs
    # -------------------------
    bundle = aggregate_narrative_inputs(run_id, base_dir)
    print("[Module4] Aggregation complete")

    # -------------------------
    # Step 2 — Build prompt
    # -------------------------
    prompt_obj = build_prompt(
        bundle,
        config={
            "tone": "formal",
            "length": "medium"
        }
    )

    import os

    # ---------- SAVE PROMPT ----------
    run_folder = os.path.join(base_dir, run_id)
    prompt_text = prompt_obj["prompt"]

    prompt_path = os.path.join(run_folder, "llm_prompt.txt")

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)

    print(f"[Module4] Prompt saved → {prompt_path}")

    print("[Module4] Prompt built")

    # -------------------------
    # Step 3 — LLM Generate
    # -------------------------
    import time
    start = time.time()
    llm = OllamaClient()

    narrative = generate_narrative(
        prompt_obj,
        llm,
        config={
            "temperature": 0.2,
            "max_tokens": 900,
            "model": "llama3"
        }
    )
    end = time.time()
    print("[Module4] Narrative generated")
    print(f"\n[TIME] Took {round(end - start, 2)} seconds")

    # -------------------------
    # Step 4 — Save output
    # -------------------------
    run_folder = os.path.join(base_dir, run_id)
    out_path = os.path.join(run_folder, "narrative.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(narrative["text"])

    print(f"[Module4] Saved → {out_path}")

    return {
        "text": narrative["text"],
        "path": out_path
    }

