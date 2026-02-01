# module4/prompt_builder.py
import json

def build_prompt(narrative_bundle, config):
    """
    Build a synthesis-oriented LLM prompt that produces
    a concise, insightful, and non-technical analytical narrative.
    """

    P = []

    # --------------------------------------------------
    # SYSTEM ROLE
    # --------------------------------------------------
    P.append(
        "You are an analytical data storyteller. "
        "Your task is to synthesize analytical findings into a clear, "
        "human-readable narrative for non-expert users. "
        "Focus on overall patterns, directional changes, and meaningful shifts. "
        "Avoid technical language, formulas, or statistical jargon. "
        "Use only the information provided, combined with general real-world context."
    )

    # --------------------------------------------------
    # DATASET CONTEXT
    # --------------------------------------------------
    context = narrative_bundle.get("Context", {})
    metadata = context.get("dataset", {})
    if metadata:
        P.append("\nDATASET CONTEXT:\n")

        dataset_desc = metadata.get(
            "dataset_description",
            "A global COVID-19 time-series dataset compiled from publicly reported "
            "statistics, tracking how the pandemic evolved over time across countries. "
            "The dataset captures key indicators such as cases, deaths, testing, and "
            "vaccination activity, and has been cleaned and resampled to support "
            "trend-focused analysis."
        )
        P.append(f"- Dataset overview: {dataset_desc}")

        kept_cols = metadata.get("kept_columns")
        if kept_cols:
            P.append(
                "- Key measures available for analysis include: "
                + ", ".join(kept_cols)
            )

    # --------------------------------------------------
    # ANALYTICAL CONTEXT (RL Alternatives)
    # --------------------------------------------------
    rl_alts = narrative_bundle.get("RLAlternatives", [])
    if rl_alts:
        P.append(
            "\nANALYTICAL CONTEXT:\n"
            "Multiple analytical perspectives were explored to understand "
            "how patterns change over time. These perspectives inform the interpretation "
            "but should not be repeated explicitly."
        )

    # --------------------------------------------------
    # TASK DEFINITION
    # --------------------------------------------------
    P.append(
        "\nTASK:\n"
        "Write a concise analytical narrative describing how the primary measure changes "
        "over time. Group similar periods together, describe whether trends are increasing, "
        "decreasing, or stabilizing, and explain notable shifts in simple terms. "
        "When relevant, relate changes to widely known real-world developments "
        "(such as policy responses or vaccination efforts) without asserting causality."
    )

    # --------------------------------------------------
    # EVIDENCE — INSIGHT SUMMARIES
    # --------------------------------------------------
    P.append("\nEVIDENCE — ANALYTICAL FINDINGS:\n")

    for block in narrative_bundle["SummaryBlocks"]:
        claim = block.get("claim", "").strip()
        if claim:
            P.append(f"- {claim}")

    # --------------------------------------------------
    # USER EXPLORATION CONTEXT
    # --------------------------------------------------
    user_logic = narrative_bundle.get("UserLogic", [])
    if user_logic:
        actions = [u["action"] for u in user_logic if "action" in u]
        if actions:
            P.append(
                "\nUSER CONTEXT:\n"
                "The user explored the data interactively, indicating an interest "
                "in understanding how patterns evolve and compare over time."
            )

    # --------------------------------------------------
    # NARRATIVE CONSTRAINTS (VERY IMPORTANT)
    # --------------------------------------------------
    P.append(
        "\nNARRATIVE RULES:\n"
        "- Do NOT mention numeric slopes, coefficients, or regression values\n"
        "- Describe changes qualitatively (e.g., sharp rise, gradual decline)\n"
        "- Do NOT list events chronologically one by one\n"
        "- Limit each trend explanation to at most two sentences\n"
        "- You may suggest plausible contributing factors, but avoid strong causal claims\n"
        "- Ensure the narrative ends with a clear concluding paragraph\n"
    )

    # --------------------------------------------------
    # OUTPUT FORMAT
    # --------------------------------------------------
    P.append(
        "\nOUTPUT FORMAT:\n"
        "- Short multi-paragraph narrative\n"
        "- Plain language suitable for non-expert readers\n"
        "- Smooth flow from early trends to later stabilization or change\n"
        "- Final paragraph summarizing the overall pattern and key takeaways\n"
    )

    # --------------------------------------------------
    # STYLE CONTROL
    # --------------------------------------------------
    tone = config.get("tone", "neutral")
    length = config.get("length", "concise")

    P.append(
        f"\nSTYLE SETTINGS:\n"
        f"- Tone: {tone}\n"
        f"- Length: {length}\n"
        "- Prioritize clarity, brevity, and interpretability\n"
    )

    # --------------------------------------------------
    # FINAL PROMPT
    # --------------------------------------------------
    prompt = "\n".join(P)

    return {
        "prompt": prompt,
        "meta": {
            "tone": tone,
            "length": length,
            "evidence_blocks": len(narrative_bundle["SummaryBlocks"]),
            "has_metadata": bool(metadata),
            "has_rl_context": bool(rl_alts),
            "prompt_type": "contextual_synthesized_narrative"
        }
    }
