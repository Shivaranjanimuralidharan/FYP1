import torch
import numpy as np
import random
import os
import json

from rl.ppo_policy import PPOAgent
from rl.eva_env import ACTION_TYPES

# -----------------------------
# CONFIG
# -----------------------------
PPO_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ppo_weights.pt")
TOP_K = 5


class PPOInferenceEngine:
    def __init__(self, state_dim, action_dim, metadata_path, device="cpu"):
        self.device = device

        # -----------------------------
        # LOAD METADATA (DATASET-AGNOSTIC)
        # -----------------------------
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.measures = metadata["kept_columns"]
        self.time_col = metadata["time_col"]

        if not self.measures:
            raise ValueError("No measures found in metadata.json")

        # -----------------------------
        # ACTION SPACE
        # -----------------------------
        self.action_space = ACTION_TYPES

        # insight types are semantic — safe to keep
        self.insight_types = [
            "trend",
            "distribution",
            "correlation",
            "outlier",
            "seasonality",
            "autocorrelation"
        ]

        # -----------------------------
        # LOAD PPO MODEL
        # -----------------------------
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

        checkpoint = torch.load(PPO_WEIGHTS_PATH, map_location=device)
        self.agent.actor.load_state_dict(checkpoint["actor"])
        self.agent.critic.load_state_dict(checkpoint["critic"])

        self.agent.actor.eval()
        self.agent.critic.eval()

    # -------------------------
    # Normalize UI insight
    # -------------------------
    def _normalize_insight(self, insight):
        return {
            "insight_type": insight.get(
                "insight_type",
                insight.get("type", "trend")
            ),
            "measure": insight["measure"],
            "breakdown": insight.get("breakdown", "day"),
            "start": insight.get("start"),
            "end": insight.get("end"),
        }

    # -------------------------
    # Main API
    # -------------------------
    def get_alternate_insights(self, state_vector, clicked_insight):
        clicked_insight = self._normalize_insight(clicked_insight)

        state_tensor = torch.tensor(
            state_vector, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.agent.actor(state_tensor).cpu().numpy().flatten()

        action_indices = np.argsort(probs)[-TOP_K:][::-1]

        alternatives = []

        for idx in action_indices:
            action = self.action_space[idx]

            new_insight = self._apply_action(clicked_insight, action)

            alternatives.append({
                "insight": new_insight,
                "meta": {
                    "action": action,
                    "probability": float(probs[idx]),
                }
            })

        return alternatives

    # -------------------------
    # APPLY ACTION — NO HARDCODING
    # -------------------------
    def _apply_action(self, insight, action):
        new_insight = insight.copy()

        # CHANGE TYPE
        if action == "CHANGE_TYPE":
            types = [t for t in self.insight_types if t != insight["insight_type"]]
            new_insight["insight_type"] = (
                random.choice(types) if types else insight["insight_type"]
            )

        # CHANGE MEASURE — from metadata
        elif action == "CHANGE_MEASURE":
            measures = [m for m in self.measures if m != insight["measure"]]
            if measures:
                new_insight["measure"] = random.choice(measures)

        # AGGREGATION
        elif action == "AGGREGATE":
            new_insight["breakdown"] = "month"

        elif action == "REMOVE_AGGREGATE":
            new_insight["breakdown"] = "day"

        # SHIFT — keep measure same, backend will adjust window
        elif action == "SHIFT_FORWARD":
            new_insight["shift"] = "forward"

        elif action == "SHIFT_BACKWARD":
            new_insight["shift"] = "backward"

        elif action == "SHIFT_PERIODICAL":
            new_insight["shift"] = "periodic"

        VALID_INSIGHTS = {
            "trend",
            "seasonality",
            "outlier",
            "distribution",
            "correlation",
            "autocorrelation",
        }

        if new_insight.get("insight_type") not in VALID_INSIGHTS:
            new_insight["insight_type"] = "trend"

        return new_insight
