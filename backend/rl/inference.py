import torch
import numpy as np
import random
import os

from rl.ppo_policy import PPOAgent
from rl.eva_env import ACTION_TYPES

# -----------------------------
# CONFIG
# -----------------------------
PPO_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "ppo_weights.pt")
TOP_K = 5


class PPOInferenceEngine:
    def __init__(self, state_dim, action_dim, device="cpu"):
        self.device = device

        # ---- action metadata (must exist) ----
        self.action_space = ACTION_TYPES
        self.insight_types = [
            "trend", "distribution", "correlation",
            "outlier", "seasonality", "autocorrelation"
        ]
        self.measures = [
            "cases", "deaths", "vaccinations"
        ]

        # ---- load PPO agent ----
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
            "insight_type": insight.get("insight_type", "trend"),
            "measure": insight.get("measure", "cases"),
            "breakdown": insight.get("breakdown", "day"),
            "description": insight.get("description", ""),
            "thumbnail_path": insight.get("thumbnail_path", None),
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
            "insight": {
                "insight_type": new_insight.get("insight_type", "trend"),
                "measure": new_insight.get("measure", ""),
                "breakdown": new_insight.get("breakdown", "day"),
                "description": new_insight.get(
                    "description",
                    "Alternative generated insight"
                ),
                "thumbnail_path": new_insight.get("thumbnail_path", None),
            },
            "meta": {
                "action": action,
                "probability": float(probs[idx]),
            }
            })


        return alternatives

    # -------------------------
    # APPLY ACTION (FIXED)
    # -------------------------
    def _apply_action(self, insight, action):
        new_insight = insight.copy()

        if action == "CHANGE_TYPE":
            new_insight["insight_type"] = random.choice(self.insight_types)

        elif action == "CHANGE_MEASURE":
            new_insight["measure"] = random.choice(self.measures)

        elif action == "AGGREGATE":
            new_insight["breakdown"] = "month"

        elif action == "REMOVE_AGGREGATE":
            new_insight["breakdown"] = "day"

        # SHIFT actions are safe no-ops for UI
        elif action in ["SHIFT_FORWARD", "SHIFT_BACKWARD", "SHIFT_PERIODICAL"]:
            pass

        new_insight["description"] = (
            f"Alternative {new_insight['insight_type']} insight "
            f"on {new_insight['measure']} ({new_insight['breakdown']})"
        )

        return new_insight
