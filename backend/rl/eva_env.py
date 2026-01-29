import numpy as np
import pandas as pd
import random

from datetime import timedelta

# Action space (Table 1 from paper)
ACTION_TYPES = [
    "CHANGE_TYPE",
    "SHIFT_FORWARD",
    "SHIFT_BACKWARD",
    "AGGREGATE",
    "REMOVE_AGGREGATE",
    "CHANGE_MEASURE",
]


class EVAEnvironment:
    def __init__(self, dataset_path, max_steps=5):
        self.dataset_path = dataset_path
        self.max_steps = max_steps

        # -----------------------------
        # LOAD DATASET (FIX)
        # -----------------------------
        self.dataset = pd.read_csv(dataset_path)

        # Ensure date column exists
        if "date" in self.dataset.columns:
            self.dataset["date"] = pd.to_datetime(self.dataset["date"])
        else:
            raise ValueError("Dataset must contain a 'date' column")
        # -----------------------------
        # DEFINE MEASURES (variables)
        # -----------------------------
        self.measures = [
            col for col in self.dataset.columns if col != "date"
        ]

        if len(self.measures) == 0:
            raise ValueError("No measure columns found in dataset")


        # -----------------------------
        # ACTION SPACE (Table 1)
        # -----------------------------
        self.action_types = [
            "CHANGE_TYPE",
            "SHIFT_FORWARD",
            "SHIFT_BACKWARD",
            "AGGREGATE",
            "REMOVE_AGGREGATE",
            "CHANGE_MEASURE",
        ]
        self.action_dim = len(self.action_types)

        # -----------------------------
        # STATE DIMENSION
        # -----------------------------
        self.state_dim = 128

        # -----------------------------
        # ENV STATE
        # -----------------------------
        self.current_step = 0
        self.current_sequence = []

        # Initialize episode
        self.reset()


    # -------------------------
    # Environment lifecycle
    # -------------------------
    def reset(self):
        """
        Start with one initial insight
        """
        self.steps = 0
        self.sequence = []

        initial_insight = self._random_initial_insight()
        self.sequence.append(initial_insight)

        return self._get_state()

    def step(self, action_idx):
        """
        Apply one action → generate new insight
        """
        action_type = ACTION_TYPES[action_idx]
        prev_insight = self.sequence[-1]

        new_insight = self._apply_action(prev_insight, action_type)
        self.sequence.append(new_insight)

        reward = self._compute_reward(new_insight)
        self.steps += 1

        done = self.steps >= self.max_steps

        return self._get_state(), reward, done, {}

    # -------------------------
    # State representation
    # -------------------------
    def _get_state(self):
        """
        Returns a placeholder numeric state.
        Actual feature encoding happens in PPO agent.
        """
        return np.zeros(self.state_dim, dtype=np.float32)

    # -------------------------
    # Insight construction
    # -------------------------
    def _random_initial_insight(self):
        start = random.choice(self.dataset["date"])
        end = start + timedelta(days=30)

        return {
            "type": "trend",
            "measure": random.choice(self.measures),
            "breakdown": "day",
            "subspace": (start, end),
        }

    def _apply_action(self, insight, action):
        """
        Apply metadata transformation (Table 1)
        """
        new_insight = insight.copy()

        if action == "CHANGE_TYPE":
            new_insight["type"] = random.choice(
                ["trend", "seasonality", "outlier", "distribution"]
            )

        elif action == "SHIFT_FORWARD":
            start, end = insight["subspace"]
            delta = end - start
            new_insight["subspace"] = (end, end + delta)

        elif action == "SHIFT_BACKWARD":
            start, end = insight["subspace"]
            delta = end - start
            new_insight["subspace"] = (start - delta, start)

        elif action == "AGGREGATE":
            new_insight["breakdown"] = "month"

        elif action == "REMOVE_AGGREGATE":
            new_insight["breakdown"] = "day"

        elif action == "CHANGE_MEASURE":
            new_insight["measure"] = random.choice(self.measures)

        return new_insight

    # -------------------------
    # Reward (Equation 7)
    # -------------------------
    def _compute_reward(self, new_insight):
        rf = self._familiarity(new_insight)
        rcr = self._curiosity_raw(new_insight)
        rcd = self._curiosity_derived(new_insight)

        # Eq. (7) from paper
        return 0.4 * rf + 0.4 * rcr + 0.2 * rcd

    def _familiarity(self, new_insight):
        """
        Metadata overlap with previous insights
        """
        if len(self.sequence) <= 1:
            return 0.5

        matches = 0
        prev = self.sequence[:-1]

        for ins in prev:
            if ins["type"] == new_insight["type"]:
                matches += 1
            if ins["measure"] == new_insight["measure"]:
                matches += 1

        return matches / (2 * len(prev))

    def _curiosity_raw(self, new_insight):
        """
        Encourage different time windows
        (proxy for CausalCNN distance)
        """
        start, end = new_insight["subspace"]
        duration = (end - start).days
        return min(duration / 365.0, 1.0)

    def _curiosity_derived(self, new_insight):
        """
        Proxy for statistical significance
        """
        series = self.dataset[
            (self.dataset["date"] >= new_insight["subspace"][0]) &
            (self.dataset["date"] <= new_insight["subspace"][1])
        ][new_insight["measure"]]

        if len(series) < 2:
            return 0.0

        std = series.std()
        return min(std / (series.mean() + 1e-6), 1.0)
