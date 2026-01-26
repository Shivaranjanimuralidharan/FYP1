import os
import torch
import numpy as np
from tqdm import trange

from eva_env import EVAEnvironment
from ppo_policy import PPOAgent


# -----------------------------
# CONFIGURATION (paper-aligned)
# -----------------------------
DATASET_PATH = "./uploads1/covid_timeseries.csv"   # same dataset as paper
SAVE_PATH = "ppo_weights.pt"

MAX_EPISODE_LENGTH = 5        # N = 5 (paper)
TOTAL_TRAINING_STEPS = 100_000
GAMMA = 0.9                   # discount factor
UPDATE_EVERY = 2048           # PPO rollout size
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1️⃣ Create EVA environment (MDP)
    env = EVAEnvironment(
        dataset_path=DATASET_PATH,
        max_steps=MAX_EPISODE_LENGTH
    )

    # 2️⃣ Initialize PPO agent
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        gamma=GAMMA,
        lr=LEARNING_RATE,
        device=device
    )

    state = env.reset()
    step_count = 0

    # Buffers for PPO rollout
    states, actions, rewards, dones, log_probs = [], [], [], [], []

    for step in trange(TOTAL_TRAINING_STEPS, desc="PPO Training"):
        step_count += 1

        # 3️⃣ Agent selects action
        action, log_prob = agent.select_action(state)

        # 4️⃣ Environment transition
        next_state, reward, done, info = env.step(action)

        # 5️⃣ Store transition
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)

        state = next_state

        # 6️⃣ Episode ended → reset environment
        if done:
            state = env.reset()

        # 7️⃣ PPO update step
        if step_count % UPDATE_EVERY == 0:
            agent.update(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                log_probs=log_probs,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            # Clear buffers
            states, actions, rewards, dones, log_probs = [], [], [], [], []

    # 8️⃣ Save trained weights
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "critic": agent.critic.state_dict(),
        },
        SAVE_PATH
    )

    print(f"Training complete. PPO weights saved to {SAVE_PATH}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    train()
