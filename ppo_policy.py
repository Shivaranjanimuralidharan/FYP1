import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------
# LSTM-based State Encoder
# -------------------------
class LSTMEncoder(nn.Module):
    """
    Encodes a sequence of insight vectors into a single state vector
    (Section 5.3.3 in Visail paper)
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]  # final hidden state


# -------------------------
# Actor Network
# -------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        logits = self.fc(state)
        return torch.softmax(logits, dim=-1)


# -------------------------
# Critic Network
# -------------------------
class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        return self.fc(state)


# -------------------------
# PPO Agent
# -------------------------
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        gamma=0.9,
        lr=3e-4,
        clip_eps=0.2,
        device="cpu"
    ):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.device = device

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) +
            list(self.critic.parameters()),
            lr=lr
        )

    # -------------------------
    # Action Selection
    # -------------------------
    def select_action(self, state):
        """
        Used during training and inference
        """
        state_tensor = torch.tensor(
            state, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach()

    # -------------------------
    # PPO Update Step
    # -------------------------
    def update(
        self,
        states,
        actions,
        rewards,
        dones,
        log_probs,
        epochs=10,
        batch_size=64
    ):
        """
        PPO policy update (offline training only)
        """

        # Convert rollout data to tensors
        states = torch.tensor(
            np.array(states), dtype=torch.float32
        ).to(self.device)

        actions = torch.tensor(actions, dtype=torch.long).to(self.device)

        # IMPORTANT: detach old log probs
        old_log_probs = torch.stack(log_probs).detach().to(self.device)

        # -------------------------
        # Compute discounted returns
        # -------------------------
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(
            returns, dtype=torch.float32
        ).to(self.device)

        # -------------------------
        # Advantage estimation
        # -------------------------
       # Compute advantages ONCE using detached values
        with torch.no_grad():
            values_old = self.critic(states).squeeze()
            advantages = returns - values_old

        for _ in range(epochs):
            # --- Actor ---
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.clip_eps,
                1 + self.clip_eps
            ) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()

            # --- Critic (RECOMPUTE values here) ---
            values = self.critic(states).squeeze()
            critic_loss = nn.MSELoss()(values, returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

