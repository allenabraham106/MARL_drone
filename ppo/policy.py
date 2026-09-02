import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim = 16, action_dim = 4):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64,64),
            nn.Tanh(),
        )

        # Actor Head (output the mean of mean of each action dimension)
        self.actor_mean = nn.Linear(64, action_dim)

        # Learnable log
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic Head (outputs a number)
        self.critic = nn.Linear(64,1)

    def forward(self, obs):
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(self, obs):
        action_mean, value = self.forward(obs)
        log_std = torch.clamp(self.actor_log_std, min=-2.0, max=1.0)
        std = torch.exp(log_std)
        dist = Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value
