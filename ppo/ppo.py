import numpy as np
import torch
import torch.optim as optim

def compute_gae(rewards, values, dones, last_value, gamma=0.09, lam=0.95):
    advantages = np.zeros_like(rewards)
    last_gae = 0.0

    # reversed cause we need the future
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages

class PPO: 
    def __init__(self, policy, lr=3e-4, clip_eps=0.2, epochs=10, minibatch_size=64, value_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.clips_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

