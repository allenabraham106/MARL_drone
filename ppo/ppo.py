import numpy as np

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
