import numpy as np
import torch
import torch.optim as optim

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
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
    return advantages, returns

class PPO: 
    def __init__(self, policy, lr=3e-4, clip_eps=0.2, epochs=10, minibatch_size=64, value_coef=0.5, entropy_coef=0.01):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.clips_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def update(self, obs, actions, old_logs_prob, advantages, returns):
        # making training stable
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_size = obs.shape[0]

        # Specific to PPO 
        for epoch in range(self.epochs):
            indicies = torch.randperm(dataset_size)

            for start in range(0, dataset_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = indicies[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logs_prob = old_logs_prob[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]

                # rerun the current network on same stored observations
                action_mean, values = self.policy.forward(batch_obs)
                std = torch.exp(self.policy.actor_log_std)
                dist = torch.distributions.Normal(action_mean, std)
                # how likely the current networwork would consider the same action
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                # meansures how random the polcy is currently distributed
                entropy = dist.entropy().sum(dim=-1).mean()

                # how much more or less likely is the current policy making this action compared to before
                ratio = torch.exp(new_log_probs - batch_old_logs_prob)

                # action turned out beter than expected we want to reinforce it
                surr1 = ratio * batch_advantages

                # Action was worse than expected, we want to discourage it
                surr2 = torch.clamp(ratio, 1 - self.clips_eps, 1 + self.clips_eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = ((values.squeeze(-1) - batch_returns) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                

