import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import numpy as np
from hover_env import HoverEnv
from policy import ActorCritic
from buffer import RolloutBuffer
from ppo import compute_gae, PPO

env = HoverEnv()
policy = ActorCritic(obs_dim=16, action_dim=4)
buffer = RolloutBuffer(buffer_size=2048, obs_dim=16, action_dim=4)
ppo = PPO(policy)

obs, info = env.reset()
n_iterations = 200

for iteration in range(n_iterations):
    buffer.reset()

    for _ in range(buffer.buffer_size):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        action, log_prob, value = policy.get_action(obs_tensor)

        action_np = action.detach().numpy()
        clipped_action = np.clip(action_np, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, info = env.step(clipped_action)
        done = terminated or truncated

        buffer.store(obs, action_np, log_prob.item(), value.item(), reward, done)

        obs = next_obs
        if done:
            obs, info = env.reset()

    print("Buffer full:", buffer.is_full())
    print("Mean reward collected:", buffer.rewards.mean())

    with torch.no_grad():
        _, last_value = policy.forward(torch.as_tensor(obs, dtype=torch.float32))
    last_value = last_value.item()

    advantages, returns = compute_gae(
        buffer.rewards, buffer.values, buffer.dones, last_value
    )
    print("advantages: mean", advantages.mean(), "std", advantages.std())
    print("returns: mean", returns.mean(), "std", returns.std())

    data = buffer.get()
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)

    ppo.update(data["obs"], data["actions"], data["log_probs"], advantages_t, returns_t)
    mean_reward = buffer.rewards.mean()
    episode_ends = np.where(buffer.dones == 1)[0]
    mean_episode_length = (
        np.mean(np.diff(np.concatenate([[-1], episode_ends])))
        if len(episode_ends) > 0
        else buffer.buffer_size
    )
    print(f"Iteration {iteration:4d} | mean reward: {mean_reward:.3f} | mean episode length: {mean_episode_length:.1f}")
print("PPO update complete")
