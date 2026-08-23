import sys, os
import torch
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

from hover_env import HoverEnv
from policy import ActorCritic
from buffer import RolloutBuffer

env = HoverEnv()
policy = ActorCritic(obs_dim = 16, action_dim = 4)
buffer = RolloutBuffer(buffer_size = 2048, obs_dim = 16, action_dims = 4)

obs, info = env.reset()

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

