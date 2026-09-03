import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import numpy as np
from pursuit_env import PursuitEnv
from transfer import transfer_hover_to_pursuit

env = PursuitEnv()
policy = transfer_hover_to_pursuit("checkpoints/checkpoint_11900.pt")
policy.eval()

obs, info = env.reset(seed=0)

heights = []
for i in range(500):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action_mean, value = policy.forward(obs_tensor)
    action_np = action_mean.numpy()

    obs, reward, terminated, truncated, info = env.step(action_np)
    heights.append(env.d.qpos[2])

    if terminated or truncated:
        print(
            f"step {i}: episode ended (terminated={terminated}) — d1_height={env.d.qpos[2]:.3f}"
        )
        obs, info = env.reset()

heights = np.array(heights)
print(
    f"\nHeight stats over 500 steps: mean={heights.mean():.3f}, min={heights.min():.3f}, max={heights.max():.3f}"
)
print(f"Fraction of steps above 0.3m: {(heights > 0.3).mean():.2%}")
