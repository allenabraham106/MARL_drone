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
from policy import ActorCritic


def evaluate_checkpoint(path, n_episodes=50):
    env = PursuitEnv()
    env.flee_scale = 1.0
    policy = ActorCritic(obs_dim=19, action_dim=4)
    policy.load_state_dict(torch.load(path))
    policy.eval()

    catches, crashes, losts, heights, lengths = 0, 0, 0, [], []

    for _ in range(n_episodes):
        obs, info = env.reset()
        step_count = 0
        ep_heights = []
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action_mean, value = policy.forward(obs_tensor)
            thrust = torch.sigmoid(action_mean[0:1]) * 6.0
            rates = torch.tanh(action_mean[1:4]) * 5.0
            action_np = torch.cat([thrust, rates]).numpy()

            obs, reward, terminated, truncated, info = env.step(action_np)
            ep_heights.append(env.d.qpos[2])
            step_count += 1

            if terminated or truncated:
                d1_height = env.d.qpos[2]
                dist = np.linalg.norm(env.d.qpos[7:10] - env.d.qpos[0:3])
                if d1_height < 0.05:
                    crashes += 1
                elif dist < 0.2:
                    catches += 1
                elif dist > 4.0:
                    losts += 1
                heights.append(np.mean(ep_heights))
                lengths.append(step_count)
                break

    print(f"{path}")
    print(
        f"  catch rate: {catches}/{n_episodes} | crash rate: {crashes}/{n_episodes} | lost rate: {losts}/{n_episodes}"
    )
    print(
        f"  mean episode length: {np.mean(lengths):.1f} | mean height: {np.mean(heights):.3f}"
    )


for ckpt in [
    "checkpoints/pursuit_checkpoint_1788753005_11900.pt",
    "checkpoints/pursuit_checkpoint_1788753005_11800.pt",
    "checkpoints/pursuit_checkpoint_1788753005_11700.pt",
    "checkpoints/pursuit_checkpoint_1788753005_11600.pt",
    "checkpoints/pursuit_checkpoint_1788753005_11500.pt",
]:
    evaluate_checkpoint(ckpt)

