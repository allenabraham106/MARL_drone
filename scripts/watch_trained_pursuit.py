import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import numpy as np
import time
import mujoco.viewer
from pursuit_env import PursuitEnv
from policy import ActorCritic

env = PursuitEnv()
policy = ActorCritic(obs_dim=19, action_dim=4)
policy.load_state_dict(torch.load("checkpoints/pursuit_checkpoint_1788530632_900.pt"))
policy.eval()
obs, info = env.reset(seed=0)
step_count = 0

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    while viewer.is_running():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_mean, value = policy.forward(obs_tensor)
        action_np = torch.sigmoid(action_mean).numpy() * 5.0
        obs, reward, terminated, truncated, info = env.step(action_np)
        viewer.sync()
        time.sleep(0.02)

        step_count += 1
        if step_count <= 10:
            d1_pos = obs[0:3]
            d1_quat = obs[3:7]
            d1_vel = obs[7:10]
            d1_angvel = obs[10:13]
            rel_pos = obs[13:16]
            d2_vel = obs[16:19]
            print(f"step {step_count}")
            print(f"  d1_pos={d1_pos.round(3)} d1_quat={d1_quat.round(3)}")
            print(f"  d1_vel={d1_vel.round(3)} d1_angvel={d1_angvel.round(3)}")
            print(f"  rel_pos={rel_pos.round(3)} d2_vel={d2_vel.round(3)}")
            print(
                f"  action_mean(raw)={action_mean.numpy().round(3)} action(squashed)={action_np.round(3)}"
            )

        if terminated or truncated:
            print(f"Episode ended at step {step_count}\n")
            obs, info = env.reset()
            step_count = 0
