import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import time
import mujoco.viewer
from pursuit_env import PursuitEnv
from policy import ActorCritic

env = PursuitEnv()
policy = ActorCritic(obs_dim=19, action_dim=4)
policy.load_state_dict(torch.load("checkpoints/pursuit_checkpoint_1788460085_2400.pt"))
policy.eval()
obs, info = env.reset(seed=0)

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    while viewer.is_running():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, log_prob, value = policy.get_action(obs_tensor)

        action_np = action.numpy()
        obs, reward, terminated, truncated, info = env.step(action_np)
        viewer.sync()
        time.sleep(0.02)

        if terminated or truncated:
            d1_height = env.d.qpos[2]
            dist = (
                float(
                    (env.d.qpos[7:10] - env.d.qpos[0:3])
                    @ (env.d.qpos[7:10] - env.d.qpos[0:3])
                )
                ** 0.5
            )
            print(
                f"Episode ended | terminated={terminated} truncated={truncated} | d1_height={d1_height:.3f} | dist={dist:.3f} — resetting"
            )
            obs, info = env.reset()
