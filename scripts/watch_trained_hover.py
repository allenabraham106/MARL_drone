import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ppo")
)

import torch
import mujoco.viewer
from hover_env import HoverEnv
from policy import ActorCritic

env = HoverEnv()
policy = ActorCritic(obs_dim=16, action_dim=4)
policy.load_state_dict(torch.load("checkpoints/checkpoint_11900.pt"))
policy.eval()

obs, info = env.reset(seed=0)

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    while viewer.is_running():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
