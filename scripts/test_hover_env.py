import mujoco.viewer
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
envs_dir = os.path.join(script_dir, "..", "envs")
sys.path.insert(0, envs_dir)

from hover_env import HoverEnv

env = HoverEnv()
obs, info = env.reset(seed=0)

with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
    while viewer.is_running():
        action = env.action_space.sample()  # random for now, no PPO yet
        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()

        if terminated or truncated:
            print(
                f"Episode ended (terminated={terminated}, truncated={truncated}) — resetting"
            )
            obs, info = env.reset()
