import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import numpy as np
from pursuit_env import PursuitEnv

env = PursuitEnv()
env.flee_scale = 1.0  # full evasion, matching end-of-training conditions
obs, info = env.reset(seed=0)

hover_thrust = 1.2255  # ~sum(mass)*9.81/4 for a single 0.5kg drone
action = np.array([hover_thrust, hover_thrust, hover_thrust, hover_thrust])

for i in range(20):
    obs, reward, terminated, truncated, info = env.step(action)
    d1_pos = env.d.qpos[0:3]
    d2_pos = env.d.qpos[7:10]
    dist = np.linalg.norm(d2_pos - d1_pos)
    print(
        f"step {i}: d1_pos={d1_pos.round(3)} d2_pos={d2_pos.round(3)} dist={dist:.3f} terminated={terminated}"
    )
    if terminated or truncated:
        print("EPISODE ENDED")
        break