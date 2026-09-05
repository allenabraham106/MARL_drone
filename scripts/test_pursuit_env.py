import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import numpy as np
from pursuit_env import PursuitEnv

env = PursuitEnv()
obs, info = env.reset(seed=0)

for i in range(500):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        d1_height = env.d.qpos[2]
        dist = np.linalg.norm(env.d.qpos[7:10] - env.d.qpos[0:3])
        print(f"step {i}: ended | d1_height={d1_height:.3f} dist={dist:.3f}")
        obs, info = env.reset()

print("Ran 500 steps, no crashes.")
