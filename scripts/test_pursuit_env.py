import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import numpy as np
from pursuit_env import PursuitEnv

env = PursuitEnv()
print("action_space:", env.action_space)
print("observation_space:", env.observation_space)

obs, info = env.reset(seed=0)
print("reset obs shape:", obs.shape)

n_episodes = 0
for i in range(2000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        n_episodes += 1
        d1_z = env.d.qpos[2]
        dist = np.linalg.norm(env.d.qpos[7:10] - env.d.qpos[0:3])
        print(f"step {i}: ended | d1_height={d1_z:.3f} | dist={dist:.3f} | terminated={terminated} truncated={truncated}")
        obs, info = env.reset()

print(f"Ran 2000 steps, {n_episodes} episodes completed, no crashes.")
