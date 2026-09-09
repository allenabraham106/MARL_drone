import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import numpy as np
from pursuit_env import PursuitEnv, IntruderState, scripted_pursuer_ctbr

env = PursuitEnv()
env.flee_scale = 1.0
pursuer_state = IntruderState()

catches, crashes, losts, timeouts = 0, 0, 0, 0

for ep in range(50):
    obs, info = env.reset()
    pursuer_state = IntruderState()

    for step in range(1000):
        drone2_pos = env.d.qpos[7:10]
        action = scripted_pursuer_ctbr(env.d, env.m, pursuer_state, env.dt, drone2_pos)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            d1_height = env.d.qpos[2]
            dist = np.linalg.norm(env.d.qpos[7:10] - env.d.qpos[0:3])
            if d1_height < 0.05:
                crashes += 1
            elif dist < 0.2:
                catches += 1
            elif dist > 4.0:
                losts += 1
            else:
                timeouts += 1
            break

print(
    f"catches: {catches}/50 | crashes: {crashes}/50 | lost: {losts}/50 | timeouts: {timeouts}/50"
)
