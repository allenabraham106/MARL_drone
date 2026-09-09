import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import numpy as np
from pursuit_env import PursuitEnv, IntruderState, scripted_pursuer_ctbr

env = PursuitEnv()
env.flee_scale = 1.0  # full evasion difficulty, matching real training conditions

pursuer_state = IntruderState()

n_episodes = 200
observations = []
actions = []

for ep in range(n_episodes):
    obs, info = env.reset()
    pursuer_state = IntruderState()

    for step in range(1000):
        drone2_pos = env.d.qpos[7:10]
        action = scripted_pursuer_ctbr(env.d, env.m, pursuer_state, env.dt, drone2_pos)

        observations.append(obs.copy())
        actions.append(action.copy())

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    if ep % 20 == 0:
        print(
            f"episode {ep}/{n_episodes}, collected {len(observations)} samples so far"
        )

observations = np.array(observations, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

print(f"Final dataset: {observations.shape[0]} samples")
np.save("demo_observations.npy", observations)
np.save("demo_actions.npy", actions)
print("Saved demo_observations.npy and demo_actions.npy")
