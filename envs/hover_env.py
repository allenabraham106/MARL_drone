import gymnasium as gym 
from gymnasium import spaces
import mujoco
import numpy as np
import os

class HoverEnv(gym.Env):
    def __init__(self):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "models", "single_drone.xml")
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.max_steps = 1000
        self.step_count = 0

        # Action Space, 4 continuous rotor thrust
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(4,), dtype=np.float32)

        # Observation Space, what the policy gets to see
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # Hardcoded value, will be randomized
        self.target = np.array([0.0, 0.0, 0.5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Reset all physics states back to zero
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = [0.0, 0.0, 0.5] # start position
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0] # level quaterion

        # Pick a new target for the episode
        self.target = self.np_random.uniform(low=-0.3, high=0.3, size=3)
        self.target[2] = np.clip(self.target[2] + 0.5, 0.4, 0.6)  # keep the target above the ground

        mujoco.mj_forward(self.m, self.d) # recompute derived quantities
        obs = self._get_obs()
        info = {}

        return obs, info

    def _get_obs(self):
        return np.concatenate([
            self.d.qpos[0:3],
            self.d.qpos[3:7],
            self.d.qvel[0:3],
            self.d.qvel[3:6],
            self.target,
        ]).astype(np.float32)

    def step(self, action):
        self.d.ctrl[:] = np.clip(action, self.action_space.low, self.action_space.high)
        mujoco.mj_step(self.m, self.d)
        self.step_count += 1

        obs = self._get_obs()

        # Reward
        pos_err = np.linalg.norm(self.d.qpos[0:3] - self.target)
        reward = -pos_err # closer the target the less negative and the higher the number

        # Terminating Conditions
        terminating = False

        # crashed into the ground
        if self.d.qpos[2] < 0.05:
            terminating = True
            reward -= 10

        # flew way off
        if pos_err > 5.0:
            terminating = True
            reward -= 10.0

        # max episode length
        truncated = self.step_count >= self.max_steps
        info = {}
        return obs, reward, terminating, truncated, info

    

