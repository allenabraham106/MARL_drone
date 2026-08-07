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

        # Action Space, 4 continuous rotor thrust
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(4,), dtype=np.float32)

        # Observation Space, what the policy gets to see
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # Hardcoded value, will be randomized
        self.target = np.array([0.0, 0.0, 0.5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset all physics states back to zero
        mujoco.mj_resetData(self.m, self.d)