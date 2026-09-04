import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
import os

class PursuitEnv(gym.Env):
    def __init__(self):
        super().__init__()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "models", "pursuit_evasion.xml")
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.max_steps = 1000
        self.step_count = 0
        self.flee_scale = 0.0
        self.intruder_state = IntruderState()
        self.dt = 0.005

        # only one RL controlled drone for now
        self.action_space = spaces.Box(low=0.0, high=5.0, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.intruder_state = IntruderState()

        # Drone1 fixed start
        self.d.qpos[0:3] = [0.0, 0.0, 0.5]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        #Drone2 random start some distance away
        offset = self.np_random.uniform(low=-1.5, high=1.5, size=2)
        self.d.qpos[7:9] = offset
        self.d.qpos[9] = 0.5
        self.d.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.m, self.d)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        drone1_pos = self.d.qpos[0:3]
        drone1_quat = self.d.qpos[3:7]
        drone1_vel = self.d.qvel[0:3]
        drone1_angvel = self.d.qvel[3:6]

        drone2_pos = self.d.qpos[7:10]
        drone2_vel = self.d.qvel[6:9]

        relative_pos = drone2_pos - drone1_pos

        return np.concatenate([
            drone1_pos,
            drone1_quat,
            drone1_vel,
            drone1_angvel,
            relative_pos,
            drone2_vel,
        ]).astype(np.float32)

    def step(self, action):
        d1_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.d.ctrl[0:4] = d1_action

        drone1_pos = self.d.qpos[0:3]
        intruder_action = scripted_intruder_step(
            self.d, self.m, self.intruder_state, self.dt, drone1_pos,
            flee_distance=1.0 * self.flee_scale
        )
        self.d.ctrl[4:8] = intruder_action

        mujoco.mj_step(self.m, self.d)
        self.step_count += 1

        obs = self._get_obs()

        d1_pos = self.d.qpos[0:3]
        d2_pos = self.d.qpos[7:10]
        dist = np.linalg.norm(d2_pos - d1_pos)

        reward = -dist

        terminating = False
        # did it crash
        if d1_pos[2] < 0.05:
            terminating = True
            reward -= 50.0

        # the closer the better
        if dist < 0.2:
            terminating = True
            reward += 20.0
        if dist > 4.0:
            terminating = True
            reward -= 10.0

        truncated = self.step_count >= self.max_steps
        info = {}
        return obs, reward, terminating, truncated, info


class IntruderState:
    def __init__(self):
        self.z_integral = 0.0
        self.roll_integral = 0.0
        self.pitch_integral = 0.0
        self.x_integral = 0.0
        self.y_integral = 0.0


def get_roll_pitch(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def scripted_intruder_step(d, m, state, dt, drone1_pos, flee_distance=1.0):
    d2_pos = d.qpos[7:10]
    d2_quat = d.qpos[10:14]
    d2_vel = d.qvel[6:9]
    d2_angvel = d.qvel[9:12]

    # evasion target: flee directly away from drone1
    away = d2_pos[0:2] - drone1_pos[0:2]
    dist = np.linalg.norm(away)
    if dist < 1e-6:
        away_dir = np.array([1.0, 0.0])
    else:
        away_dir = away / dist
    target_xy = np.clip(d2_pos[0:2] + away_dir * flee_distance, -2.0, 2.0)
    target_z = 0.5

    # outer loop: position -> desired roll/pitch 
    x_err = target_xy[0] - d2_pos[0]
    y_err = target_xy[1] - d2_pos[1]
    x_vel, y_vel = d2_vel[0], d2_vel[1]

    state.x_integral = np.clip(state.x_integral + x_err * dt, -0.5, 0.5)
    state.y_integral = np.clip(state.y_integral + y_err * dt, -0.5, 0.5)
    Kp_pos, Ki_pos, Kd_pos = 0.15, 0.01, 0.3

    desired_pitch = Kp_pos * x_err + Ki_pos * state.x_integral - Kd_pos * x_vel
    desired_roll = -(Kp_pos * y_err + Ki_pos * state.y_integral - Kd_pos * y_vel)
    max_tilt = 0.3
    desired_roll = np.clip(desired_roll, -max_tilt, max_tilt)
    desired_pitch = np.clip(desired_pitch, -max_tilt, max_tilt)

    # inner loop: attitude + altitude -> rotor thrusts 
    z_err = target_z - d2_pos[2]
    z_vel = d2_vel[2]
    state.z_integral = np.clip(state.z_integral + z_err * dt, -1.0, 1.0)
    Kp_z, Ki_z, Kd_z = 20.0, 2.0, 8.0
    base_thrust = (
        (sum(m.body_mass[1:]) * 9.81 / 4)
        + Kp_z * z_err
        + Ki_z * state.z_integral
        - Kd_z * z_vel
    )

    roll, pitch = get_roll_pitch(d2_quat)
    roll_rate, pitch_rate = d2_angvel[0], d2_angvel[1]
    roll_err = desired_roll - roll
    pitch_err = desired_pitch - pitch
    state.roll_integral = np.clip(state.roll_integral + roll_err * dt, -0.5, 0.5)
    state.pitch_integral = np.clip(state.pitch_integral + pitch_err * dt, -0.5, 0.5)
    Kp_att, Ki_att, Kd_att = 3.0, 0.5, 0.5
    roll_corr = Kp_att * roll_err + Ki_att * state.roll_integral - Kd_att * roll_rate
    pitch_corr = (
        Kp_att * pitch_err + Ki_att * state.pitch_integral - Kd_att * pitch_rate
    )

    thrust_fr = base_thrust - roll_corr - pitch_corr
    thrust_fl = base_thrust + roll_corr - pitch_corr
    thrust_bl = base_thrust + roll_corr + pitch_corr
    thrust_br = base_thrust - roll_corr + pitch_corr

    return np.clip([thrust_fr, thrust_fl, thrust_bl, thrust_br], 0, 5)