import mujoco
import mujoco.viewer
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "single_drone.xml")

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
hover = sum(m.body_mass) * 9.81 / 4 # per rotor

class PIDState:
    def __init__(self):
        self.z_integral = 0.0
        self.pitch_integral = 0.0
        self.roll_integral = 0.0
        self.x_integral = 0.0
        self.y_integral = 0.0


def pid_controller(d, m, state, dt, target_z = 0.5, target_roll = 0.0, target_pitch = 0.0):
    # Altitude
    z_err = target_z - d.qpos[2]
    z_vel = d.qvel[2]
    state.z_integral = np.clip(state.z_integral + z_err * dt, -1.0, 1.0)
    Kp_z, Ki_z, Kd_z = 20.0, 2.0, 8.0
    base_thrust = (
        (sum(m.body_mass) * 9.81 / 4)
        + Kp_z * z_err
        + Ki_z * state.z_integral
        - Kd_z * z_vel
    )

    # Attitude
    roll, pitch = get_roll_pitch(d.qpos[3:7])
    roll_rate, pitch_rate = d.qvel[3], d.qvel[4]
    roll_err = target_roll - roll
    pitch_err = target_pitch - pitch
    state.roll_integral = np.clip(state.roll_integral + roll_err * dt, -0.5, 0.5)
    state.pitch_integral = np.clip(state.pitch_integral + pitch_err * dt, -0.5, 0.5)
    kp_att, ki_att, kd_att = 3.0, 0.5 , 0.5
    roll_corr = kp_att * roll_err + ki_att * state.roll_integral - kd_att * roll_rate
    pitch_corr = kp_att * pitch_err + ki_att * state.pitch_integral - kd_att * pitch_rate

    thrust_fr = base_thrust - roll_corr - pitch_corr
    thrust_fl = base_thrust + roll_corr - pitch_corr
    thrust_bl = base_thrust + roll_corr + pitch_corr
    thrust_br = base_thrust - roll_corr + pitch_corr

    ctrl = np.array([thrust_fr, thrust_fl, thrust_bl, thrust_br])
    return np.clip(ctrl, 0, 5)

def position_to_altitude(d, state, dt, target_x, target_y, max_tilt = 0.3):
    x_err = target_x - d.qpos[0]
    y_err = target_y - d.qpos[1]
    x_vel, y_vel = d.qvel[0], d.qvel[1]
    state.x_integral = np.clip(state.x_integral + x_err * dt, -0.5, 0.5)
    state.y_integral = np.clip(state.y_integral + y_err * dt, -0.5, 0.5)
    Kp_pos, Ki_pos, Kd_pos = 0.5, 0.02, 0.8

    desired_pitch = Kp_pos * x_err + Ki_pos * state.x_integral - Kd_pos * x_vel
    desired_roll = Kp_pos * y_err + Ki_pos * state.y_integral - Kd_pos * y_vel

    return np.clip(desired_roll, -max_tilt, max_tilt), np.clip(desired_pitch, -max_tilt, max_tilt)

def get_roll_pitch(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch

state = PIDState()
dt = 0.005
target_x, target_y = 1.0, 0.5  # pick a nonzero point to actually test movement

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        target_roll, target_pitch = position_to_altitude(
            d, state, dt, target_x, target_y
        )
        d.ctrl[:] = pid_controller(
            d, m, state, dt, target_roll=target_roll, target_pitch=target_pitch
        )
        mujoco.mj_step(m, d)
        viewer.sync()