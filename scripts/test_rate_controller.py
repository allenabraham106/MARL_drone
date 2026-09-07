import sys, os

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs")
)

import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "models", "single_drone.xml"
    )
)
d = mujoco.MjData(m)


def rate_controller(d, m, desired_thurst, desired_rates, body_offset=0, qpos_offset=0):
    angvel = d.qvel[3 + body_offset : 6 + body_offset]
    rate_err = np.array(desired_rates) - angvel
    Kp_rate = 0.15

    J_diag = np.array([0.0023, 0.0023, 0.004])
    Jw = J_diag * angvel
    gyro_term = np.cross(angvel, Jw)

    roll_corr = Kp_rate * rate_err[0] + gyro_term[0]
    pitch_corr = Kp_rate * rate_err[1] + gyro_term[1]
    yaw_corr = Kp_rate * rate_err[2] + gyro_term[2]

    base = desired_thurst / 4.0

    thrust_fr = base - roll_corr - pitch_corr + yaw_corr
    thrust_fl = base + roll_corr - pitch_corr - yaw_corr
    thrust_bl = base + roll_corr + pitch_corr + yaw_corr
    thrust_br = base - roll_corr + pitch_corr - yaw_corr

    return np.clip([thrust_fr, thrust_fl, thrust_bl, thrust_br], 0, 5)


hover_thrust_total = sum(m.body_mass) * 9.81  # total, not per-rotor

with mujoco.viewer.launch_passive(m, d) as viewer:
    step_count = 0
    while viewer.is_running():
        # ctrl = rate_controller(d, m, hover_thrust_total, [0.0, 0.0, 0.0])
        ctrl = rate_controller(d, m, hover_thrust_total, [0.0, 0.0, 2.0])

        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)
        viewer.sync()

        step_count += 1
        if step_count % 100 == 0:
            print(f"pos: {d.qpos[:3].round(3)}  angvel: {d.qvel[3:6].round(3)}")
