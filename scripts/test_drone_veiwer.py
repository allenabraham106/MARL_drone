import mujoco
import mujoco.viewer
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "single_drone.xml")

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
hover = sum(m.body_mass) * 9.81 / 4 # per rotor

def get_roll_pitch(quat):
    w, x, y, z = quat
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch

with mujoco.viewer.launch_passive(m, d) as viewer:
    step_count = 0
    while viewer.is_running():
        d.ctrl[:] = hover
        mujoco.mj_step(m,d)
        viewer.sync()

        step_count += 1
        if step_count % 100 == 0:
            roll, pitch = get_roll_pitch(d.qpos[3:7])
            print(f"pos: {d.qpos[:3].round(3)}  roll: {np.degrees(roll):6.2f}°  pitch: {np.degrees(pitch):6.2f}°")
