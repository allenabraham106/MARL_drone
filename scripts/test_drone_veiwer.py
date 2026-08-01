import mujoco
import mujoco.viewer
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "..", "models", "single_drone.xml")

m = mujoco.MjModel.from_xml_path(model_path)
d = mujoco.MjData(m)
hover = sum(m.body_mass) * 9.81 / 4 # per rotor

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        d.ctrl[:] = hover
        mujoco.mj_step(m,d)
        viewer.sync()