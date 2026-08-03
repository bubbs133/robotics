import mujoco
import mujoco.viewer
import os
import time
from ..my_models.loader import V1_SVC_MODEL, SVC_MODEL_PATH


model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

step = 0

gesture_map = {
    1: "thumbs up",
    2: "extension of index and middle, flexion on others",
    3: "flexion on ring and little finger, extension of others",
    6: "fingers flexed into fist",
    7: "pointing index",
}

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)
