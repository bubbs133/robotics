import mujoco
import mujoco.viewer
import os
import time
import joblib
from ..ml.loader import V1_SVC_MODEL, SVC_MODEL_PATH, V1_SVC_SCALER


world = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(world)

with mujoco.viewer.launch_passive(model=world, data=data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(world, data)

        viewer.sync()

        time.sleep(0.01)
