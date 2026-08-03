import mujoco
import mujoco.viewer
import os
import time

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

step = 0

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            print(data.qpos[2])

        step += 1

        viewer.sync()

        time.sleep(0.01)