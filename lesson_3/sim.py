import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)

step = 0
target = 2.0

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
    while viewer.is_running():

        if step % 100 == 0:
            print(f"Step {step}, Joint 1: {data.qpos[0]}")

        step += 1

        error = target - data.qpos[0]

        data.ctrl[0] = error * 10

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)
