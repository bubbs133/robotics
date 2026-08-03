import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("world2.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        data.ctrl[0] = 10

        mujoco.mj_step(model, data)

        viewer.sync()
