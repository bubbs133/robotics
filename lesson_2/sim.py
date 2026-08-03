import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt

model = mujoco.MjModel.from_xml_path("world.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model=model)

step = 0

target_angle = 1.0

with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
    while viewer.is_running():

        mujoco.mj_step(model, data)
        renderer.update_scene(data=data)
        img = renderer.render()

        if step % 100 == 0:
            print(f"angle: {data.qpos[0]}")
            #print(f"img: {img.shape}")
            # plt.imsave(f"frame{step}.png", img)
            # plt.show()

        step += 1

        current_angle = data.qpos[0]

        if current_angle < target_angle:
            # this controls the motor, keeps pushing w the force of 1 forever
            data.ctrl[0] = 1
        else:
            data.ctrl[0] = 0

        viewer.sync()

        time.sleep(0.01)
