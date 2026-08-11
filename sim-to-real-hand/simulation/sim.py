import simulation
import mujoco.viewer
import time
from pathlib import Path

from ml.loader import RAW_EMG_TEST_WINDOWS
from ml.predict import predict

GESTURE_TARGETS = {
    1: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    2: [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    3: [2.0, 0.8, 0.8, 0.0, 0.0, 0.0],
    6: [2.0, 0.5, 0.5, 0.5, 0.5, 0.5],
    7: [2.0, 0.2, 0.2, 0.2, 0.2, 0.2],
}


WORLD_PATH = Path(__file__).parent / "world.xml"

model = simulation.MjModel.from_xml_path(str(WORLD_PATH))
data = simulation.MjData(model)


with simulation.viewer.launch_passive(model=model, data=data) as viewer:

    for window in RAW_EMG_TEST_WINDOWS:

        prediction = predict(window)

        print("Predicted gesture:", prediction)


        if prediction in GESTURE_TARGETS:

            target = GESTURE_TARGETS[prediction]

            # Apply target to all actuators
            data.ctrl[:] = target

        # Let MuJoCo simulate for a little while
        for _ in range(2):

            simulation.mj_step(model, data)

            viewer.sync()

            time.sleep(0.01)

        if not viewer.is_running():
            break
