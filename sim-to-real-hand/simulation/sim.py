import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from simulation.gesture_map import GESTURE_MAP, GESTURE_TARGETS

WORLD_PATH = Path(__file__).parent / "world.xml"


class HandSimulation:

    def __init__(self):

        self.model = mujoco.MjModel.from_xml_path(str(WORLD_PATH))

        self.data = mujoco.MjData(self.model)

        self.viewer = None

    def launch(self):

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def move_to_gesture(self, prediction, steps=100, settle_steps=30):
        """
        Ramp data.ctrl (the position-actuator setpoint) from wherever it
        currently is to the gesture target, stepping physics along the
        way. Because the actuators in world.xml are now <position> type,
        MuJoCo's internal PD control does the actual work of moving each
        joint toward data.ctrl -- this method just needs to move the
        setpoint smoothly and let physics catch up.
        """

        if prediction not in GESTURE_TARGETS:
            return

        target = np.array(GESTURE_TARGETS[prediction], dtype=float)

        if len(target) != self.model.nu:

            raise ValueError(
                f"Expected {self.model.nu} actuator targets " f"but got {len(target)}."
            )

        start = self.data.ctrl.copy()

        for i in range(steps):

            alpha = (i + 1) / steps

            self.data.ctrl[:] = start + alpha * (target - start)

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

            time.sleep(0.01)

        # hold the final pose for a bit so the joints settle instead of
        # snapping to a stop mid-motion
        for _ in range(settle_steps):

            mujoco.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

            time.sleep(0.01)

    def step(self):

        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()
