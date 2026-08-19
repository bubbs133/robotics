import socket
import time

import mujoco
import mujoco.viewer
import numpy as np

from simulation.gesture_map import GESTURE_MAP, GESTURE_TARGETS

HOST = "127.0.0.1"
PORT = 5001


model = mujoco.MjModel.from_xml_path("simulation/world.xml")

data = mujoco.MjData(model)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)

print(f"MuJoCo server listening on {HOST}:{PORT}")


with mujoco.viewer.launch_passive(model, data) as viewer:

    server.settimeout(0.01)

    connection = None

    while viewer.is_running():

        # -----------------------------
        # Check for new connection
        # -----------------------------

        if connection is None:

            try:

                connection, address = server.accept()

                print(f"Connected from {address}")

                connection.settimeout(0.01)

            except socket.timeout:
                pass

        # -----------------------------
        # Receive gesture
        # -----------------------------

        if connection is not None:

            try:

                message = connection.recv(1024)

                if message:

                    prediction = int(message.decode().strip())

                    print("Received gesture:", prediction)

                    if prediction in GESTURE_TARGETS:

                        target = np.array(GESTURE_TARGETS[prediction], dtype=float)

                        data.ctrl[:] = target

                else:

                    connection.close()
                    connection = None

            except socket.timeout:
                pass

        # -----------------------------
        # MuJoCo
        # -----------------------------

        mujoco.mj_step(model, data)

        viewer.sync()

        time.sleep(0.01)
