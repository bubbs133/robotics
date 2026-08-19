import streamlit as st
import numpy as np
import pandas as pd
import socket

from ml.loader import RAW_EMG_TEST_WINDOWS, RAW_EMG_TEST_LABEL_WINDOWS
from ml.predict import predict
from ml.feature_extractions import extracting_features
from simulation.gesture_map import GESTURE_MAP, GESTURE_TARGETS

st.set_page_config(
    page_title="EMG to MuJoCo Hand Demo", page_icon="🦾", layout="centered"
)

st.title("Real Time EMG Gesture Classification & MuJoCo Control")

tab_demo, tab_about = st.tabs(["Run Demo", "About the Project"])

# --------------------------------------------------
# MUJOCO SEND
# --------------------------------------------------


def send_gesture(prediction):

    try:

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        sock.connect(("127.0.0.1", 5001))
        sock.sendall(str(prediction).encode())
        sock.close()

        return True

    except (ConnectionRefusedError, socket.timeout, OSError):

        return False


# --------------------------------------------------
# ABOUT TAB
# --------------------------------------------------

with tab_about:

    st.subheader("What this is")

    st.markdown(
        "An end-to-end pipeline that takes a raw surface-EMG window from the "
        "**Ninapro DB1** dataset, extracts hand-crafted features, classifies "
        "the gesture with a trained SVM, and drives a simulated hand in "
        "**MuJoCo** in real time."
    )

    st.subheader("Pipeline")

    st.markdown(
        "1. **Raw EMG window** — 250 samples × 10 channels\n"
        "2. **Feature extraction** — 6 features per channel (60 total): "
        "Mean Absolute Value, Root Mean Square, Waveform Length, Zero "
        "Crossings, Slope Sign Changes, Variance\n"
        "3. **Scaling** — features normalized with a fitted `StandardScaler`\n"
        "4. **Classification** — RBF-kernel SVM predicts one of the trained "
        "gesture classes\n"
        "5. **Simulation** — the predicted gesture ID is sent over a local "
        "TCP socket to a MuJoCo viewer process, which drives 6 position "
        "actuators (wrist, thumb, index, middle, ring, pinky) to the target "
        "joint angles for that gesture"
    )

    st.subheader("Gestures in this demo")

    gesture_df = pd.DataFrame(
        [
            {"ID": gid, "Name": name, "Joint targets (rad)": GESTURE_TARGETS.get(gid)}
            for gid, name in GESTURE_MAP.items()
        ]
    )
    st.dataframe(gesture_df, use_container_width=True, hide_index=True)

    st.caption(
        "Joint targets are ordered [wrist, thumb, index, middle, ring, pinky]. "
        "For the thumb/wrist, lower = more flexed; for the four fingers, "
        "higher = more flexed."
    )

    st.subheader("Dataset")
    st.markdown(
        "[Ninapro DB1](http://ninapro.hevs.ch/) — 10-channel surface EMG "
        "recorded from the forearm at 100 Hz while subjects performed a set "
        "of hand and finger movements."
    )

# --------------------------------------------------
# DEMO TAB
# --------------------------------------------------

with tab_demo:

    st.markdown(
        "Select a random EMG window and run it through the full pipeline: "
        "feature extraction → SVM classification → MuJoCo hand control."
    )

    st.divider()

    if st.button("Run Random EMG Window", use_container_width=True):

        window_index = np.random.randint(len(RAW_EMG_TEST_WINDOWS))
        window = RAW_EMG_TEST_WINDOWS[window_index]

        features = extracting_features(window)
        prediction = predict(window)

        true_label = None
        if RAW_EMG_TEST_LABEL_WINDOWS is not None:
            try:
                true_label = int(np.ravel(RAW_EMG_TEST_LABEL_WINDOWS[window_index])[0])
            except (IndexError, TypeError, ValueError):
                true_label = None

        st.session_state.window = window
        st.session_state.window_index = window_index
        st.session_state.features = features
        st.session_state.prediction = prediction
        st.session_state.true_label = true_label

        sent = send_gesture(prediction)

        if not sent:
            st.warning(
                "MuJoCo is not running. Start it with "
                "`mjpython simulation/server.py`."
            )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    if "prediction" in st.session_state:

        prediction = st.session_state.prediction
        window_index = st.session_state.window_index
        window = st.session_state.window
        features = st.session_state.features
        true_label = st.session_state.get("true_label")
        pred_gesture = GESTURE_MAP.get(prediction, "Unknown")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("EMG Window", f"#{window_index}")
        with col2:
            st.metric("Samples", window.shape[0])
        with col3:
            st.metric("Channels", window.shape[1])

        st.divider()

        st.subheader("ML Pipeline")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("\n\n**Raw EMG**\n\n250 × 10")
        with col2:
            st.info("\n\n**Features**\n\n60 features")
        with col3:
            st.info("\n\n**SVM**\n\nRBF classifier")
        with col4:
            st.success(f"\n\n**Gesture**\n\n{prediction}\n\n{pred_gesture}")

        if true_label is not None:
            if true_label == prediction:
                st.success(
                    f"✅ Predicted class **{prediction}** matches the true "
                    f"label **{true_label}** ({GESTURE_MAP.get(true_label, 'Unknown')})."
                )
            else:
                st.error(
                    f"❌ Predicted class **{prediction}** "
                    f"({pred_gesture}) — true label is **{true_label}** "
                    f"({GESTURE_MAP.get(true_label, 'Unknown')})."
                )

        st.divider()

        st.subheader("Selected EMG Window")

        channel = st.selectbox("Select EMG channel", range(window.shape[1]))
        st.line_chart(window[:, channel])

        st.divider()

        st.subheader("Extracted EMG Features")

        feature_matrix = np.array(features).reshape(6, 10)
        feature_names = ["MAV", "RMS", "Waveform Length", "ZC", "SSC", "Variance"]

        feature_df = pd.DataFrame(
            feature_matrix,
            index=feature_names,
            columns=[f"Ch {i + 1}" for i in range(10)],
        )

        selected_feature = st.selectbox(
            "Feature to chart across channels", feature_names
        )
        st.bar_chart(feature_df.loc[selected_feature])

        with st.expander("Full feature table"):
            st.dataframe(
                feature_df.style.background_gradient(cmap="Blues", axis=1),
                use_container_width=True,
            )

        with st.expander("View raw window"):
            st.write(window)
