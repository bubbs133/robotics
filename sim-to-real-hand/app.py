import streamlit as st
import numpy as np
import pandas as pd

from ml.loader import RAW_EMG_TEST_WINDOWS
from ml.predict import predict
from ml.feature_extractions import extracting_features
from simulation.gesture_map import GESTURE_MAP

st.set_page_config(
    page_title="EMG to MuJoCo Hand Demo", page_icon="🤖", layout="centered"
)

st.title("Real Time EMG Gesture Classification & MuJoCo Control")
st.markdown(
    "This mini project was made possible by the **Ninapro DB1** dataset. In this demo, you'll select a random EMG window and run the ML pipline as follows: \n\n" 
    "• six features are be calculated from the raw EMG window. These features are Mean Absolute Values (MAV), Wave Length (WL), Root Mean Square (RMS), Zero Crossing (ZC), Slope Sign Changes (SSC), and Variance. \n\n "
    "• a **SVM** then scales the values and classifies the features into a gesture \n\n"
    "• this triggers the corresponding hand movement on **MuJoCo**."
)

st.divider()


# --------------------------------------------------
# RUN DEMO
# --------------------------------------------------

if st.button("Run Random EMG Window", use_container_width=True):

    window_index = np.random.randint(len(RAW_EMG_TEST_WINDOWS))

    window = RAW_EMG_TEST_WINDOWS[window_index]

    st.session_state.window = window
    st.session_state.window_index = window_index

    # Extract the 60 features
    features = extracting_features(window)

    st.session_state.features = features

    # Predict gesture
    prediction = predict(window)
    pred_gesture = GESTURE_MAP.get(prediction)

    st.session_state.prediction = prediction

    # ------------------------------
    # MUJOCO
    # ------------------------------


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "prediction" in st.session_state:

    prediction = st.session_state.prediction

    window_index = st.session_state.window_index

    window = st.session_state.window

    # --------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric("EMG Window", f"#{window_index}")

    with col2:

        st.metric("Samples", window.shape[0])

    with col3:

        st.metric("Channels", window.shape[1])

    st.divider()

    # --------------------------------------------------
    # PIPELINE
    # --------------------------------------------------

    st.subheader("ML Pipeline")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.info("\n\n" "**Raw EMG**\n\n" "250 × 10")

    with col2:

        st.info("\n\n" "**Features**\n\n" "60 features")

    with col3:

        st.info("\n\n" "**SVM**\n\n" "RBF classifier")

    with col4:

        st.success("\n\n" f"**Gesture**\n\n" f"{prediction} \n\n {pred_gesture}")

    st.divider()

    # --------------------------------------------------
    # EMG SIGNAL
    # --------------------------------------------------

    st.subheader("Selected EMG Window")

    channel = st.selectbox("Select EMG channel", range(window.shape[1]))

    st.line_chart(window[:, channel])

    st.divider()

    with st.expander("Extracted EMG Features"):

        feature_matrix = features.reshape(6, 10)

        feature_df = pd.DataFrame(
            feature_matrix,
            index=[
                "MAV",
                "RMS",
                "Waveform Length",
                "ZC",
                "SSC",
                "Variance",
            ],
            columns=[f"EMG {i + 1}" for i in range(10)],
        )

        st.dataframe(feature_df, use_container_width=True)

    # --------------------------------------------------
    # RAW DATA INFO
    # --------------------------------------------------

    with st.expander("View raw window"):

        st.write(window)
