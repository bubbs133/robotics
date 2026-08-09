from ml.loader import (
    V1_SVC_MODEL,
    V1_SVC_SCALER,
    RAW_EMG_TEST_WINDOWS,
    RAW_EMG_TEST_LABEL_WINDOWS,
)
from ml.feature_extractions import extracting_features
from mujoco.gesture_map import GESTURE_MAP

import numpy as np

"""def predict(emg_window):
    features_per_window = extracting_features(emg_window)

    scaled_features_per_window = V1_SVC_SCALER.transform(features_per_window)

    model_pred = V1_SVC_MODEL.predict(scaled_features_per_window)[0]

    return GESTURE_MAP.get(model_pred, "Unknown")"""


def predict(window):

    print("Raw window:", window.shape)

    features = extracting_features(window)

    print("Extracted features:", features.shape)

    scaled = V1_SVC_SCALER.transform(features)

    prediction = V1_SVC_MODEL.predict(scaled)[0]

    return prediction


pred = predict(RAW_EMG_TEST_WINDOWS[0])
print(pred)
