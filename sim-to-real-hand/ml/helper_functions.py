from loader import V1_SVC_MODEL, V1_SVC_SCALER, EMG_FEATURE_WINDOWS, EMG_LABLE_WINDOWS
from feature_extractions import extracting_features
from gesture_map import GESTURE_MAP

import numpy as np


def calc_features(data):
    features = extracting_features(data)
    return features

def scaling_data(data):
    calculated_features = calc_features(data)

    scaled_features = V1_SVC_SCALER.transform(calculated_features)

    return scaled_features

def predict(data):
    scaled_data = scaling_data(data)

    model_pred = V1_SVC_MODEL.predict(scaled_data)

    model_pred = GESTURE_MAP.get(model_pred, "unknown")

    return model_pred


pred = predict(EMG_FEATURE_WINDOWS)
print(pred)