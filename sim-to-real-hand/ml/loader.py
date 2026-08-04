import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SVC_MODEL_PATH = os.path.join(BASE_DIR, "ml", "svc_emg_6features_C10_gamma0.1_78_acc.pkl")

V1_SCALER_PATH = os.path.join(
    BASE_DIR, "ml", "v1_svc_scaler.pkl"
)

RAW_EMG_WINDOWS_PATH = os.path.join(BASE_DIR, "ml", "feature_batches.pkl")
RAW_EMG_LABELS_PATH = os.path.join(BASE_DIR, "ml", "label_batches.pkl")

V1_SVC_MODEL = joblib.load(SVC_MODEL_PATH)
V1_SVC_SCALER = joblib.load(V1_SCALER_PATH)

EMG_FEATURE_WINDOWS = joblib.load(RAW_EMG_WINDOWS_PATH)
EMG_LABLE_WINDOWS =joblib.load(RAW_EMG_LABELS_PATH)

"""def deserializing_data(emg_data):

    emg_data = joblib.load(RAW_EMG_WINDOWS_PATH)
    print(type(emg_data))
    print(emg_data.shape)


deserialize = deserializing_data(RAW_EMG_WINDOWS_PATH)"""