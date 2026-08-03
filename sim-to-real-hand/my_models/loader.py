import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SVC_MODEL_PATH = os.path.join(BASE_DIR, "my_models", "svc_emg_6features_C10_gamma0.1_78_acc.pkl")

"""V2_SCALER_PATH = os.path.join(
    BASE_DIR, "ml", "v2_scaler_gmm_diag_updated3_features.pkl"
)"""

V1_SVC_MODEL = joblib.load(SVC_MODEL_PATH)
#v2_gmm_scaler = joblib.load(V2_SCALER_PATH)
