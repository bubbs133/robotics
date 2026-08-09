import numpy as np
import joblib

from ml.loader import RAW_EMG_TEST_WINDOWS, RAW_EMG_TEST_LABEL_WINDOWS, RAW_EMG_WINDOWS_PATH


def mav(emg_window):
    mav_baches = np.mean(np.abs(emg_window), axis=1)
    return mav_baches


def rms(emg_window):
    rms_values = np.sqrt(np.mean(emg_window**2, axis=1))
    return rms_values


def wl(emg_window):
    wave_length = np.sum(np.abs(np.diff(emg_window, axis=1)), axis=1)
    return wave_length


def zc(emg_window, threshold=0):
    diff = emg_window[:, :-1, :] * emg_window[:, 1:, :]
    zero_crossings = np.sum(diff < threshold, axis=1)
    return zero_crossings


def ssc(emg_window, threshold=0):
    diff1 = emg_window[:, 1:-1, :] - emg_window[:, :-2, :]

    diff2 = emg_window[:, 1:-1, :] - emg_window[:, 2:, :]

    ssc = diff1 * diff2

    slope_sign_changes = np.sum(ssc > threshold, axis=1)

    return slope_sign_changes


def variance(emg_window):
    var = np.var(emg_window, ddof=1, axis=1)

    return var


def extracting_features(emg_window):
    emg_window = emg_window[np.newaxis, :, :]

    mav_features = mav(emg_window)
    rms_features = rms(emg_window)
    wl_features = wl(emg_window)
    zc_features = zc(emg_window)
    ssc_features = ssc(emg_window)
    variance_features = variance(emg_window)

    all_features = np.hstack(
        [
            mav_features,
            rms_features,
            wl_features,
            zc_features,
            ssc_features,
            variance_features,
        ]
    )

    return all_features


# order of features
# mav_features, rms_features, wl_features, zc_features, ssc_features, variance_features
