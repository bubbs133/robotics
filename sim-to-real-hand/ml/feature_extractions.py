import numpy as np


def mav(emg_baches):
    mav_baches = np.mean(np.abs(emg_baches), axis=1)
    return mav_baches


def rms(emg_batches):
    rms_values = np.sqrt(np.mean(emg_batches**2, axis=1))
    return rms_values


def wl(emg_batches):
    wave_length = np.sum(np.abs(np.diff(emg_batches, axis=1)), axis=1)
    return wave_length


def zc(emg_batches, threshold=0):
    diff = emg_batches[:, :-1, :] * emg_batches[:, 1:, :]
    zero_crossings = np.sum(diff < threshold, axis=1)
    return zero_crossings


def ssc(emg_batches, threshold=0):
    diff1 = emg_batches[:, 1:-1, :] - emg_batches[:, :-2, :]

    diff2 = emg_batches[:, 1:-1, :] - emg_batches[:, 2:, :]

    ssc = diff1 * diff2

    slope_sign_changes = np.sum(ssc > threshold, axis=1)

    return slope_sign_changes


def variance(emg_batches):
    var = np.var(emg_batches, ddof=1, axis=1)

    return var


# order of features
# mav_features, rms_features, wl_features, zc_features, ssc_features, variance_features