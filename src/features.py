import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from itertools import combinations

# =================================================
# Frequency utilities
# =================================================

def band_energy(freqs, psd, f_low, f_high):
    mask = (freqs >= f_low) & (freqs <= f_high)
    return np.sum(psd[mask])


def spectral_entropy(psd):
    psd = psd / (np.sum(psd) + 1e-8)
    return entropy(psd + 1e-8)


def spectral_flatness(psd):
    return np.exp(np.mean(np.log(psd + 1e-8))) / (np.mean(psd) + 1e-8)


def hr_stability(freqs, psd):
    peak_idxs = np.argsort(psd)[-3:]
    return np.std(freqs[peak_idxs])


# =================================================
# Core rPPG features (per ROI)
# =================================================

def extract_core_features(signal, fs=30):
    if len(signal) < 30 or np.std(signal) < 1e-5:
        return np.zeros(7)

    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    psd = psd / (np.sum(psd) + 1e-8)

    hr_band = (0.7, 3.0)
    low_band = (0.7, 1.5)
    high_band = (1.5, 3.0)

    hr_energy = band_energy(freqs, psd, *hr_band)
    low_energy = band_energy(freqs, psd, *low_band)
    high_energy = band_energy(freqs, psd, *high_band)

    peak_idx = np.argmax(psd)
    peak_freq = freqs[peak_idx]
    peak_val = psd[peak_idx]

    peak_sharpness = peak_val / (np.mean(psd) + 1e-8)

    return np.array([
        peak_freq,                         # HR location
        peak_sharpness,                   # Peak consistency
        hr_energy,                        # HR band energy
        low_energy / (high_energy + 1e-8),# Multi-band ratio
        spectral_entropy(psd),            # PSD entropy
        hr_stability(freqs, psd),         # HR stability
        spectral_flatness(psd)            # Spectral flatness
    ])


# =================================================
# Window-based temporal features
# =================================================

def temporal_window_features(signal, fs=30, window_sec=5):
    win = fs * window_sec
    if len(signal) < win:
        return np.zeros(2)

    stds = []
    for i in range(0, len(signal) - win, win):
        stds.append(np.std(signal[i:i + win]))

    return np.array([
        np.mean(stds),
        np.std(stds)
    ])


# =================================================
# Signal quality features
# =================================================

def signal_quality_features(signal):
    if len(signal) < 10:
        return np.zeros(2)

    diff = np.diff(signal)
    return np.array([
        np.mean(np.abs(diff)),   # jitter
        np.std(diff)             # instability
    ])


# =================================================
# Multi-ROI Feature Fusion (MAIN ENTRY)
# =================================================

def extract_multi_roi_features(signal, fs=30):
    """
    signal: (T, R)
    ROI order MUST be fixed across dataset
    """

    if signal.ndim == 1:
        signal = signal[:, None]

    T, R = signal.shape
    features = []

    # ---- ROI-wise features ----
    for r in range(R):
        sig = signal[:, r]

        features.extend(extract_core_features(sig, fs))
        features.extend(temporal_window_features(sig, fs))
        features.extend(signal_quality_features(sig))

    # ---- Cross-ROI correlation ----
    if R > 1:
        corr = np.corrcoef(signal.T)
        upper = corr[np.triu_indices(R, k=1)]
        features.extend([
            np.nanmean(upper),
            np.nanstd(upper)
        ])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)
