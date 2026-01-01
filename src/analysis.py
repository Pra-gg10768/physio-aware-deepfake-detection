import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.stats import ttest_ind

# ===============================
# FFT COMPUTATION
# ===============================

def compute_fft(signal, fs=30):
    """
    Compute normalized FFT magnitude of rPPG signal.
    """
    fft_vals = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1/fs)
    return freqs, fft_vals


# ===============================
# PEAK SHARPNESS METRIC
# ===============================

def peak_sharpness(freqs, spectrum, hr_range=(0.7, 3.0)):
    """
    Measures how sharp the dominant heartbeat peak is.
    """
    idx = np.where((freqs >= hr_range[0]) & (freqs <= hr_range[1]))[0]
    peak_idx = idx[np.argmax(spectrum[idx])]
    peak_val = spectrum[peak_idx]

    neighborhood = spectrum[max(0, peak_idx-2): peak_idx+3]
    return peak_val / (np.mean(neighborhood) + 1e-6)


# ===============================
# LOAD SIGNALS
# ===============================

def load_signals(signal_dir):
    """
    Load all rPPG signals from directory.
    """
    signals = []
    for file in os.listdir(signal_dir):
        if file.endswith(".npy"):
            sig = np.load(os.path.join(signal_dir, file))
            signals.append(sig)
    return signals


# ===============================
# AGGREGATE FFT
# ===============================

def average_fft(signals, fs=30):
    """
    Compute mean FFT spectrum over many videos.
    """
    specs = []
    for sig in signals:
        freqs, spec = compute_fft(sig, fs)
        specs.append(spec)

    specs = np.vstack(specs)
    return freqs, np.mean(specs, axis=0)


# ===============================
# STATISTICAL ANALYSIS
# ===============================

def compute_peak_statistics(real_signals, fake_signals, fs=30):
    """
    Compute peak sharpness statistics and t-test.
    """
    real_peaks = []
    fake_peaks = []

    for sig in real_signals:
        freqs, spec = compute_fft(sig, fs)
        real_peaks.append(peak_sharpness(freqs, spec))

    for sig in fake_signals:
        freqs, spec = compute_fft(sig, fs)
        fake_peaks.append(peak_sharpness(freqs, spec))

    t_stat, p_val = ttest_ind(real_peaks, fake_peaks)

    return {
        "real_mean": np.mean(real_peaks),
        "fake_mean": np.mean(fake_peaks),
        "t_stat": t_stat,
        "p_value": p_val
    }


# ===============================
# VISUALIZATION
# ===============================

def plot_average_fft(freqs, real_fft, fake_fft, save_path=None):
    """
    Plot averaged FFT spectra.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, real_fft, label="Real", linewidth=2)
    plt.plot(freqs, fake_fft, label="Fake", linestyle="--")

    plt.xlim(0.5, 3.5)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Average rPPG Frequency Spectrum")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ===============================
# MAIN ANALYSIS PIPELINE
# ===============================

def run_analysis(real_dir, fake_dir, results_dir):
    """
    Full FFT + statistical analysis pipeline.
    """
    os.makedirs(results_dir, exist_ok=True)

    real_signals = load_signals(real_dir)
    fake_signals = load_signals(fake_dir)

    freqs, real_fft = average_fft(real_signals)
    _, fake_fft = average_fft(fake_signals)

    stats = compute_peak_statistics(real_signals, fake_signals)

    print("\n=== Peak Sharpness Statistics ===")
    print(f"Real mean: {stats['real_mean']:.2f}")
    print(f"Fake mean: {stats['fake_mean']:.2f}")
    print(f"T-statistic: {stats['t_stat']:.2f}")
    print(f"P-value: {stats['p_value']:.4e}")

    plot_average_fft(
        freqs,
        real_fft,
        fake_fft,
        save_path=os.path.join(results_dir, "average_fft.png")
    )

    return stats
