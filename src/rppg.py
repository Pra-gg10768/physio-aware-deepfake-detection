import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt

# -------------------------------------------------
# Core rPPG utilities
# -------------------------------------------------

def extract_green_signal(frames):
    """
    frames: (T, H, W, 3)
    """
    return np.mean(frames[:, :, :, 1], axis=(1, 2))


def bandpass_filter(signal, fs=30, low=0.7, high=3.0, order=3):
    if len(signal) < fs:
        return signal

    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def fix_length(signal, target_len=300):
    if len(signal) >= target_len:
        return signal[:target_len]

    pad = target_len - len(signal)
    return np.pad(signal, (0, pad), mode="edge")


def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-6)


# -------------------------------------------------
# Multi-ROI helper
# -------------------------------------------------

ROI_NAMES = ["forehead", "left_cheek", "right_cheek", "nose", "chin"]


def load_roi_frames(video_path, resize_hw=(72, 72)):
    """
    Loads all frames from a single ROI folder
    """
    frame_files = sorted([
        f for f in os.listdir(video_path)
        if f.endswith(".png")
    ])

    frames = []
    for f in frame_files:
        img = cv2.imread(os.path.join(video_path, f))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, resize_hw)
        frames.append(img)

    if len(frames) < 30:
        return None

    return np.stack(frames)


# -------------------------------------------------
# Main extraction pipeline (SINGLE + MULTI ROI)
# -------------------------------------------------

def extract_all_signals(
    roi_root="data/rois",
    out_root="data/signals",
    roi_region="forehead",   # backward compatible
    fs=30,
    target_len=300,
    resize_hw=(72, 72),
    multi_roi=False
):
    """
    If multi_roi=False → old behavior (only forehead)
    If multi_roi=True  → saves one file per ROI per video
    """

    for label in ["real", "fake"]:

        if multi_roi:
            roi_dirs = {
                roi: os.path.join(roi_root, label, roi)
                for roi in ROI_NAMES
                if os.path.exists(os.path.join(roi_root, label, roi))
            }
        else:
            roi_dirs = {
                roi_region: os.path.join(roi_root, label, roi_region)
            }

        out_dir = os.path.join(out_root, label)
        os.makedirs(out_dir, exist_ok=True)

        video_ids = sorted(os.listdir(next(iter(roi_dirs.values()))))

        print(f"\nExtracting rPPG for {label} videos...")
        print(f"Found {len(video_ids)} video folders")

        for vid in tqdm(video_ids):

            roi_signals = {}

            for roi_name, roi_base in roi_dirs.items():
                roi_path = os.path.join(roi_base, vid)

                if not os.path.isdir(roi_path):
                    continue

                frames = load_roi_frames(roi_path, resize_hw)
                if frames is None:
                    continue

                signal = extract_green_signal(frames)
                signal = bandpass_filter(signal, fs)
                signal = fix_length(signal, target_len)
                signal = normalize(signal)

                roi_signals[roi_name] = signal

            # ❌ skip broken samples
            if len(roi_signals) == 0:
                continue

            # ✅ Save
            if multi_roi:
                np.save(
                    os.path.join(out_dir, f"{vid}.npy"),
                    roi_signals
                )
            else:
                # old behavior
                roi_name = list(roi_signals.keys())[0]
                np.save(
                    os.path.join(out_dir, f"{vid}.npy"),
                    roi_signals[roi_name]
                )
