import os
import numpy as np
from tqdm import tqdm
from src.features import extract_multi_roi_features


def load_dataset(signal_root="data/signals", fs=30, min_len=150):

    X, y = [], []

    for label, class_id in [("real", 1), ("fake", 0)]:
        dir_path = os.path.join(signal_root, label)

        if not os.path.exists(dir_path):
            print(f"[WARN] Missing directory: {dir_path}")
            continue

        files = [f for f in os.listdir(dir_path) if f.endswith(".npy")]

        for f in tqdm(files, desc=f"Loading {label}"):
            signal_path = os.path.join(dir_path, f)

            try:
                signal = np.load(signal_path)
            except Exception:
                continue

            # Ensure shape: (T, R)
            if signal.ndim == 3:
                # Case: (T, R, C) → average channels
                signal = signal.mean(axis=-1)

            elif signal.ndim == 2:
                pass  # already (T, R)

            elif signal.ndim == 1:
                signal = signal[:, None]

            else:
                continue


            # Feature extraction (MULTI-ROI)
            features = extract_multi_roi_features(signal, fs)

            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                continue

            X.append(features)
            y.append(class_id)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    print(f"[INFO] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

    return X, y
