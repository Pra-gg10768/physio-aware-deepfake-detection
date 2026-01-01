import os
import numpy as np
from tqdm import tqdm
from src.features import extract_multi_roi_features

def load_dfd_dataset(signal_root="data/signals_dfd_test", fs=30):
    X, names = [], []

    data_dir = os.path.join(signal_root, "unknown")

    for f in tqdm(os.listdir(data_dir), desc="Loading DFD test"):
        signal = np.load(os.path.join(data_dir, f))

        if signal.ndim == 3:
            signal = signal.mean(axis=2)  # RGB → 1 channel

        if signal.shape[0] < 100:
            continue

        feats = extract_multi_roi_features(signal, fs)
        X.append(feats)
        names.append(f)

    return np.array(X), names
