import joblib
import numpy as np
from src.dataset_dfd import load_dfd_dataset

MODELS = ["Logistic", "SVM-RBF", "XGBoost"]

X_test, names = load_dfd_dataset()

print(f"[INFO] Loaded {len(X_test)} DFD samples")

for model_name in MODELS:
    model = joblib.load(f"results/models/{model_name}.pkl")

    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} on DFD:")
    print(f"  Mean fake prob : {probs.mean():.3f}")
    print(f"  Std  fake prob : {probs.std():.3f}")
    print(f"  Max  fake prob : {probs.max():.3f}")
