from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib
import os

def get_models(y_train):
    return {
        "Logistic": LogisticRegression(max_iter=1000),
        "SVM-RBF": SVC(kernel="rbf", probability=True),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            eval_metric="logloss"
        )
    }


def evaluate_models(X, y, save_dir="results/models"):
    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = {}
    models = get_models(y_train)

    for name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", model)
        ])

        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, probs)
        }

        # ✅ SAVE TRAINED MODEL
        joblib.dump(pipe, os.path.join(save_dir, f"{name}.pkl"))

    return results
