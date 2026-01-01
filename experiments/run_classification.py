from src.dataset import load_dataset
from src.classifier import evaluate_models

X, y = load_dataset()
results = evaluate_models(X, y)

print("\nModel Comparison:")
for model, metrics in results.items():
    print(f"{model:12s} | Acc: {metrics['accuracy']:.3f} | AUC: {metrics['roc_auc']:.3f}")
