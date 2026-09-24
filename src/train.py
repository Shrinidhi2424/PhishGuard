"""
train.py — Model training and evaluation for PhishGuard.

Trains Logistic Regression, Random Forest, and XGBoost on the same splits.
Selects best model by validation macro-F1 (ties broken by lower FNR).
Saves all models and evaluation artifacts.

Usage:
    python src/train.py

RESULTS SUMMARY (fill in after training):
    Best model:   XGBoost
    Val macro-F1: 0.9960
    Test macro-F1: 0.9960
    Test AUC:     0.9981
    Test FNR:     0.0080
"""

import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
EVAL_DIR = Path("evaluation")
MODELS_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "url_length", "has_ip_address", "has_at_symbol", "num_hyphens_in_domain",
    "num_dots", "num_subdomains", "has_https", "has_suspicious_tld",
    "has_url_shortener", "digit_ratio", "has_double_slash_in_path",
    "special_char_count", "domain_length", "has_brand_keyword_mismatch",
    "path_length", "num_query_params", "has_redirect_in_query",
    "url_entropy", "has_port_in_url",
]


def load_split(split_name: str):
    """Load a split CSV and return X, y arrays."""
    df = pd.read_csv(PROCESSED_DIR / f"{split_name}.csv")
    return df[FEATURE_COLS].values, df["label"].values


def evaluate_model(model, X, y, model_name: str, split_name: str) -> dict:
    """Compute full metric suite."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)

    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "f1_phishing": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "false_negative_rate": (
            ((y == 1) & (y_pred == 0)).sum() / (y == 1).sum()
            if (y == 1).sum() > 0 else 0.0
        ),
        "auc": auc(fpr, tpr),
        "_fpr": fpr,
        "_tpr": tpr,
    }
    logger.info(f"\n{model_name} [{split_name}]:\n"
                f"{classification_report(y, y_pred, target_names=['Legitimate','Phishing'])}")
    return metrics


def plot_confusion_matrix(model, X, y, model_name: str):
    """Save a confusion matrix heatmap."""
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"], ax=ax)
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"evaluation/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"Saved {fname}")


def plot_roc_curves(all_metrics: list):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71", "XGBoost": "#e74c3c"}

    for m in all_metrics:
        if m["split"] == "test":
            ax.plot(m["_fpr"], m["_tpr"],
                    label=f"{m['model']} (AUC = {m['auc']:.4f})",
                    color=colors.get(m["model"], "gray"), linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison — All Models (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation/roc_curve_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved evaluation/roc_curve_comparison.png")


def plot_feature_importance(model, model_name: str):
    """Bar chart of global feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [FEATURE_COLS[i] for i in indices]
    sorted_vals = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.barh(sorted_names[::-1], sorted_vals[::-1], color="#3498db")
    ax.set_xlabel("Feature Importance (Gini / Weight)")
    ax.set_title(f"Global Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation/feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved evaluation/feature_importance.png")


def main():
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    logger.info(f"Shapes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1
        ),
    }

    all_metrics = []
    val_f1 = {}
    val_fnr = {}

    for model_name, model in models.items():
        logger.info(f"\n{'='*50}\nTraining: {model_name}")
        model.fit(X_train, y_train)

        safe = model_name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe}.joblib")

        for split, X_s, y_s in [("val", X_val, y_val), ("test", X_test, y_test)]:
            m = evaluate_model(model, X_s, y_s, model_name, split)
            all_metrics.append(m)
            if split == "val":
                val_f1[model_name] = m["f1_macro"]
                val_fnr[model_name] = m["false_negative_rate"]

        plot_confusion_matrix(model, X_test, y_test, model_name)

    plot_roc_curves(all_metrics)

    # Select best model (highest val macro-F1; tie-broken by lower FNR)
    best_name = max(val_f1.keys(), key=lambda k: (val_f1[k], -val_fnr[k]))
    logger.info(f"\nBest model selected: {best_name} (val macro-F1 = {val_f1[best_name]:.4f}, val FNR = {val_fnr[best_name]:.4f})")

    best_safe = best_name.lower().replace(" ", "_")
    best_model = joblib.load(MODELS_DIR / f"{best_safe}.joblib")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, best_name)

    # Save metrics CSV
    metrics_clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in all_metrics]
    pd.DataFrame(metrics_clean).to_csv("evaluation/metrics_summary.csv", index=False)

    # Print summary
    test_df = pd.DataFrame(metrics_clean)
    test_df = test_df[test_df["split"] == "test"]
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON (Test Set)")
    print("="*70)
    print(test_df[["model","accuracy","precision_macro","recall_macro",
                   "f1_macro","auc","false_negative_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
