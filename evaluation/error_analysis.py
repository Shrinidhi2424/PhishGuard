"""
error_analysis.py — Analyzes model misclassifications on the test set.

Usage:
    python evaluation/error_analysis.py

Outputs:
    evaluation/false_positives.csv
    evaluation/false_negatives.csv
    Printed SHAP-based analysis of sample misclassifications
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.explain import predict_and_explain, FEATURE_NAMES

PROCESSED_DIR = Path("data/processed")
EVAL_DIR = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)


def main():
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        print("Error: models/best_model.joblib not found. Run src/train.py first.")
        sys.exit(1)

    model = joblib.load(model_path)
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    X_test = test_df[FEATURE_NAMES].values
    y_true = test_df["label"].values
    y_pred = model.predict(X_test)

    test_df["y_pred"] = y_pred
    false_positives = test_df[(test_df["label"] == 0) & (test_df["y_pred"] == 1)].copy()
    false_negatives = test_df[(test_df["label"] == 1) & (test_df["y_pred"] == 0)].copy()

    legit_count = len(test_df[test_df['label'] == 0])
    phish_count = len(test_df[test_df['label'] == 1])
    fpr = len(false_positives) / legit_count if legit_count > 0 else 0.0
    fnr = len(false_negatives) / phish_count if phish_count > 0 else 0.0

    print("=" * 65)
    print("PHISHGUARD TEST SET ERROR ANALYSIS")
    print("=" * 65)
    print(f"Total Test Set URLs: {len(test_df):,}")
    print(f"Legitimate URLs:    {legit_count:,}")
    print(f"Phishing URLs:      {phish_count:,}")
    print(f"False Positives (legit -> flagged phishing): {len(false_positives)} (FPR = {fpr:.4f} or {fpr*100:.2f}%)")
    print(f"False Negatives (phishing -> passed as legit): {len(false_negatives)} (FNR = {fnr:.4f} or {fnr*100:.2f}%)")

    print("\n=== False Positives: Mean Feature Values vs All Legit ===")
    all_legit_means = test_df[test_df["label"] == 0][FEATURE_NAMES].mean()
    fp_means = false_positives[FEATURE_NAMES].mean()
    fp_comparison = pd.DataFrame({
        "FP Mean": fp_means,
        "All Legit Mean": all_legit_means,
        "Difference": fp_means - all_legit_means
    }).sort_values(by="Difference", ascending=False)
    print(fp_comparison.head(8).to_string())

    print("\n=== False Negatives: Mean Feature Values vs All Phishing ===")
    all_phish_means = test_df[test_df["label"] == 1][FEATURE_NAMES].mean()
    fn_means = false_negatives[FEATURE_NAMES].mean()
    fn_comparison = pd.DataFrame({
        "FN Mean": fn_means,
        "All Phish Mean": all_phish_means,
        "Difference": fn_means - all_phish_means
    }).sort_values(by="Difference", ascending=True)
    print(fn_comparison.head(8).to_string())

    false_positives.to_csv(EVAL_DIR / "false_positives.csv", index=False)
    false_negatives.to_csv(EVAL_DIR / "false_negatives.csv", index=False)
    print(f"\nSaved {EVAL_DIR / 'false_positives.csv'}")
    print(f"Saved {EVAL_DIR / 'false_negatives.csv'}")

    # Qualitative analysis of sample FPs
    print("\n" + "=" * 65)
    print("SAMPLE FALSE POSITIVES (SHAP Explanation)")
    print("=" * 65)
    for i, (_, row) in enumerate(false_positives.head(5).iterrows()):
        url = row.get("url", "N/A")
        r = predict_and_explain(url, model=model)
        flags = [f"{f['feature']} ({f['contribution']:+.2f})" for f in r["top_contributing_features"][:3]]
        print(f"\n[FP #{i+1}] {url[:70]}")
        print(f"  Confidence: {r['probability']:.3f} | Top risk drivers: {', '.join(flags)}")
        print(f"  Explanation: {r['explanation_text']}")

    # Qualitative analysis of sample FNs
    print("\n" + "=" * 65)
    print("SAMPLE FALSE NEGATIVES (SHAP Explanation)")
    print("=" * 65)
    for i, (_, row) in enumerate(false_negatives.head(5).iterrows()):
        url = row.get("url", "N/A")
        r = predict_and_explain(url, model=model)
        flags = [f"{f['feature']} ({f['contribution']:+.2f})" for f in r["top_contributing_features"][:3]]
        print(f"\n[FN #{i+1}] {url[:70]}")
        print(f"  Confidence: {r['probability']:.3f} | Features lowering risk: {', '.join(flags)}")
        print(f"  Explanation: {r['explanation_text']}")

    # Qualitative test set (20 hand-verified URLs)
    qual_path = PROCESSED_DIR / "qualitative_test_set.csv"
    if qual_path.exists():
        qual_df = pd.read_csv(qual_path)
        print("\n" + "=" * 65)
        print("QUALITATIVE TEST SET (20 Hand-Verified Challenge Cases)")
        print("=" * 65)
        correct_count = 0
        for _, row in qual_df.iterrows():
            r = predict_and_explain(row["url"], model=model)
            pred = 1 if r["prediction"] == "phishing" else 0
            is_correct = (pred == row["label"])
            if is_correct:
                correct_count += 1
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"{status} [{row['category']:<16}] {row['url'][:50]:<50} -> {r['prediction']:<10} (p={r['probability']:.2f})")

        acc = correct_count / len(qual_df)
        print(f"\nQualitative Set Accuracy: {correct_count}/{len(qual_df)} ({acc*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
