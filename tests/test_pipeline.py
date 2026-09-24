"""
test_pipeline.py — End-to-end integration tests for the PhishGuard pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
from pathlib import Path

MODELS_EXIST = (Path("models") / "best_model.joblib").exists()


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet trained — run Phase 4 first")
class TestEndToEndPipeline:

    def test_phishing_url_returns_valid_prediction(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("http://paypal-secure.verify-account.tk/login")
        assert result["prediction"] in ("phishing", "legitimate")
        assert 0.0 <= result["probability"] <= 1.0

    def test_output_schema_complete(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("https://www.google.com")
        for key in ["url", "prediction", "probability", "risk_label",
                    "top_contributing_features", "explanation_text"]:
            assert key in result, f"Missing key: {key}"

    def test_top_five_features_returned(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("https://www.github.com")
        assert len(result["top_contributing_features"]) == 5

    def test_contributions_are_numeric(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("http://192.168.1.1/login")
        for feat in result["top_contributing_features"]:
            assert isinstance(feat["contribution"], float)

    def test_risk_label_is_valid(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("https://www.amazon.com")
        assert result["risk_label"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_explanation_text_is_non_empty(self):
        from src.explain import predict_and_explain
        result = predict_and_explain("http://apple-id.verify.ml/account")
        assert len(result["explanation_text"]) > 20


@pytest.mark.skipif(not MODELS_EXIST, reason="Models not yet trained")
class TestQualitativeSet:
    """Regression test: model must correctly classify ≥80% of hand-verified URLs."""

    def test_qualitative_accuracy(self):
        import pandas as pd
        from src.explain import predict_and_explain

        qual_df = pd.read_csv("data/processed/qualitative_test_set.csv")
        correct = 0
        for _, row in qual_df.iterrows():
            try:
                result = predict_and_explain(row["url"])
                predicted = 1 if result["prediction"] == "phishing" else 0
                if predicted == row["label"]:
                    correct += 1
            except Exception:
                pass

        accuracy = correct / len(qual_df)
        assert accuracy >= 0.80, (
            f"Qualitative accuracy {accuracy:.2f} below threshold 0.80 "
            f"({correct}/{len(qual_df)} correct)"
        )
