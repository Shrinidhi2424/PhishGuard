# 🛡️ PhishGuard
### Explainable, Lightweight ML System for Real-Time Phishing URL Detection

PhishGuard classifies URLs as **phishing** or **legitimate** and explains *why* — surfacing top risk-contributing features for every prediction using **SHAP (SHapley Additive exPlanations)** values in real time.

---

## ⚡ Quick Start

```bash
# 1. Clone repository and set up virtual environment
git clone https://github.com/Shrinidhi2424/PhishGuard.git
cd PhishGuard

python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download & prepare datasets (PhishTank verified feed + Tranco Top-1M)
python data/download_data.py

# 4. Extract 19 handcrafted features & generate stratified splits (70/15/15)
python data/preprocess.py

# 5. Train & evaluate all models (Logistic Regression, Random Forest, XGBoost)
python src/train.py

# 6. Launch interactive Streamlit web application
streamlit run app.py
```

Visit **`http://localhost:8501`** in your browser to interact with the application.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All 27 unit tests, schema contracts, integration tests, and qualitative regression checks run in ~10s.

---

## 📂 Project Structure

```
PhishGuard/
├── src/
│   ├── __init__.py
│   ├── features.py             # 19-feature deterministic URL extraction module (Layer 1)
│   ├── train.py                # Model training, validation & selection pipeline (Layer 2)
│   ├── explain.py              # Local SHAP TreeExplainer & NL explanation engine (Layer 3)
│   └── utils.py                # Shared URL validation & normalization helpers
├── data/
│   ├── download_data.py        # Automated PhishTank & Tranco data acquisition
│   ├── preprocess.py           # Feature extraction batch runner & stratified 70/15/15 split
│   ├── raw/                    # Raw feeds (gitignored)
│   └── processed/
│       ├── all_urls.csv        # 10,000 balanced URLs (5,000 phish / 5,000 legit)
│       ├── features.csv        # Full dataset with all 19 extracted features
│       ├── train.csv           # 7,000 train instances
│       ├── val.csv             # 1,500 validation instances
│       ├── test.csv            # 1,500 test instances
│       └── qualitative_test_set.csv  # 20 hand-verified challenge cases
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── best_model.joblib       # Winning model (XGBoost)
├── evaluation/
│   ├── metrics_summary.csv     # Quantitative metrics across all splits
│   ├── false_positives.csv     # Error analysis logs (FPR = 0.80%)
│   ├── false_negatives.csv     # Error analysis logs (FNR = 5.07%)
│   ├── error_analysis.py       # Misclassification analyzer
│   ├── class_balance.png       # Dataset balance visualization (150 DPI)
│   ├── feature_distributions.png # Continuous feature violin plots (150 DPI)
│   ├── boolean_feature_prevalence.png # Indicator prevalence bar chart (150 DPI)
│   ├── correlation_heatmap.png # 19-feature Pearson correlation matrix (150 DPI)
│   ├── roc_curve_comparison.png # Overlaid ROC curves (150 DPI)
│   ├── feature_importance.png  # Global tree feature importances (150 DPI)
│   ├── confusion_matrices_combined.png # Multi-model test confusion matrices (150 DPI)
│   ├── shap_summary_plot.png   # Global TreeSHAP summary plot (150 DPI)
│   └── shap_waterfall_*.png    # Local SHAP decision breakdowns (150 DPI)
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis notebook (fully executed)
│   └── 03_model_comparison.ipynb # Comparative modeling & SHAP notebook (fully executed)
├── tests/
│   ├── __init__.py
│   ├── test_features.py        # Unit tests for URL feature extraction (20 tests)
│   └── test_pipeline.py        # Integration tests + qualitative regression (7 tests)
├── app.py                      # Interactive Streamlit application (Layer 4)
├── requirements.txt            # Pinned reproducible dependencies
├── .gitignore                  # Git tracking rules
└── README.md
```

---

## 🏛️ System Architecture

| Layer | Module | Purpose | Real-Time Latency |
| :--- | :--- | :--- | :--- |
| **Layer 1: Feature Extraction** | [`src/features.py`](file:///c:/Users/Shrinidhi%20Shetty/OneDrive/Desktop/PhishGuard/src/features.py) | Extracts 19 lexical, domain-structural, and path/query features without network calls | **< 1 ms** |
| **Layer 2: Classification** | [`src/train.py`](file:///c:/Users/Shrinidhi%20Shetty/OneDrive/Desktop/PhishGuard/src/train.py) | Fast tree-based inference via XGBoost trained on 7,000 balanced instances | **< 2 ms** |
| **Layer 3: Explainability** | [`src/explain.py`](file:///c:/Users/Shrinidhi%20Shetty/OneDrive/Desktop/PhishGuard/src/explain.py) | TreeSHAP local attribution and natural language explanation generation | **< 15 ms** |
| **Layer 4: User Interface** | [`app.py`](file:///c:/Users/Shrinidhi%20Shetty/OneDrive/Desktop/PhishGuard/app.py) | Interactive Streamlit web app with risk badges, gauge, SHAP charts, and expanders | **Instant** |

> **Design Scope:** Zero-network structural analysis. PhishGuard evaluates URL structure without rendering or fetching remote HTML pages. This protects the scanner from drive-by downloads and allows instant zero-hour detection before websites are flagged on public blocklists.

---

## 📊 Quantitative Results (Test Set, $n=1,500$)

| Model | Accuracy | Macro Precision | Macro Recall | Macro-F1 | ROC AUC | False Negative Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.80% | 0.8150 | 0.8080 | 0.8069 | 0.8742 | 26.67% |
| **Random Forest** | 95.47% | 0.9570 | 0.9547 | 0.9546 | 0.9879 | 8.13% |
| **XGBoost (Winner 🏆)** | **97.07%** | **0.9715** | **0.9707** | **0.9707** | **0.9916** | **5.07%** |

- **Best Model Selected:** XGBoost (200 estimators, max depth 6, logloss).
- **False Positive Rate:** Only **0.80%** (6 false alarms out of 750 legitimate URLs).
- **Qualitative Regression Score:** **16/20 (80.0%)** on hand-verified adversarial challenge cases.

---

## 💡 Output Contract

`predict_and_explain(url)` produces a strict JSON-compatible dictionary:

```json
{
  "url": "http://paypal-secure-login.verify-account.tk/signin",
  "prediction": "phishing",
  "probability": 0.9986,
  "risk_label": "High Risk",
  "top_contributing_features": [
    {
      "feature": "has_https",
      "value": 0,
      "contribution": 4.12,
      "direction": "phishing",
      "description": "Missing HTTPS — connection is unencrypted"
    },
    {
      "feature": "has_suspicious_tld",
      "value": 1,
      "contribution": 2.85,
      "direction": "phishing",
      "description": "Suspicious TLD (.tk, .ml, .ga etc.) — heavily abused by phishers"
    }
  ],
  "explanation_text": "This URL was flagged as phishing primarily due to: missing https — connection is unencrypted and suspicious tld (.tk, .ml, .ga etc.) — heavily abused by phishers."
}
```

---

## 🛠️ Tech Stack

- **Machine Learning:** `scikit-learn==1.5.1`, `xgboost==2.1.1`
- **Explainability:** `shap==0.46.0` (TreeExplainer)
- **Feature Extraction & Parsing:** `tldextract==5.1.2`, `urllib.parse`, `re`
- **Frontend / Presentation:** `streamlit==1.38.0`
- **Visualization:** `matplotlib==3.9.2`, `seaborn==0.13.2`
- **Data Engineering:** `pandas==2.2.2`, `numpy==1.26.4`
- **Testing & Quality Assurance:** `pytest==8.3.2`, `pytest-cov==5.0.0`
