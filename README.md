# PhishGuard 🛡️
### Explainable, Lightweight ML System for Real-Time Phishing URL Detection

PhishGuard is a lightweight, real-time phishing URL detection system with feature-level risk justification. It evaluates URL structural patterns, domain characteristics, and lexical signals without executing or rendering remote pages.

## Key Features
- **Real-Time Inference**: Evaluates URLs in under 20ms using lexical and structural features.
- **Explainable Decisions**: SHAP-powered risk contributions explaining why a URL was marked as phishing or legitimate.
- **Interactive UI**: Clean Streamlit dashboard for URL scanning and feature importance visualization.

## Stack
- Python 3.10+
- scikit-learn, XGBoost
- SHAP
- Streamlit
- pandas, numpy, tldextract

## Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import sklearn, xgboost, shap, streamlit, tldextract, pandas; print('All imports OK')"
   ```
