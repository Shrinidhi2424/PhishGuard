"""
preprocess.py — Feature extraction + train/val/test split generation.

Usage (after Phase 2):
    python data/preprocess.py

Outputs:
    data/processed/features.csv
    data/processed/train.csv  (70%)
    data/processed/val.csv    (15%)
    data/processed/test.csv   (15%)
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")


def main():
    input_file = PROCESSED_DIR / "all_urls.csv"
    if not input_file.exists():
        logger.error(f"{input_file} not found. Run data/download_data.py first.")
        sys.exit(1)

    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} URLs for feature extraction.")

    feature_rows = []
    failed = 0
    for i, row in df.iterrows():
        try:
            features = extract_features(row["url"])
            features["url"] = row["url"]
            features["label"] = row["label"]
            feature_rows.append(features)
        except Exception as e:
            logger.warning(f"Failed on {row['url']}: {e}")
            failed += 1
        if (i + 1) % 500 == 0:
            logger.info(f"Progress: {i+1}/{len(df)} URLs processed.")

    logger.info(f"Done: {len(feature_rows)} success, {failed} failed.")
    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(PROCESSED_DIR / "features.csv", index=False)

    # Stratified 70/15/15 split
    X = features_df.drop(columns=["url", "label"])
    y = features_df["label"]
    urls = features_df["url"]

    X_train, X_temp, y_train, y_temp, u_train, u_temp = train_test_split(
        X, y, urls, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test, u_val, u_test = train_test_split(
        X_temp, y_temp, u_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    for name, X_s, y_s, u_s in [
        ("train", X_train, y_train, u_train),
        ("val", X_val, y_val, u_val),
        ("test", X_test, y_test, u_test),
    ]:
        out = X_s.copy()
        out["label"] = y_s.values
        out["url"] = u_s.values
        out.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        logger.info(f"Saved {name}.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
