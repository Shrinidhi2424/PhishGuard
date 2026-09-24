"""
download_data.py — Acquires, cleans, and splits the PhishGuard dataset.

Usage:
    python data/download_data.py

Outputs:
    data/raw/phishtank_urls.csv
    data/raw/tranco_legitimate.csv
    data/processed/all_urls.csv
"""

import os
import io
import zipfile
import requests
import pandas as pd
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

PHISHTANK_URLS = [
    "http://data.phishtank.com/data/online-valid.csv",
    "https://data.phishtank.com/data/online-valid.csv",
    # OpenPhish community feed as reliable fallback if PhishTank rate limits or blocks
    "https://openphish.com/feed.txt",
]
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
MAX_PER_CLASS = 5000


def download_phishtank():
    """Download verified phishing URLs from PhishTank or fallback sources."""
    logger.info("Downloading PhishTank feed...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhishGuardResearch/1.0"}
    
    # Try primary PhishTank URLs
    for url in PHISHTANK_URLS[:2]:
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            if resp.status_code == 200 and len(resp.text) > 1000:
                df = pd.read_csv(io.StringIO(resp.text))
                if "url" in df.columns:
                    urls = df["url"].dropna().unique().tolist()
                    logger.info(f"Downloaded {len(urls)} phishing URLs from PhishTank.")
                    return urls
        except Exception as e:
            logger.warning(f"PhishTank download from {url} failed: {e}")

    # Fallback to OpenPhish if PhishTank throttles/blocks
    try:
        logger.info("Attempting fallback phishing feed (OpenPhish)...")
        resp = requests.get("https://openphish.com/feed.txt", headers=headers, timeout=60)
        if resp.status_code == 200 and len(resp.text) > 500:
            urls = [line.strip() for line in resp.text.splitlines() if line.strip()]
            logger.info(f"Downloaded {len(urls)} phishing URLs from OpenPhish fallback.")
            return urls
    except Exception as e:
        logger.warning(f"OpenPhish download failed: {e}")

    # Local fallback file check
    fallback = RAW_DIR / "phishtank_fallback.txt"
    if fallback.exists():
        logger.info("Using local fallback file.")
        return [line.strip() for line in open(fallback, encoding="utf-8").read().splitlines() if line.strip()]

    raise RuntimeError("Phishing URL download failed and no fallback available.")


def download_tranco():
    """Download Tranco Top-1M list and convert to URLs."""
    logger.info("Downloading Tranco Top-1M list...")
    try:
        resp = requests.get(TRANCO_URL, timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open("top-1m.csv") as f:
                df = pd.read_csv(f, names=["rank", "domain"])
        domains = df["domain"].dropna().tolist()[:MAX_PER_CLASS * 2]
        # In the wild, legitimate URLs include www, common subdomains, and diverse paths.
        # Generating realistic URL variations prevents synthetic distribution shift (e.g. assuming any path > 10 chars is phishing).
        subdomains = ["", "www.", "", "www.", "", "docs.", "blog.", "support.", "en."]
        paths = [
            "", "", "",
            "/about", "/contact", "/home", "/login", "/news", "/help", "/faq",
            "/privacy-policy", "/terms-and-conditions", "/products/item/10293",
            "/dp/B08N5WRWNW", "/in/profile-name", "/r/technology",
            "/conditions/diabetes", "/docs/forms/d/e/viewform",
            "/watch?v=dQw4w9WgXcQ", "/search?q=cybersecurity+research",
            "/article/2026/05/global-update", "/category/news/today"
        ]
        urls = []
        for i, d in enumerate(domains):
            sub = subdomains[i % len(subdomains)]
            p = paths[i % len(paths)]
            urls.append(f"https://{sub}{d}{p}")
        logger.info(f"Loaded {len(urls)} legitimate URLs from Tranco.")
        return urls
    except Exception as e:
        logger.warning(f"Tranco download failed: {e}")
        raise


def clean_urls(urls: list, label: int) -> pd.DataFrame:
    """Validate, deduplicate, and label a list of URLs."""
    from src.utils import is_valid_url, normalize_url

    cleaned = []
    seen = set()
    for url in urls:
        try:
            url = normalize_url(str(url).strip())
            if url in seen:
                continue
            if is_valid_url(url):
                cleaned.append({"url": url, "label": label})
                seen.add(url)
        except Exception:
            continue
    logger.info(f"Cleaned {len(cleaned)} valid unique URLs for label={label}.")
    return pd.DataFrame(cleaned)


def main():
    phishing_urls = download_phishtank()
    legitimate_urls = download_tranco()

    phishing_df = clean_urls(phishing_urls, label=1)
    legitimate_df = clean_urls(legitimate_urls, label=0)

    # Balance classes
    min_count = min(len(phishing_df), len(legitimate_df), MAX_PER_CLASS)
    phishing_df = phishing_df.sample(n=min_count, random_state=42)
    legitimate_df = legitimate_df.sample(n=min_count, random_state=42)

    combined = pd.concat([phishing_df, legitimate_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    phishing_df.to_csv(RAW_DIR / "phishtank_urls.csv", index=False)
    legitimate_df.to_csv(RAW_DIR / "tranco_legitimate.csv", index=False)
    combined.to_csv(PROCESSED_DIR / "all_urls.csv", index=False)

    logger.info(f"Saved {len(combined)} total URLs to data/processed/all_urls.csv")
    logger.info(f"Class distribution:\n{combined['label'].value_counts()}")


if __name__ == "__main__":
    main()
