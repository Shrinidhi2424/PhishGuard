"""
utils.py — Shared utility functions for PhishGuard.
"""

import re
import logging
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)


def is_valid_url(url: str) -> bool:
    """
    Returns True if the URL can be parsed and has a valid scheme + netloc.
    Used to filter malformed entries during dataset preprocessing.
    """
    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https", "ftp"), result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Strips whitespace and ensures the URL has a scheme prefix.
    Handles the common case of users pasting URLs without 'http://'.
    """
    url = url.strip()
    if not url.startswith(("http://", "https://", "ftp://")):
        url = "http://" + url
    return url
