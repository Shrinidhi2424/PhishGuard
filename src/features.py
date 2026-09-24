"""
features.py — URL Feature Extraction Module for PhishGuard.

Extracts 19 lexical, structural, and domain-based features from a raw URL string.
All features are deterministic, require no network calls, and are independently testable.

Feature Categories:
    A. Lexical / URL-string features
    B. Domain / structural features
    C. Path / query features

Each feature includes its security rationale in the docstring.
"""

import re
import math
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any

import tldextract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    # Freenom free TLDs — heavily abused (APWG eCrime reports)
    "tk", "ml", "ga", "cf", "gq",
    # Cheap gTLDs with high phishing prevalence
    "xyz", "pw", "top", "club", "work",
    "site", "online", "tech", "live",
    "info", "biz",
}

URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "adf.ly", "short.link", "rebrand.ly",
    "cutt.ly", "shorturl.at", "tiny.cc", "rb.gy", "clck.ru",
}

BRAND_KEYWORDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "facebook",
    "instagram", "twitter", "netflix", "bank", "chase", "wellsfargo",
    "citibank", "hsbc", "linkedin", "dropbox", "icloud", "outlook",
    "office365",
]

SUSPICIOUS_KEYWORDS = [
    "signin", "login", "verify", "secure", "account",
    "update", "support", "confirm",
]

# ---------------------------------------------------------------------------
# A. Lexical / URL-string features
# ---------------------------------------------------------------------------

def get_url_length(url: str) -> int:
    """
    Total character length of the URL string.
    RATIONALE: Phishing URLs average ~20 chars longer than legitimate URLs.
    Extra length comes from additional subdomains, path segments, and query
    parameters used to mimic legitimate site structure or obfuscate the true destination.
    """
    return len(url)


def has_ip_address(url: str) -> bool:
    """
    True if the URL hostname is a raw IPv4 or IPv6 address.
    RATIONALE: Legitimate services almost never expose raw IP addresses in
    user-facing URLs. Using an IP avoids domain registration costs and
    WHOIS scrutiny — a strong phishing signal.
    """
    hostname = urlparse(url).hostname or ""
    ipv4_pattern = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
    ipv6_pattern = re.compile(r"^\[?[0-9a-fA-F:]+\]?$")
    return bool(ipv4_pattern.match(hostname) or ipv6_pattern.match(hostname))


def has_at_symbol(url: str) -> bool:
    """
    True if '@' appears in the netloc portion of the URL.
    RATIONALE: Classic obfuscation trick. Browsers treat everything before '@'
    as credentials, so 'paypal.com@attacker.ru' resolves to attacker.ru.
    The paypal.com part is silently discarded.
    """
    return "@" in urlparse(url).netloc


def get_num_hyphens_in_domain(url: str) -> int:
    """
    Count of hyphens in the registered domain name (not path, not subdomains).
    RATIONALE: Phishing domains string brand names with security words using hyphens:
    'paypal-secure-login.com', 'apple-id-verify.com'. Legitimate brand domains
    rarely contain hyphens in the main domain.
    """
    extracted = tldextract.extract(url)
    return (extracted.domain or "").count("-")


def get_num_dots(url: str) -> int:
    """
    Total dot count in the full URL string.
    RATIONALE: High dot count correlates with deep subdomain nesting — an
    attacker technique: 'paypal.login.verify.attacker.com' looks like PayPal
    until the registered domain is examined.
    """
    return url.count(".")


def get_num_subdomains(url: str) -> int:
    """
    Number of subdomain segments (split by dot in the subdomain field).
    RATIONALE: 3+ subdomains is a common brand impersonation pattern.
    tldextract correctly separates subdomain from registered domain, handling
    cases like 'paypal.com.attacker.ru' where 'paypal.com' is in the subdomain.
    """
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    if not subdomain:
        return 0
    return len(subdomain.split("."))


def has_https(url: str) -> bool:
    """
    True if the URL scheme is HTTPS.
    RATIONALE: HTTP absence is a red flag (no encryption, no certificate validation).
    IMPORTANT NUANCE: HTTPS does NOT imply legitimacy — attackers obtain free
    TLS certificates (e.g., Let's Encrypt). This feature is informative only
    in combination with others; document this nuance in the report.
    """
    return urlparse(url).scheme == "https"


def has_suspicious_tld(url: str) -> bool:
    """
    True if the TLD is in the SUSPICIOUS_TLDS set.
    RATIONALE: Free-registration TLDs (.tk, .ml, .ga, .cf, .gq — Freenom)
    are heavily abused because they cost nothing and require no identity verification.
    Per APWG eCrime reports, these TLDs consistently appear in top-10 phishing lists.
    Source: APWG Phishing Activity Trends Report, Spamhaus TLD reputation data.
    """
    extracted = tldextract.extract(url)
    suffix = (extracted.suffix or "").lower()
    tld = suffix.split(".")[-1]
    return tld in SUSPICIOUS_TLDS


def has_url_shortener(url: str) -> bool:
    """
    True if the hostname matches a known URL shortener service.
    RATIONALE: URL shorteners hide the actual destination from users and from
    link-based blocklist detection. Phishing links distributed via email,
    SMS, or social media commonly use shorteners to evade detection.
    """
    hostname = (urlparse(url).hostname or "").lower().lstrip("www.")
    return hostname in URL_SHORTENERS


def get_digit_ratio(url: str) -> float:
    """
    Proportion of characters in the URL that are digits (0.0–1.0).
    RATIONALE: DGA-generated or randomly-created phishing domains and paths
    have higher digit ratios than human-readable legitimate URLs.
    Also catches IP-address-style obfuscation in subdomain strings.
    """
    if len(url) == 0:
        return 0.0
    return sum(1 for c in url if c.isdigit()) / len(url)


def has_double_slash_in_path(url: str) -> bool:
    """
    True if '//' appears in the URL path (after the scheme '://').
    RATIONALE: Double slashes in the path (not the scheme) have historically
    been used to trick URL parsers about the actual destination, e.g.,
    'http://legitimate.com//http://phishing.com'.
    """
    return "//" in urlparse(url).path


def get_special_char_count(url: str) -> int:
    """
    Count of special characters (%, =, &, !, ~, #) in the URL.
    RATIONALE: Excessive special characters indicate URL encoding obfuscation
    (%20, %2F to hide keywords from simple string matching) or deep query
    parameter nesting typical of phishing kit-generated pages.
    """
    return sum(1 for c in url if c in set("%=&!~#^*"))


# ---------------------------------------------------------------------------
# B. Domain / structural features
# ---------------------------------------------------------------------------

def get_domain_length(url: str) -> int:
    """
    Character length of the registered domain name (excluding TLD and subdomains).
    RATIONALE: Very long domains are often DGA-generated; very short domains
    are also anomalous. Legitimate brand domains typically fall in 5–15 characters.
    """
    extracted = tldextract.extract(url)
    return len(extracted.domain or "")


def has_brand_keyword_mismatch(url: str) -> bool:
    """
    True if a known brand keyword appears in the URL but the registered domain
    does NOT match that brand.
    RATIONALE: Strongest handcrafted signal for brand impersonation.
    'paypal.secure-login.attacker.com' contains 'paypal' in the subdomain,
    but the registered domain is 'attacker.com' — the mismatch exposes the attack.
    tldextract is essential here to correctly identify the registered domain.
    """
    extracted = tldextract.extract(url)
    registered_domain = (extracted.domain or "").lower()
    url_lower = url.lower()

    for brand in BRAND_KEYWORDS:
        if brand in url_lower and brand not in registered_domain:
            return True
    return False


# ---------------------------------------------------------------------------
# C. Path / query features
# ---------------------------------------------------------------------------

def get_path_length(url: str) -> int:
    """
    Character length of the URL path component.
    RATIONALE: Phishing landing pages often have long, random-looking paths
    auto-generated by phishing kits to evade duplicate-detection blocklists.
    """
    return len(urlparse(url).path)


def get_num_query_params(url: str) -> int:
    """
    Number of distinct query parameters in the URL.
    RATIONALE: Excessive query parameters suggest tracking, obfuscation,
    or auto-generated phishing page parameters (session tokens, redirect targets).
    """
    query = urlparse(url).query
    return len(parse_qs(query)) if query else 0


def has_redirect_in_query(url: str) -> bool:
    """
    True if any query parameter value contains 'http' (open redirect pattern).
    RATIONALE: Open redirects use a legitimate domain as a front:
    'legitimate.com/redirect?url=http://phishing.com'. Detecting 'http'
    inside a query value is a lightweight heuristic for this pattern.
    """
    return "http" in urlparse(url).query.lower()


def get_url_entropy(url: str) -> float:
    """
    Shannon entropy of the URL string (bits).
    RATIONALE: DGA-generated domains and random paths have high entropy because
    their character distribution is more uniform (random-looking). Legitimate
    URLs tend to contain recognizable words with lower entropy.
    Formula: H = -sum(p_i * log2(p_i)) for each unique character's probability.
    """
    if not url:
        return 0.0
    probs = [url.count(c) / len(url) for c in set(url)]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def has_port_in_url(url: str) -> bool:
    """
    True if the URL explicitly specifies a non-standard port.
    RATIONALE: Legitimate web services almost never expose non-standard ports
    (not 80 or 443) in user-facing URLs. A URL like 'http://paypal.com:8080/login'
    is anomalous and may be used to evade port-based firewall rules.
    """
    port = urlparse(url).port
    return port is not None and port not in (80, 443)


# ---------------------------------------------------------------------------
# Master feature list and extraction function
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "url_length", "has_ip_address", "has_at_symbol", "num_hyphens_in_domain",
    "num_dots", "num_subdomains", "has_https", "has_suspicious_tld",
    "has_url_shortener", "digit_ratio", "has_double_slash_in_path",
    "special_char_count", "domain_length", "has_brand_keyword_mismatch",
    "path_length", "num_query_params", "has_redirect_in_query",
    "url_entropy", "has_port_in_url",
]


def extract_features(url: str) -> Dict[str, Any]:
    """
    Master function: takes a raw URL string and returns a dictionary of all 19 features.
    Boolean features are cast to int (0/1) for scikit-learn compatibility.

    Args:
        url: A raw URL string (normalized internally via normalize_url).

    Returns:
        Dict mapping feature_name -> numeric value (int or float).

    Raises:
        ValueError: If the URL cannot be parsed.
    """
    from src.utils import normalize_url
    url = normalize_url(url)

    return {
        "url_length":               get_url_length(url),
        "has_ip_address":           int(has_ip_address(url)),
        "has_at_symbol":            int(has_at_symbol(url)),
        "num_hyphens_in_domain":    get_num_hyphens_in_domain(url),
        "num_dots":                 get_num_dots(url),
        "num_subdomains":           get_num_subdomains(url),
        "has_https":                int(has_https(url)),
        "has_suspicious_tld":       int(has_suspicious_tld(url)),
        "has_url_shortener":        int(has_url_shortener(url)),
        "digit_ratio":              get_digit_ratio(url),
        "has_double_slash_in_path": int(has_double_slash_in_path(url)),
        "special_char_count":       get_special_char_count(url),
        "domain_length":            get_domain_length(url),
        "has_brand_keyword_mismatch": int(has_brand_keyword_mismatch(url)),
        "path_length":              get_path_length(url),
        "num_query_params":         get_num_query_params(url),
        "has_redirect_in_query":    int(has_redirect_in_query(url)),
        "url_entropy":              get_url_entropy(url),
        "has_port_in_url":          int(has_port_in_url(url)),
    }
