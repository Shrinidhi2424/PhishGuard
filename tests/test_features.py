"""
test_features.py — Unit tests for PhishGuard feature extraction.
Run with: pytest tests/test_features.py -v
"""

import pytest
from src.features import extract_features, FEATURE_NAMES

# Test URLs with known expected feature values
PHISHING_IP     = "http://192.168.1.1/bank/login"
PHISHING_AT     = "http://paypal.com@attacker.ru/verify"
PHISHING_TK     = "http://paypal-secure-login.verify-account.tk/signin"
PHISHING_SHORT  = "http://bit.ly/3xAbc9q"
LEGIT_GOOGLE    = "https://www.google.com"
LEGIT_GITHUB    = "https://github.com/user/repository"
LEGIT_PAYPAL    = "https://www.paypal.com/signin"
BRAND_MISMATCH  = "http://paypal.login.verify.attacker.com/account"


class TestIPAddress:
    def test_ip_url_flagged(self):
        assert extract_features(PHISHING_IP)["has_ip_address"] == 1

    def test_domain_not_flagged(self):
        assert extract_features(LEGIT_GOOGLE)["has_ip_address"] == 0


class TestAtSymbol:
    def test_at_symbol_detected(self):
        assert extract_features(PHISHING_AT)["has_at_symbol"] == 1

    def test_no_at_symbol(self):
        assert extract_features(LEGIT_GOOGLE)["has_at_symbol"] == 0


class TestSuspiciousTLD:
    def test_tk_tld_flagged(self):
        assert extract_features(PHISHING_TK)["has_suspicious_tld"] == 1

    def test_com_tld_clean(self):
        assert extract_features(LEGIT_GOOGLE)["has_suspicious_tld"] == 0


class TestURLShortener:
    def test_bitly_flagged(self):
        assert extract_features(PHISHING_SHORT)["has_url_shortener"] == 1

    def test_legitimate_not_flagged(self):
        assert extract_features(LEGIT_GOOGLE)["has_url_shortener"] == 0


class TestHTTPS:
    def test_https_present(self):
        assert extract_features(LEGIT_GOOGLE)["has_https"] == 1

    def test_http_not_https(self):
        assert extract_features(PHISHING_TK)["has_https"] == 0


class TestBrandKeywordMismatch:
    def test_mismatch_detected(self):
        assert extract_features(BRAND_MISMATCH)["has_brand_keyword_mismatch"] == 1

    def test_no_mismatch_for_real_paypal(self):
        # 'paypal' appears in URL AND registered domain — no mismatch
        assert extract_features(LEGIT_PAYPAL)["has_brand_keyword_mismatch"] == 0


class TestURLLength:
    def test_phishing_longer_than_legit(self):
        phish_len = extract_features(PHISHING_TK)["url_length"]
        legit_len = extract_features(LEGIT_GOOGLE)["url_length"]
        assert phish_len > legit_len


class TestSubdomains:
    def test_multiple_subdomains(self):
        # paypal.login.verify = 3 subdomain segments
        assert extract_features(BRAND_MISMATCH)["num_subdomains"] >= 3

    def test_www_counts_as_one(self):
        assert extract_features(LEGIT_GOOGLE)["num_subdomains"] == 1


class TestEntropy:
    def test_random_string_has_higher_entropy(self):
        low_entropy_url = "https://www.google.com"
        high_entropy_url = "https://x89q2w8ef7v89b34yv789by4v9.tk/q98f7y23b498"
        assert extract_features(high_entropy_url)["url_entropy"] > extract_features(low_entropy_url)["url_entropy"]


class TestSpecialChars:
    def test_special_chars_counted(self):
        special_url = "https://example.com/test?a=1&b=2%20#tag"
        assert extract_features(special_url)["special_char_count"] >= 3


class TestFeatureSchema:
    def test_all_features_present(self):
        features = extract_features(LEGIT_GOOGLE)
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_no_none_values(self):
        features = extract_features(LEGIT_GOOGLE)
        for name, value in features.items():
            assert value is not None, f"Feature {name} is None"

    def test_feature_count(self):
        assert len(FEATURE_NAMES) == 19
