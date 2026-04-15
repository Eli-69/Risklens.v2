import json
import math
from pathlib import Path
from urllib.parse import urlparse

import tldextract

BASE_DIR = Path(__file__).resolve().parent

ALLOWLIST_PATH = BASE_DIR / "trusted_domains.json"
POPULARITY_PATH = BASE_DIR / "domain_counts.json"

try:
    with open(ALLOWLIST_PATH, "r", encoding="utf-8") as f:
        TRUSTED_DOMAINS = set(json.load(f))
except Exception:
    TRUSTED_DOMAINS = {
        "google.com",
        "github.com",
        "microsoft.com",
        "amazon.com",
        "bbc.co.uk",
        "pvamu.edu",
    }

try:
    with open(POPULARITY_PATH, "r", encoding="utf-8") as f:
        DOMAIN_COUNTS = json.load(f)
except Exception:
    DOMAIN_COUNTS = {}


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return url


def _get_hostname(url: str) -> str:
    try:
        return (urlparse(_normalize_url(url)).hostname or "").lower()
    except Exception:
        return ""


def _registered_domain(hostname: str) -> str:
    if not hostname:
        return ""
    ext = tldextract.extract(hostname)
    return ext.registered_domain


def extract_reputation_features(url: str) -> dict:
    hostname = _get_hostname(url)
    reg_domain = _registered_domain(hostname)

    trusted_suffixes = (
        ".github.io",
        ".vercel.app",
        ".netlify.app",
        ".pages.dev",
    )

    in_allowlist = int(
        reg_domain in TRUSTED_DOMAINS or hostname in TRUSTED_DOMAINS
    )

    is_gov_or_edu = int(
        hostname.endswith(".gov")
        or hostname.endswith(".edu")
        or hostname.endswith(".gov.uk")
        or hostname.endswith(".ac.uk")
    )

    in_trusted_suffix = int(any(hostname.endswith(s) for s in trusted_suffixes))

    domain_popularity_score = math.log1p(DOMAIN_COUNTS.get(reg_domain, 1))

    return {
        "domain_in_allowlist": in_allowlist,
        "is_gov_or_edu": is_gov_or_edu,
        "in_trusted_suffix": in_trusted_suffix,
        "domain_popularity_score": domain_popularity_score,
    }