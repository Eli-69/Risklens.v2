import re
from urllib.parse import urljoin, urlparse

import requests

HEADERS = {
    "User-Agent": "RiskLens/2.0"
}

REQUEST_TIMEOUT = 5


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return url


def _safe_get(url: str):
    try:
        return requests.get(
            _normalize_url(url),
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
    except Exception:
        return None


def _find_links(html: str, base_url: str) -> list:
    if not html:
        return []

    hrefs = re.findall(r'href=["\'](.*?)["\']', html, flags=re.IGNORECASE)
    out = []
    for href in hrefs:
        try:
            out.append(urljoin(base_url, href))
        except Exception:
            continue
    return out


def _contains_any(text: str, keywords: list) -> int:
    if not text:
        return 0
    text = text.lower()
    return int(any(k in text for k in keywords))


def _same_domain(url1: str, url2: str) -> int:
    try:
        h1 = (urlparse(url1).hostname or "").lower()
        h2 = (urlparse(url2).hostname or "").lower()
        return int(h1 == h2)
    except Exception:
        return 0


def extract_page_features(url: str) -> dict:
    features = {
        "page_fetch_ok": 0,
        "final_url_is_https": 0,
        "has_privacy_policy_link": 0,
        "has_terms_link": 0,
        "has_contact_link": 0,
        "has_about_link": 0,
        "has_login_form": 0,
        "num_forms": 0,
        "has_footer": 0,
        "page_title_present": 0,
        "same_domain_final_url": 0,
    }

    response = _safe_get(url)
    if response is None:
        return features

    final_url = response.url
    html = response.text or ""

    features["page_fetch_ok"] = 1
    features["final_url_is_https"] = int(final_url.lower().startswith("https://"))
    features["same_domain_final_url"] = _same_domain(_normalize_url(url), final_url)

    html_lower = html.lower()
    links = _find_links(html, final_url)
    joined_links = " ".join(links).lower()

    features["has_privacy_policy_link"] = _contains_any(
        joined_links + " " + html_lower,
        ["privacy policy", "/privacy", "privacy"]
    )

    features["has_terms_link"] = _contains_any(
        joined_links + " " + html_lower,
        ["terms of service", "terms and conditions", "/terms", "terms"]
    )

    features["has_contact_link"] = _contains_any(
        joined_links + " " + html_lower,
        ["contact us", "/contact", "contact"]
    )

    features["has_about_link"] = _contains_any(
        joined_links + " " + html_lower,
        ["about us", "/about", "about"]
    )

    features["has_login_form"] = int(
        ('type="password"' in html_lower) or ("type='password'" in html_lower)
    )

    features["num_forms"] = len(re.findall(r"<form\b", html_lower))
    features["has_footer"] = int("<footer" in html_lower)
    features["page_title_present"] = int("<title" in html_lower)

    return features