import re
from urllib.parse import urljoin, urlparse

import requests
import tldextract

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


def _extract_form_actions(html: str, base_url: str) -> list:
    actions = re.findall(r'<form[^>]*action=["\'](.*?)["\']', html, flags=re.IGNORECASE)
    out = []
    for action in actions:
        try:
            out.append(urljoin(base_url, action))
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


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _registered_domain(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
        ext = tldextract.extract(host)
        return ext.registered_domain
    except Exception:
        return ""


def _domain_token(url: str) -> str:
    domain = _registered_domain(url)
    return domain.split(".")[0] if domain else ""


def extract_page_features(url: str) -> dict:
    features = {
        "page_fetch_ok": 0,
        "final_url_is_https": 0,
        "has_privacy_policy_link": 0,
        "has_terms_link": 0,
        "has_contact_link": 0,
        "has_about_link": 0,
        "has_login_form": 0,
        "has_password_field": 0,
        "num_forms": 0,
        "has_footer": 0,
        "page_title_present": 0,
        "same_domain_final_url": 0,
        "form_action_external_domain": 0,
        "submits_to_same_domain": 0,
        "word_count": 0,
        "has_phone_number": 0,
        "has_email_address": 0,
        "title_matches_domain_brand": 0,
        "thin_page": 0,
        "suspicious_login_pattern": 0,
        "auth_keyword_present": 0,
        "auth_legit_context": 0,
        "has_forgot_password_link": 0,
        "has_docs_or_reference_context": 0,
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

    # --- basic trust links ---
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

    # --- form + login ---
    features["has_login_form"] = int(
        ('type="password"' in html_lower) or ("type='password'" in html_lower)
    )
    features["has_password_field"] = features["has_login_form"]

    features["num_forms"] = len(re.findall(r"<form\b", html_lower))
    features["has_footer"] = int("<footer" in html_lower)
    features["page_title_present"] = int("<title" in html_lower)

    # --- form behavior ---
    form_actions = _extract_form_actions(html, final_url)
    if form_actions:
        same_domain_flags = [_same_domain(final_url, action) for action in form_actions]
        features["submits_to_same_domain"] = int(any(same_domain_flags))
        features["form_action_external_domain"] = int(not all(same_domain_flags))
    else:
        features["submits_to_same_domain"] = 0
        features["form_action_external_domain"] = 0

    # --- content ---
    text_only = re.sub(r"<[^>]+>", " ", html)
    words = re.findall(r"\b\w+\b", text_only)
    features["word_count"] = len(words)
    features["thin_page"] = int(features["word_count"] < 200)

    # --- identity signals ---
    features["has_phone_number"] = int(
        re.search(r"(\+?\d[\d\-\(\) ]{7,}\d)", html) is not None
    )

    features["has_email_address"] = int(
        re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", html) is not None
    )

    # --- branding ---
    title_text = _extract_title(html).lower()
    domain_token = _domain_token(final_url).lower()

    features["title_matches_domain_brand"] = int(
        domain_token != "" and domain_token in title_text
    )

    # --- suspicious login ---
    features["suspicious_login_pattern"] = int(
        features["has_password_field"] == 1 and
        features["form_action_external_domain"] == 1
    )

    # --- auth keyword detection ---
    auth_keywords = ["login", "signin", "account", "billing", "verify", "auth", "settings"]

    features["auth_keyword_present"] = int(
        any(k in final_url.lower() for k in auth_keywords)
    )

    # --- legit auth context (VERY IMPORTANT) ---
    features["auth_legit_context"] = int(
        features["auth_keyword_present"] == 1 and
        features["submits_to_same_domain"] == 1 and
        features["has_password_field"] == 1
    )

    features["has_forgot_password_link"] = _contains_any(
    joined_links + " " + html_lower,
    ["forgot password", "reset password", "can't sign in"]
    )

    features["has_docs_or_reference_context"] = _contains_any(
    final_url.lower() + " " + html_lower,
    ["docs", "documentation", "reference", "developer", "guide", "api", "manual"]
    )

    return features