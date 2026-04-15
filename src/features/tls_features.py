import ssl
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

TRUSTED_CAS = {
    "DigiCert", "GlobalSign", "GeoTrust", "Entrust", "GoDaddy",
    "Comodo", "USERTrust", "Sectigo", "Verisign", "SecureTrust",
    "Certum", "QuoVadis", "AddTrust", "Actalis"
}


def _normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    return url


def _flatten_name(name_tuple_list):
    flat = {}
    for t in name_tuple_list:
        if len(t) > 0 and len(t[0]) == 2:
            key, value = t[0]
            flat[key] = value
    return flat


def fetch_server_cert(url: str, timeout: int = 5):
    parsed = urlparse(_normalize_url(url))
    hostname = (parsed.hostname or "").strip()

    if not hostname:
        return {"error": "missing hostname"}

    context = ssl.create_default_context()

    try:
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return ssock.getpeercert()
    except Exception as e:
        return {"error": str(e)}


def extract_tls_features(url: str) -> dict:
    parsed = urlparse(_normalize_url(url))
    hostname = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "").lower()

    features = {
        "is_https_url": int(scheme == "https"),
        "tls_check_attempted": 0,
        "has_cert": 0,
        "cert_valid_now": 0,
        "cert_days_to_expiry": 0,
        "cert_is_self_signed": 0,
        "cert_issuer_trusted": 0,
        "cert_hostname_match": 0,
        "cert_expiring_soon": 0,
        "certificate_score": 0,
    }

    if scheme != "https":
        return features

    features["tls_check_attempted"] = 1
    cert = fetch_server_cert(url)

    if "error" in cert:
        return features

    try:
        start_date = datetime.strptime(
            cert["notBefore"], "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=timezone.utc)
        exp_date = datetime.strptime(
            cert["notAfter"], "%b %d %H:%M:%S %Y %Z"
        ).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days_remaining = (exp_date - now).days
        valid_now = int(start_date <= now <= exp_date)
    except Exception:
        return features

    issuer = _flatten_name(cert.get("issuer", []))
    subject = _flatten_name(cert.get("subject", []))

    issuer_org = issuer.get("organizationName", "")
    is_self_signed = int(issuer == subject)
    issuer_trusted = int(any(ca in issuer_org for ca in TRUSTED_CAS))

    try:
        ssl.match_hostname(cert, hostname)
        hostname_match = 1
    except Exception:
        hostname_match = 0

    score = 50
    if days_remaining > 90:
        score += 20
    elif days_remaining > 30:
        score += 10
    elif days_remaining <= 0:
        score -= 50

    if is_self_signed:
        score -= 40

    if issuer_trusted:
        score += 20
    else:
        score -= 20

    score = max(0, min(score, 100))

    features.update({
        "has_cert": 1,
        "cert_valid_now": valid_now,
        "cert_days_to_expiry": max(days_remaining, 0),
        "cert_is_self_signed": is_self_signed,
        "cert_issuer_trusted": issuer_trusted,
        "cert_hostname_match": hostname_match,
        "cert_expiring_soon": int(0 < days_remaining <= 30),
        "certificate_score": score,
    })

    return features