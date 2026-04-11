def extract_tls_features(url: str) -> dict:
    return {
        "has_cert": 0,
        "cert_valid_now": 0,
        "cert_days_to_expiry": 0,
        "cert_is_self_signed": 0,
        "cert_hostname_match": 0,
    }