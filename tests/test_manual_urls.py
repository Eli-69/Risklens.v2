import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from urllib.parse import urlparse

from src.features.build_features import extract_all_features

MODEL_PATH = Path("models/trained/risklens_model.pkl")
COLS_PATH = Path("models/trained/feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_cols = joblib.load(COLS_PATH)

test_urls = [
    "https://www.google.com",
    "https://paypal-login-secure.ru/verify",
    "https://www.github.com",
    "http://example.com",
    "https://expired.badssl.com",
    "https://self-signed.badssl.com",
    "https://www.bbc.co.uk",
    "https://random-small-site.org",
    "https://nonsauto.com/",
    "https://www.pvamu.edu/",
    "https://vercel.com/eli-69s-projects/risklens",
    "https://github.com/jomo22/seniordesign_extrafeatures",
    "https://www.nationalarchives.gov.uk/webarchive/",
]

trusted_domains = [
    "github.com",
    "google.com",
    "microsoft.com",
    "amazon.com",
    "bbc.co.uk",
    "pvamu.edu",
    "nationalarchives.gov.uk",
]

for url in test_urls:
    print("\n==============================")
    print("URL:", url)

    features = extract_all_features(url, use_page_features=True)

    X = pd.DataFrame([features])
    X = X.reindex(columns=feature_cols, fill_value=0)

    proba = float(model.predict_proba(X)[0][1])

    if proba > 0.9:
        label = "PHISHING"
    elif proba > 0.6:
        label = "SUSPICIOUS"
    else:
        label = "LEGITIMATE"

    decision_source = "model"

    # HTTP softener
    if features.get("is_https_url", 0) == 0 and label == "PHISHING":
        label = "SUSPICIOUS"
        decision_source = "http_softener"

    # TLS invalid / missing on HTTPS should not end as LEGITIMATE
    if features.get("is_https_url", 0) == 1 and features.get("has_cert", 0) == 0:
        if label == "LEGITIMATE":
            label = "SUSPICIOUS"
            decision_source = "tls_invalid"

    # Trusted domain override
    hostname = (urlparse(url).hostname or "").lower()
    if any(hostname.endswith(d) for d in trusted_domains):
        if label == "PHISHING":
            label = "LEGITIMATE"
            decision_source = "trusted_domain_override"

    print("Score:", round(proba, 4))
    print("is_https:", features.get("is_https_url"))
    print("has_cert:", features.get("has_cert"))
    print("cert_valid_now:", features.get("cert_valid_now"))
    print("certificate_score:", features.get("certificate_score"))
    print("cert_days_to_expiry:", features.get("cert_days_to_expiry"))
    print("page_fetch_ok:", features.get("page_fetch_ok"))
    print("word_count:", features.get("word_count"))
    print("thin_page:", features.get("thin_page"))
    print("has_password_field:", features.get("has_password_field"))
    print("form_action_external_domain:", features.get("form_action_external_domain"))
    print("submits_to_same_domain:", features.get("submits_to_same_domain"))
    print("suspicious_login_pattern:", features.get("suspicious_login_pattern"))

    print("Prediction:", label)
    print("Decision source:", decision_source)