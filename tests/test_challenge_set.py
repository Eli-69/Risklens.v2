import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib
from urllib.parse import urlparse

from src.features.build_features import extract_all_features

# Load CSV (ignore comments)
df = pd.read_csv("data/challenge_set.csv", comment="#")

model = joblib.load("models/trained/risklens_model.pkl")
cols = joblib.load("models/trained/feature_columns.pkl")

correct = 0

# --- Trusted domains ---
trusted_domains = [
    "github.com",
    "google.com",
    "microsoft.com",
    "amazon.com",
    "bbc.co.uk",
    "pvamu.edu",
    "nationalarchives.gov.uk",
]

# --- Trusted auth domains ---
trusted_auth_domains = [
    "accounts.google.com",
    "login.microsoftonline.com",
]

for _, row in df.iterrows():
    url = str(row["url"]).strip()
    expected = str(row["expected_label"]).strip().lower()
    category = str(row["category"]).strip()

    features = extract_all_features(url, use_page_features=True)

    X = pd.DataFrame([features])
    X = X.reindex(columns=cols, fill_value=0)

    proba = float(model.predict_proba(X)[0][1])

    # --- Model prediction ---
    if proba > 0.9:
        pred = "phishing"
    elif proba > 0.6:
        pred = "suspicious"
    else:
        pred = "legitimate"

    decision_source = "model"

    # 1. HTTP softener
    if features.get("is_https_url", 0) == 0 and pred == "phishing":
        pred = "suspicious"
        decision_source = "http_softener"

    # 2. TLS rule
    if (
        features.get("is_https_url", 0) == 1
        and features.get("has_cert", 0) == 0
        and pred == "legitimate"
        and features.get("page_fetch_ok", 0) == 0
    ):
        pred = "suspicious"
        decision_source = "tls_no_page_no_cert"

    hostname = (urlparse(url).hostname or "").lower()

    # 3. Trusted domain override
    if any(hostname.endswith(d) for d in trusted_domains):
        if pred == "phishing":
            pred = "legitimate"
            decision_source = "trusted_domain_override"

    # 4. Trusted auth override
    if hostname in trusted_auth_domains:
        if pred == "phishing":
            pred = "legitimate"
            decision_source = "trusted_auth_override"

    print("\n==============================")
    print("URL:", url)
    print("Category:", category)
    print("Expected:", expected)
    print("Predicted:", pred)
    print("Score:", round(proba, 4))
    print("Decision source:", decision_source)

    if (expected == "legitimate" and pred == "legitimate") or \
       (expected == "phishing" and pred == "phishing"):
        correct += 1

accuracy = correct / len(df)

print("\n==============================")
print("FINAL ACCURACY:", round(accuracy, 4))