import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib

from src.features.build_features import extract_all_features

csv_path = Path("data/challenge_set.csv")

print("Exists:", csv_path.exists())
print("Size:", csv_path.stat().st_size)

with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
    preview = f.read(200)
print("Preview:", repr(preview))

df = pd.read_csv(
    csv_path,
    comment="#",
    encoding="utf-8-sig",
    skip_blank_lines=True
)

print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

model = joblib.load("models/trained/risklens_model.pkl")
cols = joblib.load("models/trained/feature_columns.pkl")

correct = 0

for _, row in df.iterrows():
    url = str(row["url"]).strip()
    expected = str(row["expected_label"]).strip().lower()
    category = str(row["category"]).strip()

    features = extract_all_features(url, use_page_features=True)
    X = pd.DataFrame([features]).reindex(columns=cols, fill_value=0)

    proba = float(model.predict_proba(X)[0][1])

    if proba > 0.9:
        pred = "phishing"
    elif proba > 0.6:
        pred = "suspicious"
    else:
        pred = "legitimate"

    print(f"\nURL: {url}")
    print(f"Category: {category}")
    print(f"Expected: {expected}")
    print(f"Predicted: {pred}")
    print(f"Score: {round(proba,4)}")

    if (expected == "legitimate" and pred == "legitimate") or \
       (expected == "phishing" and pred == "phishing"):
        correct += 1

print("\nAccuracy:", correct / len(df))