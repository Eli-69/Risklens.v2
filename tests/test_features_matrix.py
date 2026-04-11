import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.features.url_features import extract_features

df = pd.read_csv("data/raw/better_url_dataset.csv")
df["url"] = df["url"].astype(str).str.strip()

urls = df["url"].tolist()[:1000]

print("Building features...")

rows = [extract_features(u) for u in urls]
X = pd.DataFrame(rows)

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X.select_dtypes(include=[np.number])

print("Feature shape:", X.shape)
print("First 10 columns:", list(X.columns)[:10])
print(X.head())