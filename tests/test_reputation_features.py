import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.reputation_features import extract_reputation_features

test_urls = [
    "https://github.com/jomo22/seniordesign_extrafeatures",
    "https://vercel.com/eli-69s-projects/risklens",
    "https://www.pvamu.edu/",
    "https://random-small-site.org",
    "https://paypal-login-secure.ru/verify"
]

for url in test_urls:
    feats = extract_reputation_features(url)
    print("\nURL:", url)
    for k, v in feats.items():
        print(f"{k}: {v}")