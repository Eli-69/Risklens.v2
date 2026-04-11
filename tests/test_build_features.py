import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.build_features import extract_all_features

test_urls = [
    "google.com",
    "paypal-login-secure.ru/verify",
    "github.com"
]

for url in test_urls:
    feats = extract_all_features(url)
    print("\nURL:", url)
    print("Feature count:", len(feats))
    print(list(feats.items())[:10])