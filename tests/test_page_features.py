import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.features.page_features import extract_page_features

test_urls = [
    "https://google.com",
    "https://github.com",
    "https://bbc.co.uk",
    "http://example.com",
]

for url in test_urls:
    feats = extract_page_features(url)
    print("\nURL:", url)
    for k, v in feats.items():
        print(f"{k}: {v}")