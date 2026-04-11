from src.features.url_features import extract_features as extract_url_features
from src.features.tls_features import extract_tls_features

def extract_all_features(url: str) -> dict:
    features = {}
    features.update(extract_url_features(url))
    features.update(extract_tls_features(url))
    return features