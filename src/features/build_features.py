from src.features.url_features import extract_features as extract_url_features
from src.features.tls_features import extract_tls_features
from src.features.page_features import extract_page_features


def extract_all_features(url: str, use_page_features: bool = True) -> dict:
    features = {}
    features.update(extract_url_features(url))
    features.update(extract_tls_features(url))

    if use_page_features:
        features.update(extract_page_features(url))

    return features