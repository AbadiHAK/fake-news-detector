from sklearn.feature_extraction.text import TfidfVectorizer
from .config import TFIDF_MAX_FEATURES


def build_tfidf():
    return TfidfVectorizer(max_features=TFIDF_MAX_FEATURES , ngram_range=(1,2))
