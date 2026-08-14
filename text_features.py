"""TF-IDF vectorization helpers shared across the classifier and recommender."""
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, MIN_DF

ENGLISH_STOP_WORDS = "english"


def build_tfidf_vectorizer(max_features: int = TFIDF_MAX_FEATURES) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=MIN_DF,
        stop_words=ENGLISH_STOP_WORDS,
        sublinear_tf=True,
    )
