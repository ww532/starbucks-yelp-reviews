"""
nlp_utils.py
------------
Natural language processing utilities for the Starbucks Customer Voice Intelligence project.

Provides VADER-based sentiment scoring, TF-IDF keyword extraction,
and stop-word filtering tailored for restaurant/café review text.
"""

import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ---------------------------------------------------------------------------
# Sentiment Scoring — VADER
# ---------------------------------------------------------------------------

_analyzer = SentimentIntensityAnalyzer()

# Yelp-specific additions to the VADER lexicon
_CUSTOM_LEXICON = {
    "overpriced":  -2.0,
    "burnt":       -1.5,
    "undercooked": -2.0,
    "watery":      -1.5,
    "stale":       -2.0,
    "rude":        -2.5,
    "unfriendly":  -2.0,
    "amazing":      2.5,
    "fantastic":    2.5,
    "obsessed":     2.0,
    "gem":          2.0,
    "cozy":         1.5,
}

for word, score in _CUSTOM_LEXICON.items():
    _analyzer.lexicon[word] = score


def score_sentiment(text: str) -> tuple[float, str]:
    """
    Compute a VADER compound sentiment score and label for a single review.

    Thresholds follow the standard VADER convention:
      - Positive : compound ≥  0.05
      - Neutral  : -0.05 < compound < 0.05
      - Negative : compound ≤ -0.05

    Parameters
    ----------
    text : str
        Raw review text (original case preserved — VADER is case-sensitive).

    Returns
    -------
    tuple[float, str]
        (compound_score, sentiment_label)
    """
    scores = _analyzer.polarity_scores(str(text))
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label


def apply_sentiment(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply VADER sentiment scoring to all rows in the DataFrame.

    Adds two columns:
      - sentiment_score : float, compound score in [-1.0, 1.0]
      - sentiment_label : str, one of Positive / Neutral / Negative

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
        Column containing review text.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentiment columns appended.
    """
    print(f"Running VADER sentiment scoring on {len(df):,} reviews...")
    results = df[text_col].apply(lambda t: pd.Series(score_sentiment(t)))
    df["sentiment_score"] = results[0]
    df["sentiment_label"] = results[1]
    dist = df["sentiment_label"].value_counts().to_dict()
    print(f"  Sentiment distribution: {dist}")
    return df


# ---------------------------------------------------------------------------
# Keyword Extraction — TF-IDF
# ---------------------------------------------------------------------------

# Extended stop-word list for café/restaurant review text
_STOP_WORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "they",
    "it", "its", "the", "a", "an", "and", "but", "or", "so", "yet",
    "for", "of", "to", "in", "on", "at", "by", "with", "from", "as",
    "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "not", "no", "nor", "very", "too", "also", "just", "really",
    "this", "that", "these", "those", "here", "there", "when", "where",
    "how", "what", "which", "who", "if", "then", "than", "because",
    "get", "got", "go", "went", "come", "came", "one", "two", "time",
    "more", "most", "even", "back", "can", "like", "about", "up", "out",
    "place", "location", "starbucks", "coffee", "café", "cafe",
}


def extract_top_keywords(
    texts: pd.Series,
    n_top: int = 30,
    ngram_range: tuple = (1, 2),
) -> pd.DataFrame:
    """
    Extract the most frequent meaningful terms from a corpus of review texts.

    Uses TF-IDF vectorization with a custom stop-word list tailored to
    coffee shop review language. Both unigrams and bigrams are considered.

    Parameters
    ----------
    texts : pd.Series
        Review text series (typically pre-filtered by sentiment or brand).
    n_top : int
        Number of top terms to return (default: 30).
    ngram_range : tuple
        N-gram range for the TF-IDF vectorizer (default: (1, 2)).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: term, tfidf_score, sorted descending.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    clean_texts = texts.fillna("").apply(_clean_for_tfidf)
    clean_texts = clean_texts[clean_texts.str.len() > 0]

    if len(clean_texts) < 5:
        return pd.DataFrame(columns=["term", "tfidf_score"])

    vectorizer = TfidfVectorizer(
        stop_words=list(_STOP_WORDS),
        ngram_range=ngram_range,
        max_features=5000,
        min_df=3,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    mean_scores = tfidf_matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()

    result = pd.DataFrame({"term": terms, "tfidf_score": mean_scores})
    result = result.sort_values("tfidf_score", ascending=False).head(n_top)
    return result.reset_index(drop=True)


def _clean_for_tfidf(text: str) -> str:
    """
    Minimal text cleaning for TF-IDF input.

    Lowercases, removes punctuation, and strips extra whitespace.
    Does NOT remove all stop words here — that is handled by TfidfVectorizer.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


