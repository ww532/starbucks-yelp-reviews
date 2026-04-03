"""
mappings.py
-----------
Category mapping functions for the Starbucks Customer Voice Intelligence project.

Each function transforms one or more raw columns into a derived analytical
dimension. All mappings are deterministic and can be applied independently.
"""

import pandas as pd
import re


# ---------------------------------------------------------------------------
# 1. Star Tier — convert numeric rating to business-meaningful category
# ---------------------------------------------------------------------------

def map_star_tier(stars: pd.Series) -> pd.Series:
    """
    Map numeric star ratings (1–5) to three-tier satisfaction categories.

    Tiers are aligned with standard NPS/CSAT frameworks:
      - Critical  (1–2): high churn-risk, requires service recovery attention
      - Neutral   (3):   passive satisfaction, no strong brand advocacy
      - Positive  (4–5): loyal customers, potential brand promoters

    Returns
    -------
    pd.Series of str
    """
    def _tier(s):
        if pd.isna(s):
            return "Unknown"
        s = int(s)
        if s <= 2:
            return "Critical"
        if s == 3:
            return "Neutral"
        return "Positive"

    return stars.apply(_tier)


# ---------------------------------------------------------------------------
# 2. Brand Category — normalize business names to competitive segments
# ---------------------------------------------------------------------------

_BRAND_MAP = [
    ("Starbucks",        ["starbucks"]),
    ("Dunkin'",          ["dunkin'", "dunkin donuts", "dunkin\u2019"]),
    ("Dutch Bros",       ["dutch bros"]),
    ("Peet's Coffee",    ["peet's coffee", "peets coffee", "peet's coffee & tea"]),
    ("The Coffee Bean",  ["the coffee bean", "coffee bean & tea leaf"]),
]


def map_brand_category(name: pd.Series) -> pd.Series:
    """
    Normalize Yelp business names into five competitive brand segments.

    Segments:
      - Starbucks         : primary brand under analysis
      - Dunkin'           : primary chain competitor for benchmarking
      - Dutch Bros         : fast-casual drive-through chain competitor
      - Peet's Coffee      : premium chain competitor
      - The Coffee Bean    : regional chain competitor
      - Independent Café   : non-chain local cafés

    Returns
    -------
    pd.Series of str
    """
    def _categorize(n):
        n_lower = str(n).lower()
        for brand, keywords in _BRAND_MAP:
            if any(kw in n_lower for kw in keywords):
                return brand
        return "Independent Café"

    return name.apply(_categorize)


# ---------------------------------------------------------------------------
# 3. Review Length & Length Tier
# ---------------------------------------------------------------------------

def map_review_length(text: pd.Series) -> pd.Series:
    """
    Compute word count for each review text.

    Returns
    -------
    pd.Series of int
    """
    return text.fillna("").apply(lambda t: len(str(t).split()))


def map_length_tier(length: pd.Series) -> pd.Series:
    """
    Classify review word count into three content-depth tiers.

      - Short  (< 50 words):  brief, signal-heavy
      - Medium (50–150 words): standard narrative review
      - Long   (> 150 words): detailed, high-information review

    Returns
    -------
    pd.Series of str
    """
    def _tier(n):
        if n < 50:
            return "Short"
        if n <= 150:
            return "Medium"
        return "Long"

    return length.apply(_tier)


# ---------------------------------------------------------------------------
# 4. Datetime Decomposition
# ---------------------------------------------------------------------------

def map_datetime_fields(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extract analytical time dimensions from a datetime column.

    Adds the following columns:
      year, month, quarter, day_of_week (0=Mon), year_quarter, is_weekend

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
        Name of the datetime column to decompose.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with time dimension columns appended.
    """
    dt = df[date_col]
    df["year"]         = dt.dt.year.astype("Int16")
    df["month"]        = dt.dt.month.astype("Int8")
    df["quarter"]      = dt.dt.quarter.astype("Int8")
    df["day_of_week"]  = dt.dt.dayofweek.astype("Int8")   # 0=Monday, 6=Sunday
    df["year_quarter"] = dt.dt.to_period("Q").astype(str)  # e.g. "2019Q3"
    df["is_weekend"]   = dt.dt.dayofweek >= 5
    return df


# ---------------------------------------------------------------------------
# 5. User Activity Tier — segment reviewers by engagement level
# ---------------------------------------------------------------------------

def map_user_activity_tier(review_count: pd.Series) -> pd.Series:
    """
    Segment Yelp users into four activity tiers based on lifetime review count.

    Tiers serve as a proxy for reviewer credibility and familiarity with
    rating conventions:
      - Casual  (< 10 reviews):   occasional reviewer, limited context
      - Regular (10–49):          engaged user, moderate reliability
      - Power   (50–199):         highly active, experienced reviewer
      - Elite   (200+):           top-tier contributor, high-credibility signal

    Returns
    -------
    pd.Series of str
    """
    def _tier(n):
        if pd.isna(n):
            return "Unknown"
        n = int(n)
        if n < 10:
            return "Casual"
        if n < 50:
            return "Regular"
        if n < 200:
            return "Power"
        return "Elite"

    return review_count.apply(_tier)


# ---------------------------------------------------------------------------
# 6. Topic Tag — rule-based operational theme classification
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS = {
    "Service":       r"\b(staff|barista|service|rude|friendly|cashier|employee|attitude|wait|slow|fast|line|order|wrong)\b",
    "Food Quality":  r"\b(drink|coffee|latte|espresso|taste|flavor|quality|stale|fresh|matcha|frappuccino|food|sandwich|pastry)\b",
    "Price":         r"\b(price|expensive|cheap|cost|value|overpriced|worth|pricey|afford)\b",
    "Ambiance":      r"\b(ambiance|atmosphere|vibe|music|loud|quiet|cozy|seating|decor|clean|dirty|noise|crowded)\b",
    "Wait Time":     r"\b(wait|waited|waiting|slow|minutes|forever|quick|fast|queue|line|busy)\b",
    "Cleanliness":   r"\b(clean|dirty|mess|trash|restroom|bathroom|hygiene|filthy|spotless)\b",
}


def map_topic_tag(text: pd.Series) -> pd.Series:
    """
    Assign a primary operational topic tag to each review using keyword matching.

    Classification follows a priority order; the first matching topic is assigned.
    Reviews with no keyword match are labeled 'General'.

    Topics:
      Service | Food Quality | Price | Ambiance | Wait Time | Cleanliness | General

    Returns
    -------
    pd.Series of str
    """
    compiled = {
        topic: re.compile(pattern, re.IGNORECASE)
        for topic, pattern in _TOPIC_PATTERNS.items()
    }

    def _tag(t):
        t = str(t)
        for topic, pattern in compiled.items():
            if pattern.search(t):
                return topic
        return "General"

    return text.fillna("").apply(_tag)


# ---------------------------------------------------------------------------
# Apply all mappings to a DataFrame in one call
# ---------------------------------------------------------------------------

def apply_all_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full suite of category mapping functions to the joined DataFrame.

    This is a convenience wrapper used by the project pipeline and Notebook 03.
    All mapping functions are idempotent; re-running will overwrite prior output.

    Parameters
    ----------
    df : pd.DataFrame
        Joined, cleaned DataFrame from Notebook 02.

    Returns
    -------
    pd.DataFrame
        DataFrame with all derived analytical columns added.
    """
    print("Applying category mappings...")

    df["star_tier"]          = map_star_tier(df["review_stars"])
    df["brand_category"]     = map_brand_category(df["name"])
    df["review_length"]      = map_review_length(df["text"])
    df["length_tier"]        = map_length_tier(df["review_length"])
    df["user_activity_tier"] = map_user_activity_tier(df["user_review_count"] if "user_review_count" in df.columns else pd.Series(dtype="float64"))
    df["topic_tag"]          = map_topic_tag(df["text"])
    df                       = map_datetime_fields(df, date_col="date")

    print(f"  star_tier        → {df['star_tier'].value_counts().to_dict()}")
    print(f"  brand_category   → {df['brand_category'].value_counts().to_dict()}")
    print(f"  length_tier      → {df['length_tier'].value_counts().to_dict()}")
    print(f"  user_tier        → {df['user_activity_tier'].value_counts().to_dict()}")
    print(f"  topic_tag        → {df['topic_tag'].value_counts().to_dict()}")

    return df
