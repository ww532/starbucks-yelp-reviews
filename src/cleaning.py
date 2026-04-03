"""
cleaning.py
-----------
Data cleaning and schema normalization utilities.

Handles deduplication, null auditing, type casting, column renaming,
city name standardization, and three-way table joins to produce a
clean, analysis-ready DataFrame.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# City name correction map — handles common Yelp data inconsistencies
# ---------------------------------------------------------------------------
CITY_CORRECTIONS = {
    "Las Vegas":       ["las vegas", "las  vegas", "las vagas"],
    "Philadelphia":    ["philly", "philadelphia, pa"],
    "Nashville":       ["nashville tn", "nashvlle"],
    "Tucson":          ["tucson az"],
    "New Orleans":     ["new olreans", "new orlean"],
    "Tampa":           ["tampa bay"],
    "Indianapolis":    ["indy", "indianapolis in"],
    "Pittsburgh":      ["pittsburgh pa"],
}

# Build reverse lookup: variant → canonical
_CITY_MAP: dict[str, str] = {}
for canonical, variants in CITY_CORRECTIONS.items():
    for v in variants:
        _CITY_MAP[v] = canonical


def deduplicate(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Remove duplicate rows based on a unique identifier column.

    Parameters
    ----------
    df : pd.DataFrame
    key_col : str
        Column name that should be unique (e.g., 'review_id')

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame; original index reset.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[key_col]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  Deduplication [{key_col}]: removed {removed:,} duplicate rows")
    else:
        print(f"  Deduplication [{key_col}]: no duplicates found")
    return df


def audit_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a null rate summary table for all columns in the DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: column, null_count, null_pct
    """
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "column":     null_counts.index,
        "null_count": null_counts.values,
        "null_pct":   null_pct.values,
    })
    summary = summary[summary["null_count"] > 0].sort_values("null_pct", ascending=False)
    if summary.empty:
        print("  Null audit: no missing values detected")
    else:
        print("  Null audit results:")
        print(summary.to_string(index=False))
    return summary


def drop_critical_nulls(df: pd.DataFrame, critical_cols: list[str]) -> pd.DataFrame:
    """
    Drop rows where any critical column contains a null value.

    Parameters
    ----------
    critical_cols : list[str]
        Columns that must be non-null for a record to be usable.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with critical nulls removed.
    """
    before = len(df)
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    removed = before - len(df)
    print(f"  Dropped {removed:,} rows with nulls in critical columns: {critical_cols}")
    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply consistent data type casting across the joined review DataFrame.

    Converts date strings to datetime, ensures star ratings are integers,
    and standardizes text fields as strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected column types.
    """
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["review_stars", "business_avg_stars", "business_review_count",
                "user_review_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["review_id", "user_id", "business_id", "text", "name",
                "city", "state", "user_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    print("  Data type casting complete")
    return df


def normalize_city(city: str) -> str:
    """
    Standardize a raw city name to its canonical form.

    Applies title-casing, strips whitespace, and corrects known variants.
    """
    cleaned = str(city).strip().title()
    lower = cleaned.lower()
    return _CITY_MAP.get(lower, cleaned)


def normalize_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply city name normalization across all rows.

    Creates a new column `city_normalized` preserving the original `city` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with added `city_normalized` column.
    """
    df["city_normalized"] = df.apply(
        lambda r: normalize_city(r.get("city", "")), axis=1
    )
    print(f"  City normalization complete — {df['city_normalized'].nunique():,} unique cities")
    return df


def join_frames(
    reviews: pd.DataFrame,
    businesses: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform a three-way left join: reviews ← businesses ← users.

    Produces a single wide-format analysis table containing all review,
    business, and reviewer attributes in one DataFrame.

    Parameters
    ----------
    reviews : pd.DataFrame
    businesses : pd.DataFrame
    users : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Joined DataFrame. Reviews are the base table; unmatched business
        or user records are excluded.
    """
    # Join reviews → businesses
    df = reviews.merge(
        businesses[["business_id", "name", "city", "city_normalized", "state",
                     "business_avg_stars", "business_review_count"]],
        on="business_id",
        how="left",
    )

    # Join → users
    df = df.merge(
        users[["user_id", "user_name", "user_review_count",
               "yelping_since", "user_avg_stars"]],
        on="user_id",
        how="left",
    )

    print(f"  Three-way join complete → {len(df):,} rows × {df.shape[1]} columns")
    return df


def print_schema_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted schema summary: shape, column types, and null counts.
    """
    print(f"\n{'='*60}")
    print(f"  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*60}")
    info = pd.DataFrame({
        "dtype":      df.dtypes,
        "null_count": df.isnull().sum(),
        "null_pct":   (df.isnull().sum() / len(df) * 100).round(2),
    })
    print(info.to_string())
    print(f"{'='*60}\n")
