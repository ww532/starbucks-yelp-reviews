"""
ingestion.py
------------
Data loading utilities for the Starbucks Customer Voice Intelligence project.

Handles streaming ingestion of large Yelp JSON files (business, review, user)
using memory-efficient line-by-line and chunk-based reading patterns.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Union


# ---------------------------------------------------------------------------
# Scope definition
# ---------------------------------------------------------------------------
# Strategy: include ALL U.S. coffee & tea businesses, then exclude chains
# that only incidentally serve coffee (convenience stores, fast food, etc.).
# This gives us a clean "coffee market" from which to derive:
#   - Starbucks    → primary analysis subject
#   - Dunkin'      → direct competitor (singled out only if gap warrants it)
#   - Everything else → aggregate market benchmark

EXCLUDED_BRANDS = {
    "mcdonald", "7-eleven", "wawa", "speedway", "circle k",
    "honey baked", "krispy kreme", "perkins", "first watch",
    "panera", "tropical smoothie", "einstein bros", "another broken egg",
    "the egg & i", "keke's breakfast", "jersey mike",
}

# U.S. state abbreviations (50 states + DC)
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA",
    "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY",
    "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX",
    "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

def load_businesses(business_json_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load and filter the Yelp business dataset.

    Filters to U.S. coffee chain locations matching the target brand list.
    Reads the file line-by-line to handle large file sizes efficiently.

    Parameters
    ----------
    business_json_path : str or Path
        Absolute path to yelp_academic_dataset_business.json

    Returns
    -------
    pd.DataFrame
        Filtered business records with columns:
        business_id, name, city, state, categories, stars, review_count
    """
    records = []
    path = Path(business_json_path)

    print(f"Loading businesses from: {path.name}")
    total = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            record = json.loads(line.strip())

            # Filter: U.S. locations only
            if record.get("state", "") not in US_STATES:
                continue

            # Filter: keep businesses whose primary category is coffee OR whose name
            # contains a coffee-related keyword. This two-pronged approach captures:
            #   - Chains like Starbucks (tagged as "Food" first on Yelp but name is clear)
            #   - Independent cafés with generic names but correct primary category
            categories_raw = record.get("categories") or ""
            primary_category = categories_raw.split(",")[0].strip()
            name_lower = record.get("name", "").lower().strip()

            primary_is_coffee = primary_category in {"Coffee & Tea", "Cafes", "Coffee Roasteries"}
            name_is_coffee = any(kw in name_lower for kw in {
                "coffee", "café", "cafe", "brew", "roast", "espresso",
                "tea", "latte", "boba", "starbucks", "dunkin", "dutch bros",
                "peet", "teavana", "kung fu tea",
            })

            if not (primary_is_coffee or name_is_coffee):
                continue

            # Filter: exclude non-coffee chains (convenience stores, fast food, etc.)
            if any(excl in name_lower for excl in EXCLUDED_BRANDS):
                continue

            records.append({
                "business_id":   record["business_id"],
                "name":          record["name"].strip(),
                "city":          record.get("city", ""),
                "state":         record.get("state", ""),
                "categories":    categories_raw,
                "business_avg_stars": record.get("stars"),
                "business_review_count": record.get("review_count"),
            })

    df = pd.DataFrame(records)
    print(f"  Scanned {total:,} businesses → kept {len(df):,} matching records")
    return df


def load_reviews_chunked(
    review_json_path: Union[str, Path],
    business_ids: set,
    date_start: str = "2017-01-01",
    date_end: str = "2021-12-31",
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    """
    Stream-load Yelp review records filtered by business ID and date range.

    Uses chunk-based iteration to avoid loading the full ~5 GB file into memory.
    Only rows matching the filtered business set and date window are retained.

    Parameters
    ----------
    review_json_path : str or Path
        Absolute path to yelp_academic_dataset_review.json
    business_ids : set
        Set of business_id strings to retain (from load_businesses output)
    date_start : str
        Inclusive start date in YYYY-MM-DD format
    date_end : str
        Inclusive end date in YYYY-MM-DD format
    chunk_size : int
        Number of raw lines to buffer per iteration (default: 10,000)

    Returns
    -------
    pd.DataFrame
        Filtered review records with columns:
        review_id, user_id, business_id, review_stars, date, text,
        useful, funny, cool
    """
    path = Path(review_json_path)
    print(f"Streaming reviews from: {path.name}")
    print(f"  Date window: {date_start} → {date_end}")
    print(f"  Filtering against {len(business_ids):,} business IDs")

    records = []
    buffer = []
    total_scanned = 0

    def _process_buffer(buf):
        out = []
        for line in buf:
            r = json.loads(line.strip())
            if r.get("business_id") not in business_ids:
                continue
            d = r.get("date", "")[:10]
            if not (date_start <= d <= date_end):
                continue
            out.append({
                "review_id":    r["review_id"],
                "user_id":      r["user_id"],
                "business_id":  r["business_id"],
                "review_stars": r.get("stars"),
                "date":         d,
                "text":         r.get("text", ""),
                "useful":       r.get("useful", 0),
                "funny":        r.get("funny", 0),
                "cool":         r.get("cool", 0),
            })
        return out

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            buffer.append(line)
            total_scanned += 1
            if len(buffer) >= chunk_size:
                records.extend(_process_buffer(buffer))
                buffer = []
                if total_scanned % 500_000 == 0:
                    print(f"  ... scanned {total_scanned:,} lines, "
                          f"collected {len(records):,} matching reviews")

        if buffer:
            records.extend(_process_buffer(buffer))

    df = pd.DataFrame(records)
    print(f"  Scanned {total_scanned:,} reviews → kept {len(df):,} matching records")
    return df


def load_users_subset(
    user_json_path: Union[str, Path],
    user_ids: set,
) -> pd.DataFrame:
    """
    Load a subset of Yelp user records matching the provided user ID set.

    Reads line-by-line; stops early if all target users have been found.

    Parameters
    ----------
    user_json_path : str or Path
        Absolute path to yelp_academic_dataset_user.json
    user_ids : set
        Set of user_id strings to retain

    Returns
    -------
    pd.DataFrame
        User records with columns: user_id, user_name, user_review_count,
        yelping_since, average_stars, useful, funny, cool
    """
    path = Path(user_json_path)
    print(f"Loading user profiles from: {path.name}")
    print(f"  Looking up {len(user_ids):,} user IDs")

    records = []
    remaining = set(user_ids)
    total_scanned = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            total_scanned += 1
            r = json.loads(line.strip())
            uid = r.get("user_id")
            if uid in remaining:
                records.append({
                    "user_id":           uid,
                    "user_name":         r.get("name", ""),
                    "user_review_count": r.get("review_count", 0),
                    "yelping_since":     r.get("yelping_since", ""),
                    "user_avg_stars":    r.get("average_stars"),
                    "user_useful":       r.get("useful", 0),
                    "user_funny":        r.get("funny", 0),
                    "user_cool":         r.get("cool", 0),
                })
                remaining.discard(uid)
                if not remaining:
                    break

    df = pd.DataFrame(records)
    print(f"  Scanned {total_scanned:,} user records → matched {len(df):,}")
    return df
