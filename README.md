# Starbucks Customer Voice Intelligence

A data-driven analysis of Starbucks customer reviews on Yelp, benchmarked against Dunkin' and the independent café market across the United States.

## Dataset

**Source:** [Yelp Open Dataset](https://www.yelp.com/dataset)

| Dimension | Value |
|-----------|-------|
| Reviews | 381,999 |
| Businesses | 8,203 coffee-focused locations |
| Time window | January 2017 – December 2021 |
| Geography | 13 U.S. states, 500+ cities |
| Primary subject | Starbucks (11,675 reviews) |
| Direct competitor | Dunkin' (6,866 reviews) |
| Market benchmark | Independent cafés (362,334 reviews) |

## Key Findings

**Starbucks averages 2.90 stars — 1.11 below the market benchmark of 4.01.** However, Dunkin' performs worse at 2.05 stars with a 71.2% critical review rate. The gap is a chain-format issue, not unique to Starbucks.

**Service is the dominant variable.** 75.6% of Starbucks reviews mention service. With a 44.4% positive rate against a 46.0% failure rate, front-line staff execution is the single largest determinant of customer outcome.

**Weekend demand degrades quality.** Saturday and Sunday produce the lowest average ratings (2.71 stars) despite the highest review volumes. The weekend critical rate reaches 53.2% versus 45.0% on weekdays.

**64% of negative signal comes from casual reviewers.** Reviewers with fewer than 10 lifetime Yelp reviews average 2.43 stars. Elite reviewers (200+) average 3.60 stars — a 1.17-star gap larger than the Starbucks-to-market gap.

## Project Structure

```
coffee-chain-intelligence/
├── data/
│   ├── raw/                      # Yelp JSON files (not tracked)
│   └── processed/                # Parquet intermediates (not tracked)
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_pipeline_summary.ipynb
│   ├── 05_volume_trends.ipynb
│   ├── 06_rating_distribution.ipynb
│   ├── 07_voc_loyalty.ipynb
│   ├── 08_reviewer_segmentation.ipynb
│   ├── 09_scorecard.ipynb
│   ├── 10_time_patterns.ipynb
│   └── 11_executive_summary.ipynb
├── src/
│   ├── ingestion.py              # Data loading and filtering
│   ├── cleaning.py               # Deduplication, joins, type casting
│   ├── mappings.py               # Feature engineering mappings
│   └── nlp_utils.py              # VADER sentiment, TF-IDF extraction
├── outputs/
│   └── figures/                  # HTML chart outputs (not tracked)
├── pipeline_runner.py            # Run all notebooks end-to-end
├── requirements.txt
└── .gitignore
```

## Methodology

1. **Ingestion** — Filter Yelp businesses by coffee-related categories and name keywords. Stream-load reviews within the 2017–2021 date window.
2. **Cleaning** — Deduplicate, cast types, normalize city names, join reviews with business and user metadata.
3. **Feature engineering** — Map star tiers, brand categories, topic tags (rule-based keyword classification), user activity tiers, and datetime dimensions. Score review sentiment using VADER.
4. **Analysis** — Volume trends, rating distribution, voice-of-customer topic analysis, reviewer segmentation, brand benchmarking scorecard, and time pattern analysis.

## How to Run

**Prerequisites:** Python 3.9+, Yelp Open Dataset JSON files in `data/raw/`.

```bash
pip install -r requirements.txt

# Run all notebooks sequentially
python pipeline_runner.py

# Or run individual notebooks
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_extraction.ipynb
```

## Tools and Libraries

- **pandas / pyarrow** — Data manipulation and Parquet I/O
- **VADER (vaderSentiment)** — Lexicon-based sentiment scoring
- **scikit-learn** — TF-IDF keyword extraction
- **Plotly** — Interactive HTML chart generation
- **matplotlib / WordCloud** — Word cloud visualization
