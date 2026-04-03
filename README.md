# Starbucks on Yelp: Sentiment and Review Analysis

381,999 Yelp reviews from 8,203 coffee-chain locations, 13 U.S. states, 2017–2021. The focus is Starbucks, compared against Dunkin' and independent cafes.

**[View the interactive report](https://ww532.github.io/coffee-chain-intelligence/)**

## What this project does

- VADER sentiment scoring on all reviews
- Multi-label topic tagging: Service, Food Quality, Ambiance, Wait Time, Price, Cleanliness
- Reviewer segmentation into four tiers by Yelp activity
- Brand comparison across Starbucks, Dunkin', and independent cafes
- Time patterns by day of week and month

## Key numbers

| Metric | Starbucks | Dunkin' | Independent |
|--------|-----------|---------|-------------|
| Reviews | 11,675 | 6,866 | 362,334 |
| Avg Stars | 2.90 | 2.05 | 4.01 |
| % Positive | 42.4% | 21.5% | 73.7% |
| % Critical | 47.6% | 71.2% | 17.2% |

- 1-star is Starbucks' largest rating group at 33.7%, ahead of 5-star at 29.0%
- Service comes up in 75.6% of reviews; 44.4% positive, 46.0% critical
- Weekends average 2.71 stars vs. 2.99 on weekdays
- Casual reviewers (under 10 reviews) rate 2.43 stars; Elite (200+) rate 3.60

## Dataset

[Yelp Open Dataset](https://www.yelp.com/dataset). Not included in this repo. Download the JSON files and put them in `data/raw/`.

## Project structure

```
notebooks/
  01_data_extraction.ipynb      Extract and filter from Yelp JSON
  02_data_cleaning.ipynb        Deduplicate, normalize, join
  03_feature_engineering.ipynb   Brand categories, sentiment, topic tags
  04_pipeline_summary.ipynb     Validation and data dictionary
  05_volume_trends.ipynb        Annual, quarterly, and state-level volume
  06_rating_distribution.ipynb  Star ratings, tiers, sentiment trends
  07_voc_loyalty.ipynb          Topic analysis and word clouds
  08_reviewer_segmentation.ipynb  Rating by reviewer activity tier
  09_scorecard.ipynb            Brand benchmarking across topics
  10_time_patterns.ipynb        Day-of-week and monthly patterns
  11_executive_summary.ipynb    Summary metrics and scorecard

src/
  ingestion.py                  Data loading utilities
  cleaning.py                   Dedup, null handling, normalization
  mappings.py                   Brand and topic category mappings
  nlp_utils.py                  VADER scoring, topic tagging

outputs/figures/
  index.html                    Interactive report (GitHub Pages)
  styles.css                    Compiled Tailwind CSS
  *.html                        Plotly chart files
```

## How to run

```bash
pip install -r requirements.txt
python pipeline_runner.py        # runs notebooks 01-04
# then open notebooks 05-11 individually
```

Requires Python 3.9+ and the Yelp dataset JSON files in `data/raw/`.

## Tools

Python, pandas, VADER (vaderSentiment), Plotly, WordCloud, Tailwind CSS v4

## Author

Xinwei Wang
