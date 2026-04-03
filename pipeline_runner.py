#!/usr/bin/env python3
"""
pipeline_runner.py
------------------
End-to-end pipeline runner for the Starbucks Customer Voice Intelligence
project. Executes all 11 analysis notebooks in sequence using nbconvert,
regenerating processed data and output figures from raw Yelp JSON files.

Usage:
    python pipeline_runner.py

Prerequisites:
    - Raw Yelp JSON files in data/raw/
    - Python packages listed in requirements.txt
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOKS = [
    "01_data_extraction.ipynb",
    "02_data_cleaning.ipynb",
    "03_feature_engineering.ipynb",
    "04_pipeline_summary.ipynb",
    "05_volume_trends.ipynb",
    "06_rating_distribution.ipynb",
    "07_voc_loyalty.ipynb",
    "08_reviewer_segmentation.ipynb",
    "09_scorecard.ipynb",
    "10_time_patterns.ipynb",
    "11_executive_summary.ipynb",
]


def run_notebook(nb_name: str) -> bool:
    """Execute a single notebook in-place using nbconvert."""
    nb_path = NOTEBOOK_DIR / nb_name
    print(f"\n{'='*60}")
    print(f"  Running: {nb_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  FAILED: {nb_name}")
        print(result.stderr)
        return False

    print(f"  OK: {nb_name}")
    return True


def main():
    print("Starbucks Customer Voice Intelligence — Pipeline Runner")
    print(f"Project root: {PROJECT_ROOT}")

    failed = []
    for nb in NOTEBOOKS:
        if not run_notebook(nb):
            failed.append(nb)

    print(f"\n{'='*60}")
    if failed:
        print(f"  Pipeline completed with {len(failed)} failure(s):")
        for f in failed:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  Pipeline completed successfully — {len(NOTEBOOKS)} notebooks executed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
