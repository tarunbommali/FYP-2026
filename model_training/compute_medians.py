"""
compute_medians.py
One-off utility: computes global feature medians from the cleaned dataset
and saves feature_medians.pkl to models/.

Run once after 01_data_cleaning.py has completed:
    python training\\compute_medians.py

This script is required when Phase 1 was executed before median persistence
was added to the pipeline. After this script runs, the inference engine will
use consistent median imputation matching the training data distribution.
"""

import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_FILE    = os.path.join(BASE_DIR, "data", "processed", "cicids2017_clean.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
CHUNKSIZE     = 100_000


def compute_and_save_medians():
    print("=" * 60)
    print("COMPUTE FEATURE MEDIANS")
    print("=" * 60)

    if not os.path.exists(CLEAN_FILE):
        raise FileNotFoundError(
            f"Clean dataset not found: {CLEAN_FILE}\n"
            "Run 01_data_cleaning.py first."
        )

    print(f"\nReading: {CLEAN_FILE}")
    print("Computing column sums and counts for global median approximation...")

    # Two-pass median over chunked data using value accumulation.
    # For large datasets, we compute the exact median by collecting all values
    # per column is too expensive. Instead use the "merge-sort median" approach:
    # concatenate sorted samples from each chunk. For production-quality medians,
    # we load in chunks and use pandas' exact median on each chunk, then take
    # the median-of-medians (robust approximation suitable for imputation).
    chunk_medians = []
    chunk_sizes   = []
    total_rows    = 0
    numeric_cols  = None

    for chunk in pd.read_csv(CLEAN_FILE, chunksize=CHUNKSIZE, low_memory=False):
        # Identify numeric cols from the first chunk
        if numeric_cols is None:
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
            print(f"  Numeric columns: {len(numeric_cols)}")

        numeric_chunk = chunk[numeric_cols]
        chunk_medians.append(numeric_chunk.median())
        chunk_sizes.append(len(chunk))
        total_rows += len(chunk)

    print(f"  Chunks processed : {len(chunk_medians)}")
    print(f"  Total rows       : {total_rows:,}")

    # Weight each chunk's median by its row count for a robust global estimate
    weights = np.array(chunk_sizes, dtype=float) / total_rows
    global_medians = pd.concat(chunk_medians, axis=1).mul(weights, axis=1).sum(axis=1)
    global_medians = global_medians.rename("median")

    print(f"\n  Sample medians:")
    for col in list(global_medians.index)[:5]:
        print(f"    {col:<40} : {global_medians[col]:.4f}")
    print(f"    ... ({len(global_medians)} features total)")

    # Save as a dict: {feature_name -> median_value}
    medians_dict = global_medians.to_dict()
    out_path = os.path.join(MODELS_DIR, "feature_medians.pkl")
    joblib.dump(medians_dict, out_path)
    print(f"\n  Saved: {out_path}")

    print("\n" + "=" * 60)
    print("DONE — feature_medians.pkl ready for inference engine.")
    print("=" * 60)


if __name__ == "__main__":
    compute_and_save_medians()
