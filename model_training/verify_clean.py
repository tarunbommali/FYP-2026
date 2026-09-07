"""
verify_clean.py
Post-Phase-1 verification gate — run BEFORE 02_binary_training.py.

Hard failures  : NaN count > 0, Inf count > 0
Soft warnings  : rare classes, residual cross-file duplicates
Informational  : shape, memory, imbalance ratio, label preservation
"""

import pandas as pd
import numpy as np
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_FILE = os.path.join(BASE_DIR, "data", "processed", "cicids2017_clean.csv")

# Known rare classes in CICIDS2017 — NOT cleaning failures
KNOWN_RARE = {"Heartbleed", "Web Attack - Sql Injection", "Infiltration"}

# Acceptable row-count window after dedup (based on Phase 0 findings)
MIN_EXPECTED_ROWS = 2_500_000   # lowered: per-file dedup removes ~9%
MAX_EXPECTED_ROWS = 2_830_743

print("=" * 60)
print("POST-CLEAN VERIFICATION GATE")
print("=" * 60)

if not os.path.exists(CLEAN_FILE):
    print(f"[ERROR] Clean file not found:\n  {CLEAN_FILE}")
    raise SystemExit(1)

print(f"Loading: {CLEAN_FILE}\n")
df = pd.read_csv(CLEAN_FILE, low_memory=False)

gate_passed = True  # tracks hard failures only

# -- 1. Dataset Size ----------------------------------------------------------
row_count = len(df)
size_ok   = MIN_EXPECTED_ROWS <= row_count <= MAX_EXPECTED_ROWS
print("[1] Dataset Size")
print(f"    Rows    : {row_count:,}")
print(f"    Columns : {df.shape[1]}")
if size_ok:
    print(f"    PASS  Within expected range ({MIN_EXPECTED_ROWS:,} - {MAX_EXPECTED_ROWS:,})")
else:
    print(f"    WARN  OUTSIDE expected range ({MIN_EXPECTED_ROWS:,} - {MAX_EXPECTED_ROWS:,})")
    print("          Investigate cleaning pipeline if loss seems excessive.")

# -- 2. NaN Check (HARD GATE) -------------------------------------------------
nan_total = df.isnull().sum().sum()
print(f"\n[2] NaN Count")
print(f"    Total NaN : {nan_total}")
if nan_total == 0:
    print("    PASS  Median imputation complete.")
else:
    print("    FAIL  Imputation incomplete. Do NOT proceed to training.")
    gate_passed = False

# -- 3. Infinity Check (HARD GATE) --------------------------------------------
inf_total = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print(f"\n[3] Inf Values")
print(f"    Total Inf : {inf_total}")
if inf_total == 0:
    print("    PASS")
else:
    print("    FAIL  Inf values remain. Check na_values in read_csv.")
    gate_passed = False

# -- 4. Duplicate Check (INFORMATIONAL, percentage-based) --------------------
dup_total  = df.duplicated().sum()
dup_pct    = (dup_total / row_count * 100) if row_count else 0
print(f"\n[4] Duplicate Count")
print(f"    Total duplicates : {dup_total:,}  ({dup_pct:.3f}% of cleaned rows)")
if dup_pct == 0:
    print("    PASS  Fully deduplicated.")
elif dup_pct < 0.5:
    print("    PASS  Excellent (<0.5%) — residual cross-file duplicates only.")
elif dup_pct < 2.0:
    print("    INFO  Acceptable (0.5-2%). Phase 1 deduplicates per-file only.")
elif dup_pct < 5.0:
    print("    WARN  Moderate (2-5%). Consider global dedup on merged file.")
else:
    print("    WARN  High (>5%). Likely global duplicate issue — investigate.")

# -- 5. Unique Label Count ----------------------------------------------------
unique_labels   = df["Label"].nunique()
EXPECTED_LABELS = 15
print(f"\n[5] Unique Label Count")
print(f"    Unique labels : {unique_labels}  (expected {EXPECTED_LABELS})")
if unique_labels == EXPECTED_LABELS:
    print("    PASS  All 15 CICIDS2017 attack classes present.")
elif unique_labels > EXPECTED_LABELS:
    print("    WARN  More labels than expected. Check for label noise.")
else:
    missing_n = EXPECTED_LABELS - unique_labels
    print(f"    WARN  {missing_n} class(es) missing. Some attack types were lost in cleaning.")

# -- 6. Memory Usage ----------------------------------------------------------
memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
print(f"\n[6] Memory Usage")
print(f"    {memory_mb:.2f} MB")
if memory_mb < 1_200:
    print("    PASS  dtype optimizations effective (expected 700-1200 MB)")
else:
    print("    WARN  Higher than expected. Check for float64/object columns.")

# -- 7. Label Distribution + Class Balance ------------------------------------
print(f"\n[7] Label Distribution (after cleaning)")
dist   = df["Label"].value_counts()
total  = len(df)
benign = dist.get("BENIGN", 0)
attack = sum(v for k, v in dist.items() if k != "BENIGN")
ratio  = benign / attack if attack else float("inf")

for label, count in dist.items():
    bar  = "#" * int((count / total) * 40)
    note = " [RARE - known CICIDS2017 limitation]" if count < 100 else ""
    print(f"    {label:<35} : {count:>9,}  ({count/total*100:5.2f}%)  {bar}{note}")

print(f"\n[8] Class Imbalance")
print(f"    BENIGN  : {benign:,}  ({benign/total*100:.2f}%)")
print(f"    ATTACK  : {attack:,}  ({attack/total*100:.2f}%)")
print(f"    Ratio   : {ratio:.2f}:1  (BENIGN : ATTACK)")
if 3.5 <= ratio <= 5.0:
    print("    PASS  Expected range for CICIDS2017 (~4:1)")
else:
    print("    WARN  Ratio outside expected range. Verify label cleaning.")

# -- 8. Critical Rare Class Preservation --------------------------------------
print(f"\n[9] Critical Rare Class Preservation")
critical_missing = []
for cls in KNOWN_RARE:
    count = dist.get(cls, 0)
    if count == 0:
        print(f"    MISSING  {cls:<35} (cleaning removed all samples)")
        critical_missing.append(cls)
    else:
        print(f"    OK       {cls:<35} : {count} samples preserved")

if critical_missing:
    print("\n    WARNING: Rare attack classes were lost. Investigate cleaning.")

# -- Gate Verdict -------------------------------------------------------------
print("\n" + "=" * 60)
if gate_passed:
    print("GATE PASSED -- Safe to proceed to Phase 2 (02_binary_training.py)")
    print("Next: apply locked fixes to 02_binary_training.py and run.")
else:
    print("GATE FAILED -- Resolve hard failures above before training.")
print("=" * 60)
