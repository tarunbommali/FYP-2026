import pandas as pd
import numpy as np
import os
import glob
import warnings
import gc
import joblib
from collections import Counter

warnings.filterwarnings("ignore")

DTYPE_MAP = {
    "Destination Port": np.uint16,
    "Flow Duration": np.float32,
    "Total Fwd Packets": np.float32,
    "Total Backward Packets": np.float32,
    "Total Length of Fwd Packets": np.float32,
    "Total Length of Bwd Packets": np.float32,
    "Fwd Packet Length Max": np.float32,
    "Fwd Packet Length Min": np.float32,
    "Fwd Packet Length Mean": np.float32,
    "Fwd Packet Length Std": np.float32,
    "Bwd Packet Length Max": np.float32,
    "Bwd Packet Length Min": np.float32,
    "Bwd Packet Length Mean": np.float32,
    "Bwd Packet Length Std": np.float32,
    "Flow Bytes/s": np.float32,
    "Flow Packets/s": np.float32,
    "Flow IAT Mean": np.float32,
    "Flow IAT Std": np.float32,
    "Flow IAT Max": np.float32,
    "Flow IAT Min": np.float32,
    "Fwd IAT Total": np.float32,
    "Fwd IAT Mean": np.float32,
    "Fwd IAT Std": np.float32,
    "Fwd IAT Max": np.float32,
    "Fwd IAT Min": np.float32,
    "Bwd IAT Total": np.float32,
    "Bwd IAT Mean": np.float32,
    "Bwd IAT Std": np.float32,
    "Bwd IAT Max": np.float32,
    "Bwd IAT Min": np.float32,
    "Fwd PSH Flags": np.float32,
    "Bwd PSH Flags": np.float32,
    "Fwd URG Flags": np.float32,
    "Bwd URG Flags": np.float32,
    "Fwd Header Length": np.float32,
    "Bwd Header Length": np.float32,
    "Fwd Packets/s": np.float32,
    "Bwd Packets/s": np.float32,
    "Min Packet Length": np.float32,
    "Max Packet Length": np.float32,
    "Packet Length Mean": np.float32,
    "Packet Length Std": np.float32,
    "Packet Length Variance": np.float32,
    "FIN Flag Count": np.float32,
    "SYN Flag Count": np.float32,
    "RST Flag Count": np.float32,
    "PSH Flag Count": np.float32,
    "ACK Flag Count": np.float32,
    "URG Flag Count": np.float32,
    "CWE Flag Count": np.float32,
    "ECE Flag Count": np.float32,
    "Down/Up Ratio": np.float32,
    "Average Packet Size": np.float32,
    "Avg Fwd Segment Size": np.float32,
    "Avg Bwd Segment Size": np.float32,
    "Fwd Header Length.1": np.float32,
    "Fwd Avg Bytes/Bulk": np.float32,
    "Fwd Avg Packets/Bulk": np.float32,
    "Fwd Avg Bulk Rate": np.float32,
    "Bwd Avg Bytes/Bulk": np.float32,
    "Bwd Avg Packets/Bulk": np.float32,
    "Bwd Avg Bulk Rate": np.float32,
    "Subflow Fwd Packets": np.float32,
    "Subflow Fwd Bytes": np.float32,
    "Subflow Bwd Packets": np.float32,
    "Subflow Bwd Bytes": np.float32,
    "Init_Win_bytes_forward": np.float32,
    "Init_Win_bytes_backward": np.float32,
    "act_data_pkt_fwd": np.float32,
    "min_seg_size_forward": np.float32,
    "Active Mean": np.float32,
    "Active Std": np.float32,
    "Active Max": np.float32,
    "Active Min": np.float32,
    "Idle Mean": np.float32,
    "Idle Std": np.float32,
    "Idle Max": np.float32,
    "Idle Min": np.float32,
    "Label": "str",
}

def clean_data():
    print("="*50)
    print("PHASE 1: DATA CLEANING & OPTIMIZATION")
    print("="*50)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
    
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
        
    print(f"Found {len(csv_files)} CSV files to process.")
    
    output_file = os.path.join(PROCESSED_PATH, "cicids2017_clean.csv")
    if os.path.exists(output_file):
        os.remove(output_file)
        
    header_written = False
    total_processed = 0
    total_saved = 0
    
    labels_before = Counter()
    labels_after = Counter()
    
    for i, file in enumerate(csv_files):
        if "clean" in file or "sample" in file:
            continue
            
        print(f"\n[{i+1}/{len(csv_files)}] Processing: {os.path.basename(file)}")
        
        # Read the first line to get columns, strip them, and pass as names so DTYPE_MAP works during parse
        with open(file, 'r', encoding='utf-8') as f:
            raw_cols = f.readline().strip().split(',')
            clean_cols = []
            seen = set()
            for c in raw_cols:
                c_clean = c.strip(' "\'')
                if c_clean in seen:
                    c_clean = c_clean + ".1"
                seen.add(c_clean)
                clean_cols.append(c_clean)
            
        # Max file is ~700k rows, fits easily in RAM when parsed directly as float32
        df = pd.read_csv(
            file,
            names=clean_cols,
            header=0,
            dtype=DTYPE_MAP,
            na_values=["Infinity", "-Infinity", "inf", "-inf"],
            engine="c"
        )
        

        original_len = len(df)
        total_processed += original_len
        print(f"  Loaded {original_len} rows.")
        
        # Track before cleaning
        if "Label" in df.columns:
            labels = df["Label"].astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
            labels_before.update(labels)
        
        # Replace inf with nan
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Measure missing percentage
        missing_count = df.isnull().sum().sum()
        missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
        print(f"  Missing values (NaN/Inf): {missing_count} ({missing_percentage:.2f}% of cells)")
        
        # Median imputation — preserves attack flows from zero-duration flows (PortScan/DDoS)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
        print(f"  Imputed {missing_count} missing values using median strategy.")
        
        # Drop Duplicates
        before_dup = len(df)
        df.drop_duplicates(inplace=True)
        print(f"  Dropped {before_dup - len(df)} duplicate rows.")
        
        # Clean Labels
        df["Label"] = df["Label"].astype(str).str.strip()
        df["Label"] = df["Label"].str.replace("\ufffd", "-", regex=False)
        
        # Track after cleaning
        labels_after.update(df["Label"])
        
        # Downcast any stray types just in case
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype(np.int32)
            
        # Write to disk
        df.to_csv(output_file, mode="a", index=False, header=not header_written, chunksize=10000)
        if not header_written:
            header_written = True
            
        total_saved += len(df)
        print(f"  Saved {len(df)} rows to {output_file}.")
        
        del df
        gc.collect()
        
    print("="*50)
    print("CLEANING COMPLETE")
    print(f"Total rows processed: {total_processed}")
    print(f"Total rows saved:     {total_saved}")
    print(f"Final output file:    {output_file}")
    print(f"File size:            {os.path.getsize(output_file) / 1024**2:.2f} MB")
    
    print("\nGlobal Label Distribution (Before Cleaning):")
    for label, count in labels_before.most_common():
        print(f"  {label:<20} : {count:>10,}")
        
    print("\nGlobal Label Distribution (After Cleaning):")
    for label, count in labels_after.most_common():
        print(f"  {label:<20} : {count:>10,}")
        
    print("\nLabel Loss Report")
    for label in labels_before:
        before = labels_before[label]
        after = labels_after[label]
    
        loss = before - after
        pct = (loss / before) * 100 if before > 0 else 0
    
        print(
            f"  {label:<25}"
            f"Before={before:>10,} "
            f"After={after:>10,} "
            f"Lost={loss:>8,} "
            f"({pct:.2f}%)"
        )
    # Compute and save global feature medians for inference engine imputation.
    # Reads the final output in chunks — memory-safe for 2.5M rows.
    print("\nComputing global feature medians for inference engine...")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    os.makedirs(MODELS_DIR, exist_ok=True)

    chunk_medians, chunk_sizes = [], []
    for chunk in pd.read_csv(output_file, chunksize=50_000, low_memory=False):
        num_chunk = chunk.select_dtypes(include=[np.number])
        chunk_medians.append(num_chunk.median())
        chunk_sizes.append(len(chunk))

    weights = np.array(chunk_sizes, dtype=float) / sum(chunk_sizes)
    global_medians = pd.concat(chunk_medians, axis=1).mul(weights, axis=1).sum(axis=1)
    medians_path = os.path.join(MODELS_DIR, "feature_medians.pkl")
    joblib.dump(global_medians.to_dict(), medians_path)
    print(f"  Saved feature_medians.pkl ({len(global_medians)} features)")

    print("="*50)

if __name__ == "__main__":
    clean_data()
