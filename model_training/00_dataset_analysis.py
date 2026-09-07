import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

def analyze_dataset():
    print("="*50)
    print("PHASE 0: DATASET ANALYSIS")
    print("="*50)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_PATH = os.path.join(BASE_DIR, "data", "raw")
    
    csv_files = glob.glob(os.path.join(RAW_PATH, "*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(BASE_DIR, "data", "*.csv"))
        
    print(f"Found {len(csv_files)} CSV files for analysis.")
    
    total_samples = 0
    missing_values = 0
    total_cells = 0
    duplicate_count = 0
    feature_count = 0
    
    label_distribution = Counter()
    
    for i, file in enumerate(csv_files):
        if "clean" in file or "sample" in file:
            continue
            
        print(f"  Scanning: {os.path.basename(file)}...")
        
        # Read in chunks to prevent memory error during analysis
        reader = pd.read_csv(
            file, 
            low_memory=False, 
            na_values=["Infinity", "-Infinity", "inf", "-inf"],
            chunksize=50000
        )
        
        file_samples = 0
        file_duplicates = 0
        
        for chunk in reader:
            chunk.columns = chunk.columns.str.strip()
            if feature_count == 0:
                feature_count = chunk.shape[1]
                
            # Missing values
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            missing_values += chunk.isnull().sum().sum()
            total_cells += (chunk.shape[0] * chunk.shape[1])
            
            # Duplicates (approximate within chunk)
            file_duplicates += chunk.duplicated().sum()
            
            # Labels
            if "Label" in chunk.columns:
                labels = chunk["Label"].astype(str).str.strip().str.replace("\ufffd", "-", regex=False)
                label_distribution.update(labels)
                
            file_samples += len(chunk)
            total_samples += len(chunk)
            
        duplicate_count += file_duplicates
        
    benign_count = label_distribution.get("BENIGN", 0)
    attack_count = sum(
        count
        for label, count in label_distribution.items()
        if label != "BENIGN"
    )
    
    print("\n" + "="*50)
    print("DATASET ANALYSIS REPORT")
    print("="*50)
    print(f"Total Samples:   {total_samples:,}")
    print(f"Feature Count:   {feature_count}")
    print(f"Missing Values:  {missing_values:,} ({(missing_values/total_cells)*100 if total_cells else 0:.4f}% of total cells)")
    print(f"Est. Duplicates: {duplicate_count:,} ({(duplicate_count/total_samples)*100 if total_samples else 0:.2f}% of total samples)")
    print(f"\nBenign Count:    {benign_count:,} ({(benign_count/total_samples)*100 if total_samples else 0:.2f}%)")
    print(f"Attack Count:    {attack_count:,} ({(attack_count/total_samples)*100 if total_samples else 0:.2f}%)")
    imbalance_ratio = benign_count / attack_count if attack_count else float("inf")
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1  (BENIGN : ATTACK)")
    
    print("\nFull Class Distribution:")
    for label, count in label_distribution.most_common():
        print(f"  {label:<20} : {count:>10,} ({(count/total_samples)*100:.2f}%)")
    
    print("="*50)

if __name__ == "__main__":
    analyze_dataset()
