#!/usr/bin/env python3
"""
Task 5 - Step 2/9: Data Inspection

Inspect the quality of the extracted data in detail
"""

import sys
import pandas as pd
import os


# Preferred data files (Task 5 > consolidated > multi-drug > single-drug)
DATA_FILES = ["main_data.csv", "oncology_drugs_complete.csv", 
              "oncology_drugs_data.csv", "epcoritamab_data.csv"]
DATA_FILE = None

for f in DATA_FILES:
    if os.path.exists(f):
        DATA_FILE = f
        break

if DATA_FILE is None:
    print(f"❌ Error: data file not found")
    print()
    print("Please run: python 01_extract_data.py")
    sys.exit(1)

print(f"✅ Data source: {DATA_FILE}")
if DATA_FILE == "main_data.csv":
    print("   (Task 5 dataset - 35 oncology drugs)")
elif DATA_FILE == "oncology_drugs_complete.csv":
    print("   (Complete oncology dataset)")
elif DATA_FILE == "oncology_drugs_data.csv":
    print("   (Multi-drug oncology dataset)")
else:
    print("   (Single-drug dataset)")
print()

# Load dataset
print("📂 Loading data...")
df = pd.read_csv(DATA_FILE)
print(f"✅ Data loaded successfully")
print()

# Basic info
print("=" * 80)
print("📊 Dataset overview")
print("=" * 80)
print()

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print()

print("Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")
print()

# Data quality checks
print("=" * 80)
print("🔍 Data quality checks")
print("=" * 80)
print()

# ✅ Duplicate safety report check
if 'safetyreportid' in df.columns:
    duplicates = df['safetyreportid'].duplicated().sum()
    print(f"Duplicate report check:")
    print(f"  Duplicates: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        print(f"  ⚠️  Warning: duplicate reports detected; deduplicate during preprocessing")
        # Show sample duplicate IDs
        dup_ids = df[df['safetyreportid'].duplicated(keep=False)]['safetyreportid'].unique()[:5]
        print(f"  Example duplicate IDs: {', '.join(map(str, dup_ids))}")
    else:
        print(f"  ✅ No duplicate reports")
    print()

# Missing values
print("Missing value summary:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Missing %': missing_pct.values
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

if len(missing_df) > 0:
    print(missing_df.head(10).to_string(index=False))
else:
    print("  ✅ No missing values")
print()

# ✅ Drug distribution (auto-detect column)
drug_col = None
for col in ['target_drug', 'drug_name', 'drugname']:
    if col in df.columns:
        drug_col = col
        break

if drug_col:
    print("=" * 80)
    print("💊 Drug distribution")
    print("=" * 80)
    print()
    
    drug_counts = df[drug_col].value_counts()
    print(f"Using column: {drug_col}")
    print(f"Distinct drugs: {len(drug_counts)}")
    print()
    print("Top 10 drugs:")
    for drug, count in drug_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"  {drug:20s}: {count:5d} ({pct:5.1f}%)")
    print()

# ✅ Date range audit
if 'receivedate' in df.columns:
    print("=" * 80)
    print("📅 Report date audit")
    print("=" * 80)
    print()
    
    try:
        df['receivedate_parsed'] = pd.to_datetime(df['receivedate'], format='%Y%m%d', errors='coerce')
        
        valid_dates = df['receivedate_parsed'].dropna()
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            date_span = (max_date - min_date).days
            
            print(f"Date range: {min_date.date()} to {max_date.date()}")
            print(f"Span: {date_span} days ({date_span/365:.1f}  years)")
            print(f"Valid dates: {len(valid_dates)} / {len(df)} ({len(valid_dates)/len(df)*100:.1f}%)")
            
            # Distribution by year
            df['year'] = df['receivedate_parsed'].dt.year
            print("\nAnnual distribution:")
            year_counts = df['year'].value_counts().sort_index()
            for year, count in year_counts.head(10).items():
                if pd.notna(year):
                    print(f"  {int(year)}: {count:5d} reports")
        else:
            print("⚠️  No valid date values")
    except Exception as e:
        print(f"⚠️  Date parsing failed: {str(e)[:50]}")
    
    print()

# Severity analysis
print("=" * 80)
print("🏥 Severity analysis")
print("=" * 80)
print()

severity_fields = {
    'serious': 'Serious events',
    'seriousnessdeath': 'Death',
    'seriousnesshospitalization': 'Hospitalization',
    'seriousnesslifethreatening': 'Life-threatening',
    'seriousnessdisabling': 'Disability',
    'seriousnesscongenitalanomali': 'Congenital anomaly',
    'seriousnessother': 'Other serious'
}

for field, label in severity_fields.items():
    if field in df.columns:
        count = pd.to_numeric(df[field], errors='coerce').fillna(0).sum()
        pct = (count / len(df)) * 100
        print(f"{label:12s}: {int(count):4d} cases ({pct:5.1f}%)")

# ✅ Severity consistency check
print()
if 'serious' in df.columns:
    print("Severity consistency check:")
    
    # Compute whether any sub-flag is positive
    sub_serious_cols = ['seriousnessdeath', 'seriousnesshospitalization', 
                       'seriousnesslifethreatening', 'seriousnessdisabling',
                       'seriousnesscongenitalanomali', 'seriousnessother']
    
    any_sub_serious = pd.Series([False] * len(df))
    for col in sub_serious_cols:
        if col in df.columns:
            any_sub_serious |= (pd.to_numeric(df[col], errors='coerce').fillna(0) > 0)
    
    # Compare with overall serious flag
    serious_flag = (pd.to_numeric(df['serious'], errors='coerce').fillna(0) > 0)
    inconsistent = (serious_flag != any_sub_serious).sum()
    
    print(f"  Inconsistent records: {inconsistent} ({inconsistent/len(df)*100:.2f}%)")
    if inconsistent > 0:
        print(f"  ⚠️  Note: 'serious' flag does not fully match sub-flags")
    else:
        print(f"  ✅ Flags are consistent")

print()

# Patient demographics
print("=" * 80)
print("👥 Patient demographics")
print("=" * 80)
print()

# Sex distribution
if 'patientsex' in df.columns:
    print("Sex distribution:")
    sex_map = {1: 'Male', 2: 'Female', 0: 'Unknown'}
    sex_counts = df['patientsex'].value_counts()
    for sex_code, count in sex_counts.items():
        label = sex_map.get(sex_code, f'Code{sex_code}')
        pct = (count / len(df)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print()

# Age statistics
if 'patientonsetage' in df.columns:
    age_data = pd.to_numeric(df['patientonsetage'], errors='coerce').dropna()
    if len(age_data) > 0:
        print("Age statistics:")
        print(f"  Count: {len(age_data)}")
        print(f"  Mean age: {age_data.mean():.1f}")
        print(f"  Median: {age_data.median():.1f}")
        print(f"  Min: {age_data.min():.1f}")
        print(f"  Max: {age_data.max():.1f}")
        print()

# Drug usage metrics
if 'num_drugs' in df.columns:
    print("Drug usage stats:")
    print(f"  Average drugs: {df['num_drugs'].mean():.1f}")
    print(f"  Max drugs: {df['num_drugs'].max()}")
    polypharmacy = (df['num_drugs'] > 1).sum()
    print(f"  Polypharmacy (>1 drug): {polypharmacy} ({polypharmacy/len(df)*100:.1f}%)")
    print()

# Data sample
print("=" * 80)
print("📋 Data sample")
print("=" * 80)
print()
print(df.head(5))
print()

# Summary
print("=" * 80)
print("✅ Step 2 complete - data inspection finished")
print("=" * 80)
print()

print("✅ Data quality assessment:")
if len(df) >= 1000:
    print("    🌟 Excellent: strong sample size for modelling")
elif len(df) >= 500:
    print("    ✅ Good: adequate sample size for modelling")
else:
    print("  ⚠️  Data volume is small but still usable for preliminary analysis")
print()

print("🎯 Next steps:")
print("  Run: python 03_preprocess_data.py")
print("  Purpose: data preprocessing and feature engineering")
print()

