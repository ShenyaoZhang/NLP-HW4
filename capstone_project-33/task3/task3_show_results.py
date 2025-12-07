#!/usr/bin/env python3
"""
Task 3 Results Display Script
Shows all unexpected drug-event relationships detected by the pipeline.

Usage:
    python task3_show_results.py              # Show results with per-drug cap (185)
    python task3_show_results.py --all        # Show ALL results without cap (1707)
    python task3_show_results.py --drug Epcoritamab  # Show specific drug
"""

import os
import sys
import argparse
import pandas as pd

def get_data_path():
    """Get the path to the data directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, '..', 'data')

def load_results(use_all=False):
    """Load the results CSV file"""
    data_dir = get_data_path()
    
    if use_all:
        file_path = os.path.join(data_dir, 'task3_all_unexpected_no_cap.csv')
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            print("    Run task3_improved_pipeline.py first to generate results.")
            return None
    else:
        file_path = os.path.join(data_dir, 'task3_unexpected_anomalies.csv')
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            print("    Run task3_improved_pipeline.py first to generate results.")
            return None
    
    return pd.read_csv(file_path)

def show_summary(df):
    """Show summary statistics"""
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\nTotal Unexpected AEs: {len(df)}")
    print(f"Total Drugs: {df['drug'].nunique()}")
    print(f"Average Count: {df['count'].mean():.1f}")
    print(f"Max Count: {int(df['count'].max())}")
    print(f"Min Count: {int(df['count'].min())}")
    
    print("\n📊 Count Distribution:")
    print(f"   Count = 1:     {len(df[df['count'] == 1])} events")
    print(f"   Count 2-5:     {len(df[(df['count'] >= 2) & (df['count'] <= 5)])} events")
    print(f"   Count 6-10:    {len(df[(df['count'] >= 6) & (df['count'] <= 10)])} events")
    if len(df[df['count'] > 10]) > 0:
        print(f"   Count > 10:    {len(df[df['count'] > 10])} events")

def show_all_drugs(df):
    """Show all drugs and their unexpected AEs"""
    print("\n" + "=" * 90)
    print("ALL UNEXPECTED DRUG-EVENT RELATIONSHIPS")
    print("=" * 90)
    
    for drug in sorted(df['drug'].unique()):
        drug_df = df[df['drug'] == drug].sort_values('anomaly_score', ascending=False)
        print(f"\n🔬 {drug.upper()} ({len(drug_df)} AEs)")
        print("-" * 70)
        
        for i, (_, row) in enumerate(drug_df.iterrows(), 1):
            ae = row['adverse_event']
            count = int(row['count'])
            prr = row['prr']
            score = row['anomaly_score']
            print(f"  {i:3}. {ae:<50} Count={count:<3} PRR={prr:.1f}")

def show_specific_drug(df, drug_name):
    """Show results for a specific drug"""
    # Case-insensitive search
    drug_df = df[df['drug'].str.lower() == drug_name.lower()]
    
    if len(drug_df) == 0:
        print(f"\n⚠️  Drug '{drug_name}' not found in results.")
        print("\nAvailable drugs:")
        for d in sorted(df['drug'].unique()):
            print(f"  - {d}")
        return
    
    drug_df = drug_df.sort_values('anomaly_score', ascending=False)
    actual_name = drug_df['drug'].iloc[0]
    
    print("\n" + "=" * 90)
    print(f"🔬 {actual_name.upper()} - ALL UNEXPECTED AEs ({len(drug_df)} total)")
    print("=" * 90)
    print(f"\n{'#':<4} {'Adverse Event':<50} {'Count':<7} {'Score':<8} {'PRR':<8}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(drug_df.iterrows(), 1):
        ae = row['adverse_event'][:49]
        count = int(row['count'])
        prr = row['prr']
        score = row['anomaly_score']
        print(f"{i:<4} {ae:<50} {count:<7} {score:.4f}  {prr:.1f}")

def show_top_n(df, n=20):
    """Show top N unexpected AEs across all drugs"""
    print("\n" + "=" * 90)
    print(f"TOP {n} UNEXPECTED DRUG-EVENT RELATIONSHIPS")
    print("=" * 90)
    
    top_df = df.sort_values('anomaly_score', ascending=False).head(n)
    
    print(f"\n{'#':<4} {'Drug':<20} {'Adverse Event':<40} {'Count':<7} {'PRR':<8}")
    print("-" * 90)
    
    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        drug = row['drug'][:19]
        ae = row['adverse_event'][:39]
        count = int(row['count'])
        prr = row['prr']
        print(f"{i:<4} {drug:<20} {ae:<40} {count:<7} {prr:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Display Task 3 Unexpected AE Results')
    parser.add_argument('--all', action='store_true', 
                        help='Show ALL results without per-drug cap (1707 vs 185)')
    parser.add_argument('--drug', type=str, default=None,
                        help='Show results for a specific drug')
    parser.add_argument('--top', type=int, default=None,
                        help='Show top N results only')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary statistics only')
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(use_all=args.all)
    if df is None:
        return
    
    # Show header
    print("\n" + "=" * 90)
    print("TASK 3: RARE & UNEXPECTED DRUG-EVENT RELATIONSHIPS")
    if args.all:
        print("(ALL results - no per-drug cap)")
    else:
        print("(With per-drug cap = 5)")
    print("=" * 90)
    
    # Show results based on arguments
    if args.summary:
        show_summary(df)
    elif args.drug:
        show_specific_drug(df, args.drug)
    elif args.top:
        show_top_n(df, args.top)
    else:
        show_summary(df)
        show_all_drugs(df)
    
    print("\n" + "=" * 90)
    print("Done!")
    print("=" * 90)

if __name__ == '__main__':
    main()

