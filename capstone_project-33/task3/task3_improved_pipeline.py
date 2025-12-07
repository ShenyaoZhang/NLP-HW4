#!/usr/bin/env python3
"""
Task 3 IMPROVED Pipeline: Detect Rare and UNEXPECTED Drug-Event Relationships

This is the improved version that addresses feedback:
1. Filters out known/common AEs - only reports TRULY unexpected relationships
2. Includes competitor drugs (Glofitamab, Mosunetuzumab)
3. Supports interactive drug-event queries
4. Uses BERT for clinical feature analysis

Key Improvements:
- Before: Detected "anomalies" that were actually known common AEs (e.g., CRS for Epcoritamab)
- After: Only reports truly unexpected drug-event relationships by filtering against FDA labels
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from task3_data_and_model import Task3DataProcessor, Task3Model
from task3_drug_label_filter import DrugLabelFilter, KNOWN_AES_BACKUP
from task3_interactive_query import InteractiveAnomalyQuery

try:
    from config_task3 import CONFIG
except ImportError:
    CONFIG = {
        "contamination": 0.15,
        "top_k_global": 200,
        "per_drug_cap": 5,
    }


def run_improved_pipeline(filter_known_aes=True, collect_new_data=False):
    """
    Run the improved anomaly detection pipeline
    
    Args:
        filter_known_aes: If True, filter out known/common adverse events
        collect_new_data: If True, collect fresh data from OpenFDA
    
    Returns:
        DataFrame with truly unexpected drug-event relationships
    """
    print("=" * 80)
    print("Task 3 IMPROVED: Detect TRULY UNEXPECTED Drug-Event Relationships")
    print("=" * 80)
    
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 1: Data Collection (optional)
    if collect_new_data:
        print("\n[Step 1] Collecting fresh data from OpenFDA...")
        print("(This includes Glofitamab and Mosunetuzumab)")
        from task3_data_collector import main as collect_data
        collect_data()
    else:
        print("\n[Step 1] Using existing data...")
    
    # Step 2: Load and process data
    print("\n[Step 2] Processing data and engineering features...")
    data_file = os.path.join(data_dir, 'task3_oncology_drug_event_pairs.csv')
    
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        print("Please run with collect_new_data=True first")
        return None
    
    processor = Task3DataProcessor(data_file)
    X, pair_names, additional_metrics = processor.process()
    
    print(f"✓ Processed {len(pair_names)} drug-event pairs")
    print(f"✓ Features: {processor.feature_names}")
    
    # Step 3: Train Isolation Forest
    print("\n[Step 3] Training Isolation Forest model...")
    model = Task3Model(contamination=CONFIG.get('contamination', 0.15))
    model.fit(X)
    
    # Step 4: Get all anomalies
    print("\n[Step 4] Detecting anomalies...")
    results = model.get_results()
    predictions = results['predictions']
    anomaly_scores = results['anomaly_scores']
    
    # Build results DataFrame from pair_names and processor data
    results_data = []
    for i, pair_name in enumerate(pair_names):
        parts = pair_name.split('||')
        drug = parts[0] if len(parts) > 0 else ''
        adverse_event = parts[1] if len(parts) > 1 else ''
        
        # Get stats from drug_event_features
        stats = processor.drug_event_features.get(pair_name, {})
        add_metrics = additional_metrics.get(pair_name, {})
        
        count = stats.get('count', 0)
        
        # Calculate rates
        death_rate = stats.get('death_count', 0) / count if count > 0 else 0
        hosp_rate = stats.get('hosp_count', 0) / count if count > 0 else 0
        serious_rate = stats.get('serious_count', 0) / count if count > 0 else 0
        
        # Get PRR/ROR/Chi2 from feature matrix
        feature_idx = i
        if feature_idx < X.shape[0]:
            prr = float(X[feature_idx, 1]) if X.shape[1] > 1 else 0
            ror = float(X[feature_idx, 2]) if X.shape[1] > 2 else 0
            chi2 = float(X[feature_idx, 3]) if X.shape[1] > 3 else 0
        else:
            prr, ror, chi2 = 0, 0, 0
        
        results_data.append({
            'drug': drug,
            'adverse_event': adverse_event,
            'anomaly_score': -anomaly_scores[i],  # Convert to positive (higher = more anomalous)
            'is_anomaly': predictions[i] == -1,
            'count': count,
            'prr': prr,
            'ror': ror,
            'chi2': chi2,
            'ic025': add_metrics.get('ic025', 0),
            'death_rate': death_rate,
            'hosp_rate': hosp_rate,
            'serious_rate': serious_rate,
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Filter to anomalies only
    all_anomalies = results_df[results_df['is_anomaly']].copy()
    all_anomalies = all_anomalies.sort_values('anomaly_score', ascending=False)
    
    print(f"✓ Detected {len(all_anomalies)} anomalies (before filtering)")
    
    # Step 5: Filter out known/common AEs (KEY IMPROVEMENT!)
    if filter_known_aes:
        print("\n[Step 5] Filtering out KNOWN adverse events...")
        print("(Only keeping TRULY UNEXPECTED relationships)")
        
        label_filter = DrugLabelFilter()
        
        # Use backup known AEs (or fetch from FDA API)
        label_filter.known_aes = KNOWN_AES_BACKUP
        
        # Also try to load from FDA API for drugs not in backup
        unique_drugs = all_anomalies['drug'].unique()
        for drug in unique_drugs:
            if drug.lower() not in label_filter.known_aes:
                label = label_filter.get_drug_label(drug)
                if label:
                    aes = label_filter.extract_adverse_reactions(label)
                    label_filter.known_aes[drug.lower()] = aes
        
        # Filter - apply max_count_for_rare filter AND PRR override
        max_count_for_rare = CONFIG.get('max_count_for_rare', 10)
        prr_override_threshold = CONFIG.get('prr_override_threshold', 50)
        unexpected_anomalies = label_filter.filter_anomalies(
            all_anomalies, 
            max_count_for_rare=max_count_for_rare,
            prr_override_threshold=prr_override_threshold
        )
        
        print(f"\n✓ After filtering:")
        print(f"  - Known/Expected AEs removed: {len(all_anomalies) - len(unexpected_anomalies)}")
        print(f"  - TRULY RARE & UNEXPECTED AEs kept: {len(unexpected_anomalies)}")
    else:
        unexpected_anomalies = all_anomalies
    
    # Step 6: Save ALL unexpected AEs (no cap) first
    print("\n[Step 6] Saving ALL unexpected AEs...")
    
    all_unexpected_sorted = unexpected_anomalies.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    all_unexpected_sorted['rank'] = all_unexpected_sorted.index + 1
    
    # Save ALL unexpected AEs (no cap)
    all_output_file = os.path.join(data_dir, 'task3_all_unexpected_no_cap.csv')
    all_unexpected_sorted.to_csv(all_output_file, index=False)
    print(f"✓ ALL unexpected AEs saved: {len(all_unexpected_sorted)} pairs")
    print(f"  File: {all_output_file}")
    
    # Step 7: Apply Top-K and per-drug cap for summary view
    print("\n[Step 7] Applying output limits for summary view...")
    
    top_k = CONFIG.get('top_k_global', 200)
    per_drug_cap = CONFIG.get('per_drug_cap', 5)
    
    # Per-drug cap
    unexpected_anomalies['rank_by_drug'] = unexpected_anomalies.groupby('drug')['anomaly_score'].rank(
        method='first', ascending=False
    )
    capped_results = unexpected_anomalies[unexpected_anomalies['rank_by_drug'] <= per_drug_cap]
    
    # Global Top-K
    final_results = capped_results.head(top_k).copy()
    final_results = final_results.sort_values('anomaly_score', ascending=False).reset_index(drop=True)
    final_results['rank'] = final_results.index + 1
    
    print(f"✓ Capped results: {len(final_results)} unexpected drug-event pairs")
    print(f"  (per_drug_cap={per_drug_cap}, top_k={top_k})")
    
    # Save capped results
    output_file = os.path.join(data_dir, 'task3_unexpected_anomalies.csv')
    final_results.to_csv(output_file, index=False)
    print(f"✓ Capped results saved to: {output_file}")
    
    # Step 8: Display top results from capped file
    print("\n" + "=" * 80)
    print("TOP 20 TRULY UNEXPECTED Drug-Event Relationships")
    print("=" * 80)
    
    display_cols = ['rank', 'drug', 'adverse_event', 'anomaly_score', 'prr', 'count']
    available_cols = [c for c in display_cols if c in final_results.columns]
    
    for _, row in final_results.head(20).iterrows():
        print(f"\n[{row['rank']}] Anomaly Score: {row['anomaly_score']:.4f}")
        print(f"    Drug: {row['drug']}")
        print(f"    Adverse Event: {row['adverse_event']}")
        print(f"    Report Count: {row['count']}")
        print(f"    PRR: {row['prr']:.2f}")
        if 'why_flagged' in row:
            print(f"    Why Flagged: {row['why_flagged']}")
    
    return final_results


def compare_bispecific_antibodies():
    """
    Compare Epcoritamab vs Glofitamab vs Mosunetuzumab
    (As requested in feedback)
    """
    print("\n" + "=" * 80)
    print("Bispecific Antibody Comparison")
    print("(Epcoritamab vs Glofitamab vs Mosunetuzumab)")
    print("=" * 80)
    
    query = InteractiveAnomalyQuery()
    
    if query.results_df is None:
        print("No results available. Run the pipeline first.")
        return
    
    drugs = ["Epcoritamab", "Glofitamab", "Mosunetuzumab"]
    
    # Overall comparison
    print("\nOverall Anomaly Profiles:")
    comparison = query.compare_drugs(drugs)
    if comparison is not None and len(comparison) > 0:
        print(comparison.to_string(index=False))
    
    # Compare for specific AE
    print("\nCytokine Release Syndrome Comparison:")
    crs_comparison = query.compare_drugs(drugs, adverse_event="Cytokine release syndrome")
    if crs_comparison is not None and len(crs_comparison) > 0:
        print(crs_comparison.to_string(index=False))


def analyze_clinical_features(drug, adverse_event):
    """
    Analyze clinical features for a specific drug-event combination
    (As requested in feedback - using BERT)
    """
    print(f"\n" + "=" * 80)
    print(f"Clinical Feature Analysis: {drug} + {adverse_event}")
    print("=" * 80)
    
    from task3_bert_clinical_features import ClinicalFeatureExtractor
    
    extractor = ClinicalFeatureExtractor(use_bert=False)  # Set True if GPU available
    
    # Get and analyze reports
    reports = extractor.get_reports_for_drug_event(drug, adverse_event, limit=100)
    
    if reports:
        features = extractor.extract_clinical_features(reports)
        analysis = extractor.analyze_features(features)
        
        print("\nKey Clinical Features:")
        
        if 'age' in analysis:
            print(f"\n  Age: Mean={analysis['age']['mean']:.1f} years, "
                  f"Median={analysis['age']['median']:.1f} years")
        
        if 'sex' in analysis:
            print(f"  Sex: {analysis['sex']['percentages']}")
        
        if 'top_indications' in analysis:
            print(f"  Top Indications: {list(analysis['top_indications'].keys())[:5]}")
        
        # Risk factor analysis
        risk_analysis = extractor.identify_risk_factors(drug, adverse_event)
        
        print("\nRisk Factors Identified:")
        for factor, data in risk_analysis.get('risk_factors', {}).items():
            if data.get('is_risk_factor'):
                print(f"  - {factor}: {data.get('interpretation')}")
    else:
        print("No reports found for this combination.")


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Task 3 IMPROVED: Detect TRULY UNEXPECTED Drug-Event Relationships"
    )
    parser.add_argument(
        '--collect-data', 
        action='store_true',
        help='Collect fresh data from OpenFDA (includes new competitor drugs)'
    )
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Skip filtering known AEs (not recommended)'
    )
    parser.add_argument(
        '--compare-drugs',
        action='store_true',
        help='Compare bispecific antibodies (Epcoritamab, Glofitamab, Mosunetuzumab)'
    )
    parser.add_argument(
        '--analyze',
        nargs=2,
        metavar=('DRUG', 'AE'),
        help='Analyze clinical features for a specific drug-event pair'
    )
    
    args = parser.parse_args()
    
    # Run main pipeline
    results = run_improved_pipeline(
        filter_known_aes=not args.no_filter,
        collect_new_data=args.collect_data
    )
    
    # Optional: Compare drugs
    if args.compare_drugs:
        compare_bispecific_antibodies()
    
    # Optional: Analyze specific drug-event
    if args.analyze:
        analyze_clinical_features(args.analyze[0], args.analyze[1])
    
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print("\nKey Improvements Made:")
    print("  ✓ Filters out known/common AEs from drug labels")
    print("  ✓ Only reports TRULY UNEXPECTED relationships")
    print("  ✓ Includes competitor drugs (Glofitamab, Mosunetuzumab)")
    print("  ✓ Supports interactive queries")
    print("  ✓ Clinical feature analysis with BERT")
    
    return results


if __name__ == "__main__":
    main()

