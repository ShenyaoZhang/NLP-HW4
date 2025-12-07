#!/usr/bin/env python3
"""
Task 3 Improvement: Interactive Query System
Allows users to select and query specific drug-adverse event combinations

Key Features:
1. Query specific drug-event pairs
2. Get top anomalies for a specific drug
3. Compare drugs (e.g., Epcoritamab vs Glofitamab vs Mosunetuzumab)
4. Filter by adverse event type
5. Check ANY drug-AE combo from raw data (is it rare/unexpected?)
"""

import pandas as pd
import numpy as np
import os
from joblib import load
from scipy import stats

class InteractiveAnomalyQuery:
    """
    Interactive query system for drug-event anomaly detection results
    """
    
    def __init__(self, results_file=None, model_dir=None):
        """
        Initialize with results file and optional model for re-scoring
        """
        # Default paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'data')
        
        if results_file is None:
            # Try different possible result files (prefer full results without cap)
            possible_files = [
                'task3_all_unexpected_no_cap.csv',  # Full results - preferred
                'task3_unexpected_anomalies.csv',
                'task3_ml_isolation_forest_results.csv'
            ]
            for fname in possible_files:
                candidate = os.path.join(data_dir, fname)
                if os.path.exists(candidate):
                    results_file = candidate
                    break
            else:
                results_file = os.path.join(data_dir, 'task3_all_unexpected_no_cap.csv')
        
        if model_dir is None:
            model_dir = os.path.join(data_dir, 'models')
        
        # Load results
        if os.path.exists(results_file):
            self.results_df = pd.read_csv(results_file)
            # Standardize column names (handle both 'event' and 'adverse_event')
            if 'event' in self.results_df.columns and 'adverse_event' not in self.results_df.columns:
                self.results_df = self.results_df.rename(columns={'event': 'adverse_event'})
            print(f"✓ Loaded {len(self.results_df)} anomaly results from {os.path.basename(results_file)}")
        else:
            print(f"✗ Results file not found: {results_file}")
            self.results_df = None
        
        # Try to load model (optional)
        self.model = None
        self.scaler = None
        model_path = os.path.join(model_dir, 'task3_if_model.joblib')
        scaler_path = os.path.join(model_dir, 'task3_scaler.joblib')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = load(model_path)
            self.scaler = load(scaler_path)
            print("✓ Loaded trained model and scaler")
        
        # Load raw data for checking ANY combo
        self.raw_data = None
        self.data_dir = data_dir
        raw_file = os.path.join(data_dir, 'task3_oncology_drug_event_pairs.csv')
        if os.path.exists(raw_file):
            self.raw_data = pd.read_csv(raw_file)
            print(f"✓ Loaded raw data ({len(self.raw_data):,} reports) for arbitrary queries")
        
        # Load known AEs for filtering
        self.known_aes = {}
        try:
            from task3_drug_label_filter import KNOWN_AES_BACKUP
            self.known_aes = KNOWN_AES_BACKUP
        except:
            pass
    
    def query_drug_event(self, drug, adverse_event):
        """
        Query a specific drug-adverse event combination
        
        Args:
            drug: Drug name (e.g., "Epcoritamab")
            adverse_event: Adverse event term (e.g., "Cytokine release syndrome")
        
        Returns:
            dict with all available information about this pair
        """
        if self.results_df is None:
            return {"error": "No results loaded"}
        
        # Case-insensitive search
        drug_lower = drug.lower()
        ae_lower = adverse_event.lower()
        
        mask = (
            self.results_df['drug'].str.lower() == drug_lower
        ) & (
            self.results_df['adverse_event'].str.lower().str.contains(ae_lower)
        )
        
        matches = self.results_df[mask]
        
        if len(matches) == 0:
            # Try partial match
            mask = (
                self.results_df['drug'].str.lower().str.contains(drug_lower)
            ) & (
                self.results_df['adverse_event'].str.lower().str.contains(ae_lower)
            )
            matches = self.results_df[mask]
        
        if len(matches) == 0:
            return {
                "found": False,
                "message": f"No results found for {drug} + {adverse_event}",
                "suggestion": "Try a partial match or check spelling"
            }
        
        # Return the best match
        best_match = matches.iloc[0]
        
        result = {
            "found": True,
            "drug": best_match['drug'],
            "adverse_event": best_match['adverse_event'],
            "anomaly_score": float(best_match['anomaly_score']),
            "rank": int(best_match.get('rank', 0)),
            "statistics": {
                "count": int(best_match['count']),
                "prr": float(best_match['prr']),
                "ror": float(best_match['ror']),
                "chi2": float(best_match['chi2']),
                "ic025": float(best_match.get('ic025', 0)),
            },
            "clinical_indicators": {
                "death_rate": float(best_match.get('death_rate', 0)),
                "hosp_rate": float(best_match.get('hosp_rate', 0)),
                "serious_rate": float(best_match.get('serious_rate', 0)),
            },
            "why_flagged": best_match.get('why_flagged', 'N/A'),
        }
        
        return result
    
    def check_any_combo(self, drug, adverse_event):
        """
        Check ANY drug-AE combination: Only mark as RARE & UNEXPECTED if:
        1. It appears in task3_all_unexpected_no_cap.csv (passed Isolation Forest)
        2. It passes all three statistical tests (PRR>2, IC025>0, Chi²>4)
        
        Args:
            drug: Drug name (e.g., "Epcoritamab")
            adverse_event: Adverse event term (e.g., "Cytokine release syndrome")
        
        Returns:
            dict with assessment of whether this combo is rare/unexpected
        """
        if self.raw_data is None:
            return {"error": "Raw data not loaded. Cannot check arbitrary combos."}
        
        if self.results_df is None:
            return {"error": "Filtered results not loaded. Cannot check if combo passed IF."}
        
        # Case-insensitive search
        drug_lower = drug.lower()
        ae_lower = adverse_event.lower()
        
        # STEP 1: Check if combo is in no-cap results (passed Isolation Forest)
        in_no_cap = False
        if_combo_data = None
        
        mask_if = (
            self.results_df['drug'].str.lower() == drug_lower
        ) & (
            self.results_df['adverse_event'].str.lower().str.contains(ae_lower, na=False)
        )
        
        if_results = self.results_df[mask_if]
        
        if len(if_results) > 0:
            in_no_cap = True
            if_combo_data = if_results.iloc[0]  # Take first match
        
        # STEP 2: Get raw data stats
        mask_raw = (
            self.raw_data['target_drug'].str.lower() == drug_lower
        ) & (
            self.raw_data['adverse_event'].str.lower().str.contains(ae_lower, na=False)
        )
        
        combo_reports = self.raw_data[mask_raw]
        combo_count = len(combo_reports)
        
        if combo_count == 0:
            return {
                "found": False,
                "drug": drug,
                "adverse_event": adverse_event,
                "message": f"No reports found for {drug} + {adverse_event}",
                "conclusion": "❌ NOT IN DATABASE",
                "in_no_cap_results": False,
                "is_unexpected": False,
            }
        
        # Calculate statistics from raw data (for display)
        total_reports = len(self.raw_data)
        drug_reports = len(self.raw_data[self.raw_data['target_drug'].str.lower() == drug_lower])
        ae_reports = len(self.raw_data[self.raw_data['adverse_event'].str.lower().str.contains(ae_lower, na=False)])
        
        epsilon = 0.5
        a = combo_count + epsilon
        b = ae_reports - combo_count + epsilon
        c = drug_reports + epsilon
        d = total_reports - drug_reports + epsilon
        
        prr = (a / c) / (b / d) if (b > 0 and d > 0) else 0
        ror = (a * d) / (b * c) if (b > 0 and c > 0) else 0
        
        expected = (drug_reports * ae_reports) / total_reports if total_reports > 0 else 0
        chi2 = ((combo_count - expected) ** 2) / expected if expected > 0 else 0
        
        observed = combo_count
        expected_ic = (drug_reports * ae_reports) / total_reports if total_reports > 0 else 1
        ic = np.log2((observed + 0.5) / (expected_ic + 0.5)) if expected_ic > 0 else 0
        ic025 = ic - 1.96 * (1 / np.sqrt(observed + 0.5))
        
        # Use stats from no-cap results if available (more accurate)
        if in_no_cap and if_combo_data is not None:
            prr = float(if_combo_data.get('prr', prr))
            chi2 = float(if_combo_data.get('chi2', chi2))
            ic025 = float(if_combo_data.get('ic025', ic025))
        
        # Calculate death/hospitalization rates
        death_rate = combo_reports['is_death'].sum() / combo_count * 100 if combo_count > 0 else 0
        hosp_rate = combo_reports['is_hospitalization'].sum() / combo_count * 100 if combo_count > 0 else 0
        serious_rate = combo_reports['is_serious'].sum() / combo_count * 100 if combo_count > 0 else 0
        
        # STEP 3: Check three statistical criteria
        passes_prr = prr > 2
        passes_ic025 = ic025 > 0
        passes_chi2 = chi2 > 4
        passes_all_three = passes_prr and passes_ic025 and passes_chi2
        
        # STEP 4: Final assessment
        if not in_no_cap:
            conclusion = "❌ NOT RARE/UNEXPECTED (did not pass Isolation Forest)"
            is_unexpected = False
        elif not passes_all_three:
            conclusion = f"🔶 IN NO-CAP RESULTS but FAILED statistical tests (PRR>2: {passes_prr}, IC025>0: {passes_ic025}, Chi²>4: {passes_chi2})"
            is_unexpected = False
        else:
            conclusion = "✅ RARE & UNEXPECTED (passed IF + all 3 statistical tests)"
            is_unexpected = True
        
        return {
            "found": True,
            "drug": drug,
            "adverse_event": adverse_event,
            "report_count": combo_count,
            "statistics": {
                "prr": round(prr, 2),
                "ror": round(ror, 2),
                "chi2": round(chi2, 2),
                "ic025": round(ic025, 3),
            },
            "clinical": {
                "death_rate": f"{death_rate:.1f}%",
                "hosp_rate": f"{hosp_rate:.1f}%",
                "serious_rate": f"{serious_rate:.1f}%",
            },
            "assessment": {
                "in_no_cap_results": in_no_cap,
                "passes_prr": passes_prr,
                "passes_ic025": passes_ic025,
                "passes_chi2": passes_chi2,
                "passes_all_three": passes_all_three,
            },
            "conclusion": conclusion,
            "is_unexpected": is_unexpected,
        }
    
    def get_top_events_for_drug(self, drug, n=10, exclude_known=False):
        """
        Get top N anomalous adverse events for a specific drug
        
        Args:
            drug: Drug name
            n: Number of results to return
            exclude_known: If True, filter out common/known AEs
        
        Returns:
            DataFrame with top anomalies for this drug
        """
        if self.results_df is None:
            return None
        
        # Filter by drug (case-insensitive)
        drug_lower = drug.lower()
        mask = self.results_df['drug'].str.lower() == drug_lower
        
        if mask.sum() == 0:
            # Try partial match
            mask = self.results_df['drug'].str.lower().str.contains(drug_lower)
        
        drug_results = self.results_df[mask].copy()
        
        if len(drug_results) == 0:
            print(f"No results found for drug: {drug}")
            return None
        
        # Sort by anomaly score (descending)
        drug_results = drug_results.sort_values('anomaly_score', ascending=False)
        
        # Optionally exclude known AEs
        if exclude_known:
            from task3_drug_label_filter import DrugLabelFilter, KNOWN_AES_BACKUP
            filter = DrugLabelFilter()
            filter.known_aes = KNOWN_AES_BACKUP
            drug_results = filter.filter_anomalies(drug_results)
        
        return drug_results.head(n)
    
    def compare_drugs(self, drug_list, adverse_event=None):
        """
        Compare anomaly profiles across multiple drugs
        
        Args:
            drug_list: List of drug names (e.g., ["Epcoritamab", "Glofitamab", "Mosunetuzumab"])
            adverse_event: Optional - compare for a specific AE
        
        Returns:
            DataFrame with comparison
        """
        if self.results_df is None:
            return None
        
        comparison_data = []
        
        for drug in drug_list:
            drug_lower = drug.lower()
            mask = self.results_df['drug'].str.lower() == drug_lower
            drug_results = self.results_df[mask]
            
            if len(drug_results) == 0:
                continue
            
            if adverse_event:
                # Compare specific AE across drugs
                ae_lower = adverse_event.lower()
                ae_mask = drug_results['adverse_event'].str.lower().str.contains(ae_lower)
                ae_results = drug_results[ae_mask]
                
                if len(ae_results) > 0:
                    row = ae_results.iloc[0]
                    comparison_data.append({
                        'drug': drug,
                        'adverse_event': row['adverse_event'],
                        'anomaly_score': row['anomaly_score'],
                        'count': row['count'],
                        'prr': row['prr'],
                        'death_rate': row.get('death_rate', 0),
                    })
            else:
                # Compare overall anomaly profiles
                comparison_data.append({
                    'drug': drug,
                    'total_anomalies': len(drug_results),
                    'avg_anomaly_score': drug_results['anomaly_score'].mean(),
                    'max_anomaly_score': drug_results['anomaly_score'].max(),
                    'avg_prr': drug_results['prr'].mean(),
                    'top_ae': drug_results.iloc[0]['adverse_event'] if len(drug_results) > 0 else 'N/A',
                })
        
        return pd.DataFrame(comparison_data)
    
    def search_by_adverse_event(self, adverse_event, top_n=20):
        """
        Find all drugs with a specific adverse event anomaly
        
        Args:
            adverse_event: Adverse event term to search
            top_n: Number of results
        
        Returns:
            DataFrame with drugs that have this AE as anomaly
        """
        if self.results_df is None:
            return None
        
        ae_lower = adverse_event.lower()
        mask = self.results_df['adverse_event'].str.lower().str.contains(ae_lower)
        
        ae_results = self.results_df[mask].copy()
        ae_results = ae_results.sort_values('anomaly_score', ascending=False)
        
        return ae_results.head(top_n)
    
    def get_available_drugs(self):
        """
        List all drugs in the results
        """
        if self.results_df is None:
            return []
        
        return self.results_df['drug'].unique().tolist()
    
    def get_statistics(self):
        """
        Get overall statistics about the results
        """
        if self.results_df is None:
            return {}
        
        return {
            'total_anomalies': len(self.results_df),
            'unique_drugs': self.results_df['drug'].nunique(),
            'unique_events': self.results_df['adverse_event'].nunique(),
            'avg_anomaly_score': self.results_df['anomaly_score'].mean(),
            'score_range': (
                self.results_df['anomaly_score'].min(),
                self.results_df['anomaly_score'].max()
            ),
        }


def interactive_demo():
    """
    Interactive demonstration of the query system
    """
    print("=" * 70)
    print("Task 3: Interactive Drug-Event Anomaly Query System")
    print("=" * 70)
    
    # Initialize
    query = InteractiveAnomalyQuery()
    
    if query.results_df is None:
        print("No results file found. Please run task3_train.py first.")
        return
    
    print("\n" + "-" * 70)
    print("Available Drugs:")
    print("-" * 70)
    drugs = query.get_available_drugs()
    for i, drug in enumerate(drugs, 1):
        print(f"  {i}. {drug}")
    
    print("\n" + "-" * 70)
    print("Example Queries:")
    print("-" * 70)
    
    # Example 1: Query specific drug-event pair
    print("\n[1] Query: Epcoritamab + Cytokine release syndrome")
    result = query.query_drug_event("Epcoritamab", "Cytokine release syndrome")
    if result.get('found'):
        print(f"    Anomaly Score: {result['anomaly_score']:.4f}")
        print(f"    PRR: {result['statistics']['prr']:.2f}")
        print(f"    Report Count: {result['statistics']['count']}")
        print(f"    Why Flagged: {result['why_flagged']}")
    else:
        print(f"    {result['message']}")
    
    # Example 2: Top events for a drug
    print("\n[2] Top 5 Anomalies for Epcoritamab:")
    top_events = query.get_top_events_for_drug("Epcoritamab", n=5)
    if top_events is not None:
        for _, row in top_events.iterrows():
            print(f"    - {row['adverse_event']}: Score={row['anomaly_score']:.4f}, PRR={row['prr']:.2f}")
    
    # Example 3: Compare bispecific antibodies
    print("\n[3] Comparing Bispecific Antibodies (Epcoritamab vs Glofitamab vs Mosunetuzumab):")
    comparison = query.compare_drugs(
        ["Epcoritamab", "Glofitamab", "Mosunetuzumab"],
        adverse_event="Cytokine release syndrome"
    )
    if comparison is not None and len(comparison) > 0:
        print(comparison.to_string(index=False))
    else:
        print("    No comparison data available (run data collection with new drugs first)")
    
    # Example 4: Search by adverse event
    print("\n[4] Drugs with 'Neutropenia' as Anomaly:")
    neutropenia_results = query.search_by_adverse_event("Neutropenia", top_n=5)
    if neutropenia_results is not None:
        for _, row in neutropenia_results.iterrows():
            print(f"    - {row['drug']}: Score={row['anomaly_score']:.4f}, Count={row['count']}")
    
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    stats = query.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def print_combo_result(drug, adverse_event):
    """
    Print formatted result for a drug-AE combination
    """
    query = InteractiveAnomalyQuery()
    result = query.check_any_combo(drug, adverse_event)
    
    print("=" * 70)
    print(f"Drug-Event Pair: {drug} + {adverse_event}")
    print("=" * 70)
    
    if not result.get("found", False):
        print(f"\n{result.get('conclusion', result.get('message', 'Not found'))}")
        return
    
    print(f"\nReport Count: {result['report_count']}")
    print(f"\nStatistical Metrics:")
    prr_status = "PASS" if result['assessment']['passes_prr'] else "FAIL"
    ic025_status = "PASS" if result['assessment']['passes_ic025'] else "FAIL"
    chi2_status = "PASS" if result['assessment']['passes_chi2'] else "FAIL"
    print(f"   PRR:     {result['statistics']['prr']:.2f} ({prr_status})")
    print(f"   IC025:   {result['statistics']['ic025']:.3f} ({ic025_status})")
    print(f"   Chi-square: {result['statistics']['chi2']:.2f} ({chi2_status})")
    
    print(f"\nClinical Indicators:")
    print(f"   Death Rate:      {result['clinical']['death_rate']}")
    print(f"   Hospitalization: {result['clinical']['hosp_rate']}")
    print(f"   Serious Events:  {result['clinical']['serious_rate']}")
    
    print(f"\nAssessment:")
    no_cap_status = "Yes" if result['assessment']['in_no_cap_results'] else "No"
    all_three_status = "Yes" if result['assessment']['passes_all_three'] else "No"
    print(f"   In No-Cap Results: {no_cap_status}")
    print(f"   Passes All 3 Tests: {all_three_status}")
    
    print(f"\n{'=' * 70}")
    # Clean conclusion text (remove emoji)
    conclusion = result['conclusion']
    conclusion = conclusion.replace('✅', '').replace('❌', '').replace('⚠️', '').replace('🔶', '').strip()
    print(f"CONCLUSION: {conclusion}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    import sys
    
    # If command-line arguments provided, use them
    if len(sys.argv) == 3:
        drug = sys.argv[1]
        adverse_event = sys.argv[2]
        print_combo_result(drug, adverse_event)
    else:
        # Otherwise run interactive demo
        interactive_demo()

