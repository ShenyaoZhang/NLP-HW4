#!/usr/bin/env python3
"""
Task 3: Data Processing and Model Definition for Isolation Forest
This module handles data loading, feature engineering, and model setup.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import os


class Task3DataProcessor:
    """
    Data processor for Task 3: Detect Rare and Unexpected Drug-Event Relationships
    Handles data loading, feature engineering, and statistical calculations.
    """
    
    def __init__(self, data_file_path):
        """
        Initialize the data processor.
        
        Args:
            data_file_path: Path to the CSV file containing drug-event pairs
        """
        self.data_file_path = data_file_path
        self.df = None
        self.drug_event_features = None
        self.drug_totals = None
        self.event_totals = None
        self.total_reports = 0
        self.pair_names = []
        self.feature_names = [
            'count', 'prr', 'ror', 'chi2', 'serious_rate', 'death_rate',
            'hosp_rate', 'life_threat_rate', 'disable_rate', 'report_freq',
            'log_count', 'avg_age', 'avg_drug_count'
        ]
    
    def load_data(self):
        """Load the drug-event pairs dataset."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_file_path)
        
        # Exclude outcome PTs (Death, Hospitalization, etc.) to avoid pseudo-anomalies
        try:
            from config_task3 import CONFIG
            stoplist = CONFIG.get("stoplist_outcomes", [])
        except ImportError:
            stoplist = ["Death", "Fatal outcome", "Hospitalisation", "Hospitalization", 
                       "Life-threatening", "Disability", "Prolonged hospitalization"]
        
        initial_count = len(self.df)
        self.df = self.df[~self.df['adverse_event'].isin(stoplist)].copy()
        self.n_excluded_outcomes = initial_count - len(self.df)
        self.total_reports = len(self.df)
        
        print(f"✓ Loaded {initial_count} records")
        if self.n_excluded_outcomes > 0:
            print(f"✓ Excluded {self.n_excluded_outcomes} outcome PT records")
        print(f"✓ Final dataset: {self.total_reports} records\n")
        return self
    
    def compute_statistics(self):
        """
        Compute statistics for each drug-event pair.
        Calculates counts, severity rates, and collects demographic information.
        """
        print("Computing drug-event pair statistics...")
        
        self.drug_event_features = defaultdict(lambda: {
            'count': 0,
            'serious_count': 0,
            'death_count': 0,
            'hosp_count': 0,
            'life_threat_count': 0,
            'disable_count': 0,
            'ages': [],
            'drug_counts': []
        })
        
        self.drug_totals = Counter()
        self.event_totals = Counter()
        
        for _, row in self.df.iterrows():
            drug = row['target_drug']
            event = row['adverse_event']
            pair = f"{drug}||{event}"
            
            self.drug_totals[drug] += 1
            self.event_totals[event] += 1
            self.drug_event_features[pair]['count'] += 1
            
            # Count severity indicators
            try:
                if str(row.get('is_serious', '0')).strip() in ['1', '1.0']:
                    self.drug_event_features[pair]['serious_count'] += 1
                if str(row.get('is_death', '0')).strip() in ['1', '1.0']:
                    self.drug_event_features[pair]['death_count'] += 1
                if str(row.get('is_hospitalization', '0')).strip() in ['1', '1.0']:
                    self.drug_event_features[pair]['hosp_count'] += 1
                if str(row.get('is_lifethreatening', '0')).strip() in ['1', '1.0']:
                    self.drug_event_features[pair]['life_threat_count'] += 1
                if str(row.get('is_disabling', '0')).strip() in ['1', '1.0']:
                    self.drug_event_features[pair]['disable_count'] += 1
            except:
                pass
            
            # Collect age and drug count information
            try:
                age = float(row.get('patient_age', 0))
                if 0 < age < 120:
                    self.drug_event_features[pair]['ages'].append(age)
            except:
                pass
            
            try:
                drug_count = int(row.get('drug_count', 0))
                if drug_count > 0:
                    self.drug_event_features[pair]['drug_counts'].append(drug_count)
            except:
                pass
        
        print(f"✓ Identified {len(self.drug_event_features)} unique drug-event pairs\n")
        return self
    
    def build_feature_matrix(self):
        """
        Build feature matrix for machine learning.
        
        Features include:
        - Count: Number of reports
        - PRR: Proportional Reporting Ratio
        - ROR: Reporting Odds Ratio
        - Chi-square: Statistical independence test
        - Severity rates: serious, death, hospitalization, life-threatening, disabling
        - Report frequency: Proportion of total reports
        - Log count: Logarithm of report count
        - Average age: Mean patient age
        - Average drug count: Mean number of drugs per report
        
        Returns:
            numpy.ndarray: Feature matrix (n_samples, n_features)
        """
        print("Building feature matrix...")
        
        feature_data = []
        self.pair_names = []
        # Store additional metrics for post-processing (not used in model training)
        self.additional_metrics = {}
        
        for pair, stats in self.drug_event_features.items():
            drug, event = pair.split('||')
            
            count = stats['count']
            drug_total = self.drug_totals[drug]
            event_total = self.event_totals[event]
            
            # Calculate PRR (Proportional Reporting Ratio)
            # 2x2 contingency table: a=N11, b=N10, c=N01, d=N00
            # Add small epsilon to prevent division by zero (Haldane–Anscombe smoothing)
            try:
                from config_task3 import CONFIG
                eps = CONFIG.get("eps_smoothing", 0.5)
            except ImportError:
                eps = 0.5
            
            a = count + eps
            b = drug_total - count + eps
            c = event_total - count + eps
            d = self.total_reports - drug_total - event_total + count + eps
            N = a + b + c + d
            
            # PRR = (a/(a+b)) / (c/(c+d))
            prr = ((a / (a + b)) / (c / (c + d))) if (c / (c + d)) > 0 else 0
            
            # ROR = (a*d) / (b*c)
            ror = (a * d) / (b * c) if (b * c) > 0 else 0
            
            # Chi-square: χ² = N(ad−bc)² / ((a+b)(c+d)(a+c)(b+d))
            chi2 = 0
            denominator = (a + b) * (c + d) * (a + c) * (b + d)
            if denominator > 0:
                chi2 = float(N * ((a * d - b * c) ** 2) / denominator)
            
            # Calculate log PRR (for downstream use)
            log_prr = np.log(prr) if prr > 0 else 0.0
            
            # Calculate IC/IC025 (approximate Bayesian shrinkage metrics)
            # Expected value E = ((a+b)*(a+c))/N
            E = (a + b) * (a + c) / max(N, 1.0)
            # IC = log2(A / E)
            ic = np.log2(a / max(E, 1e-9)) if a > 0 and E > 0 else 0.0
            # IC025 approximation: IC - 1.96 * sqrt(1/(A + k))
            k = 1.0
            ic025 = ic - 1.96 * np.sqrt(1.0 / (a + k)) if (a + k) > 0 else 0.0
            
            # Calculate severity rates
            serious_rate = stats['serious_count'] / count if count > 0 else 0
            death_rate = stats['death_count'] / count if count > 0 else 0
            hosp_rate = stats['hosp_count'] / count if count > 0 else 0
            life_threat_rate = stats['life_threat_count'] / count if count > 0 else 0
            disable_rate = stats['disable_count'] / count if count > 0 else 0
            
            # Calculate frequency
            report_freq = count / self.total_reports
            
            # Calculate averages
            avg_age = np.mean(stats['ages']) if stats['ages'] else 0
            avg_drug_count = np.mean(stats['drug_counts']) if stats['drug_counts'] else 0
            
            # Build feature vector
            features = [
                count,                  # Report count
                prr,                    # PRR
                ror,                    # ROR
                chi2,                   # Chi-square
                serious_rate,           # Serious rate
                death_rate,             # Death rate
                hosp_rate,              # Hospitalization rate
                life_threat_rate,       # Life-threatening rate
                disable_rate,           # Disabling rate
                report_freq,            # Report frequency
                np.log(count + 1),      # Log frequency
                avg_age,                # Average age
                avg_drug_count          # Average drug count
            ]
            
            feature_data.append(features)
            self.pair_names.append(pair)
            
            # Store additional metrics for post-processing
            self.additional_metrics[pair] = {
                'log_prr': log_prr,
                'ic': ic,
                'ic025': ic025
            }
        
        X = np.array(feature_data)
        
        # Numerical stabilization: handle inf and nan
        X = np.where(np.isinf(X), np.nan, X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Winsorize extreme values (1st and 99th percentiles)
        # Only apply to non-count features (skip count, log_count which are already bounded)
        for col_idx in range(X.shape[1]):
            if col_idx not in [0, 10]:  # Skip count (col 0) and log_count (col 10)
                col = X[:, col_idx]
                if len(np.unique(col)) > 1:  # Only if column has variation
                    p1, p99 = np.nanpercentile(col, [1, 99])
                    X[:, col_idx] = np.clip(col, p1, p99)
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Number of features: {X.shape[1]}")
        print(f"✓ Numerical stabilization applied (inf/nan handling + winsorize)\n")
        
        return X
    
    def process(self):
        """
        Run the complete data processing pipeline.
        
        Returns:
            tuple: (X, pair_names, additional_metrics) where:
                X is the feature matrix
                pair_names is the list of drug-event pairs
                additional_metrics is a dict with log_prr, ic, ic025 for each pair
        """
        self.load_data()
        self.compute_statistics()
        X = self.build_feature_matrix()
        return X, self.pair_names, self.additional_metrics


class Task3Model:
    """
    Isolation Forest model for detecting rare and unexpected drug-event relationships.
    """
    
    def __init__(self, contamination=0.15, random_state=42, n_estimators=100):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination (float): Expected proportion of anomalies (default: 0.15 = 15%)
            random_state (int): Random seed for reproducibility
            n_estimators (int): Number of isolation trees
        """
        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            verbose=0
        )
        
        self.X_scaled = None
        self.predictions = None
        self.anomaly_scores = None
    
    def fit(self, X):
        """
        Fit the model on the feature matrix.
        
        Args:
            X (numpy.ndarray): Feature matrix
            
        Returns:
            self
        """
        print("Standardizing features...")
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"✓ Standardization complete\n")
        
        print("Training Isolation Forest model...")
        self.predictions = self.model.fit_predict(self.X_scaled)
        self.anomaly_scores = self.model.score_samples(self.X_scaled)
        
        print(f"✓ Model training complete")
        print(f"✓ Using {self.model.n_estimators} decision trees")
        print(f"✓ Anomaly threshold: {self.contamination * 100}%\n")
        
        return self
    
    def save_model(self, output_dir):
        """
        Save the trained model and scaler to disk for reproducibility.
        
        Args:
            output_dir (str): Directory to save model artifacts
        """
        import os
        from joblib import dump
        
        os.makedirs(output_dir, exist_ok=True)
        
        scaler_path = os.path.join(output_dir, 'task3_scaler.joblib')
        model_path = os.path.join(output_dir, 'task3_if_model.joblib')
        
        dump(self.scaler, scaler_path)
        dump(self.model, model_path)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Scaler saved: {scaler_path}\n")
    
    def get_results(self):
        """
        Get prediction results.
        
        Returns:
            dict: Dictionary containing predictions and anomaly scores
        """
        if self.predictions is None or self.anomaly_scores is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        n_anomalies = np.sum(self.predictions == -1)
        n_normal = np.sum(self.predictions == 1)
        
        return {
            'predictions': self.predictions,
            'anomaly_scores': self.anomaly_scores,
            'n_anomalies': n_anomalies,
            'n_normal': n_normal,
            'anomaly_rate': n_anomalies / len(self.predictions)
        }


# Utility function to get data file path
def get_data_file_path():
    """
    Get the path to the data file relative to this script.
    
    Returns:
        str: Absolute path to the data file
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    data_file = os.path.join(parent_dir, 'data', 'task3_oncology_drug_event_pairs.csv')
    return data_file


if __name__ == "__main__":
    # Example usage
    print("Task 3: Data Processing and Model Definition")
    print("This module provides classes for data processing and model setup.")
    print("Use task3_train.py to train the model and view results.")

