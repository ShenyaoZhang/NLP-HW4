"""
Task 3 Configuration File
Centralized configuration for anomaly detection parameters.
"""

CONFIG = {
    # Model parameters
    "random_state": 42,
    "contamination": 0.15,
    "n_estimators": 100,
    
    # Output parameters (decoupled from contamination)
    "top_k_global": 200,
    "per_drug_cap": 5,
    "secondary_metric": "ic025",  # Options: "ic025", "log_prr", "chi2"
    
    # RARE event filter - only keep events with low count
    "max_count_for_rare": 10,  # Maximum count to be considered "rare"
    
    # PRR override - if PRR > threshold, keep even if "known" AE
    # High PRR means the AE is disproportionately reported for this drug
    "prr_override_threshold": 50,  # Keep known AEs if PRR > 50
    
    # Statistical parameters
    "eps_smoothing": 0.5,  # Haldane–Anscombe smoothing
    "winsor_limits": (0.01, 0.99),  # (lower, upper) percentiles
    
    # Outcome PT stoplist (exclude these from analysis)
    "stoplist_outcomes": [
        "Death",
        "Fatal outcome",
        "Hospitalisation",
        "Hospitalization",
        "Life-threatening",
        "Disability",
        "Prolonged hospitalization"
    ],
    
    # Why flagged rules (column, threshold, label)
    "why_rules": [
        ("log_prr", 1.0, "High PRR"),
        ("ic025", 0.0, "IC025 > 0"),
        ("chi2", 4.0, "High Chi-square"),
    ],
    
    # Time window (reserved for future use)
    "time_window_days": 180,
}

