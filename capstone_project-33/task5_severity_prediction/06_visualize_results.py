#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 5 - Step 6/9: Results Visualisation (enhanced)
- Select best model by PR-AUC
- Advanced curves use the selected best model file (no hardcoded names)
- Auto-detect label column in y_test
- Fallback to decision_function and normalise when predict_proba is unavailable
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

print("=" * 80)
print("Task 5 - Step 6/9: Results Visualisation (enhanced)")
print("=" * 80)
print()

# -----------------------------
# Global plotting style
# -----------------------------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# -----------------------------
# Pre-flight checks
# -----------------------------
REQUIRED_FILES = {
    "model_comparison.csv": "model performance table",
    "feature_importance.csv": "feature importance",
    "y_test.csv": "test labels"
}
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing_files:
    print("❌ Error: missing required files:")
    for f in missing_files:
        print(f"  -", f)
    print("\nPlease ensure step4/step5 have been run")
    sys.exit(1)

print("✅ Found all required files\n")

# -----------------------------
# Helper functions
# -----------------------------
def load_y_test_auto(path="y_test.csv"):
    """
    Adaptively read y_test:
    - Prefer a column named 'serious'
    - Otherwise use 'seriousnessdeath'
    - Else use the first column
    Return: pandas Series of 0/1
    """
    ydf = pd.read_csv(path)
    candidate_cols = []
    for col in ydf.columns:
        lc = col.lower()
        if lc in ("serious", "serious_flag", "is_serious"):
            candidate_cols.append(col)
        elif lc in ("seriousnessdeath", "death", "is_death"):
            candidate_cols.append(col)

    if candidate_cols:
        col = candidate_cols[0]
    else:
        col = ydf.columns[0]

    y = pd.to_numeric(ydf[col], errors="coerce")
    # Treat non-zero values as 1, fill missing with 0
    y = (y.fillna(0).astype(float) > 0).astype(int)
    return y

def get_best_model_name_by_prauc(model_df: pd.DataFrame) -> str:
    """
    Select best model name from comparison table based on PR-AUC.
    """
    if "pr_auc" not in model_df.columns:
        raise ValueError("model_comparison.csv is missing 'pr_auc'; cannot choose best model.")
    return model_df["pr_auc"].idxmax()

def build_model_filename(model_name: str) -> str:
    """
    Build trained model filename consistent with step4 naming convention.
    model_name here is typically the index string, e.g. 'Random Forest'
    """
    return f"trained_model_{model_name.lower().replace(' ', '_')}.pkl"

def get_model_scores_series(model_df: pd.DataFrame, model_name: str) -> pd.Series:
    return model_df.loc[model_name]

def get_probabilities(model, X):
    """
    Retrieve positive-class probabilities:
    - If predict_proba exists: use proba[:,1]
    - Else if decision_function exists: min-max normalise to [0,1]
    - Otherwise return None
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X).astype(float)
        # Min-max normalisation
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return (scores - smin) / (smax - smin)
        # Constant vector then degenerates to 0.5
        return np.full_like(scores, 0.5, dtype=float)
    else:
        return None

# -----------------------------
# Load core data
# -----------------------------
model_results: pd.DataFrame = pd.read_csv("model_comparison.csv", index_col=0)
feature_imp: pd.DataFrame = pd.read_csv("feature_importance.csv")
y_test = load_y_test_auto("y_test.csv")

# -----------------------------
# Figure 1: Model comparison (four metrics)
# -----------------------------
print("=" * 80)
print("📊 1. Model comparison")
print("=" * 80)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
metrics = ['accuracy', 'precision', 'recall', 'f1']
titles  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx // 2, idx % 2]
    if metric in model_results.columns:
        data = model_results[metric].sort_values()
        bars = ax.barh(range(len(data)), data.values)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index)
        ax.set_xlabel(title)
        ax.set_xlim([0, 1])

        for i, (bar, value) in enumerate(zip(bars, data.values)):
            ax.text(min(value + 0.02, 0.98), i, f'{value:.3f}', va='center')

plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: model_performance_comparison.png\n")

# -----------------------------
# Figure 2: Feature importance (Top-N)
# -----------------------------
print("=" * 80)
print("📊 2. Feature importance")
print("=" * 80)
print()

plt.figure(figsize=(10, 8))
top_n = min(15, len(feature_imp))
top_features = feature_imp.head(top_n)

plt.barh(range(top_n), top_features['importance'])
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('Importance Score')
plt.title(f'Top {top_n} Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('top_features.png', dpi=300, bbox_inches='tight')
print("✅ Saved: top_features.png\n")

# -----------------------------
# Figure 3: Data distribution (optional)
# -----------------------------
print("=" * 80)
print("📊 3. Data distribution analysis")
print("=" * 80)
print()

if os.path.exists("preprocessed_data.csv"):
    df = pd.read_csv("preprocessed_data.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Distribution Analysis', fontsize=16)

    # Age distribution
    if 'age_years' in df.columns or 'patientonsetage' in df.columns:
        age_col = 'age_years' if 'age_years' in df.columns else 'patientonsetage'
        age_data = pd.to_numeric(df[age_col], errors='coerce').dropna()
        if len(age_data) > 0:
            axes[0, 0].hist(age_data, bins=30, edgecolor='black')
            axes[0, 0].set_xlabel('Age (years)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Age Distribution')

    # Gender distribution
    if 'patientsex' in df.columns:
        sex_counts = df['patientsex'].value_counts()
        sex_labels = {1: 'Male', 2: 'Female', 0: 'Unknown'}
        labels = [sex_labels.get(x, f'Code {x}') for x in sex_counts.index]
        axes[0, 1].bar(range(len(sex_counts)), sex_counts.values)
        axes[0, 1].set_xticks(range(len(sex_counts)))
        axes[0, 1].set_xticklabels(labels)
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Gender Distribution')

    # Drug count distribution
    if 'num_drugs' in df.columns:
        drug_counts = df['num_drugs'].value_counts().sort_index()
        axes[1, 0].bar(drug_counts.index, drug_counts.values)
        axes[1, 0].set_xlabel('Number of Drugs')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Polypharmacy Distribution')

    # Severe outcome (example: death) distribution
    if 'seriousnessdeath' in df.columns:
        death_vals = pd.to_numeric(df['seriousnessdeath'], errors='coerce').fillna(0).astype(int)
        death_counts = death_vals.value_counts().sort_index()
        axes[1, 1].bar([0,1], [death_counts.get(0,0), death_counts.get(1,0)])
        axes[1, 1].set_xticks([0,1])
        axes[1, 1].set_xticklabels(['No Death', 'Death'])
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Outcome Distribution')

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: data_distribution.png\n")

# -----------------------------
# Generate text summary report (uniform PR-AUC)
# -----------------------------
print("=" * 80)
print("📝 Generating summary report")
print("=" * 80)
print()

best_model_by_prauc = get_best_model_name_by_prauc(model_results)
best_scores = get_model_scores_series(model_results, best_model_by_prauc)

with open("RESULTS_SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("Task 5: Adverse Event Severity Prediction - Results Summary\n")
    f.write("=" * 80 + "\n\n")

    # Data overview (optional: can be supplemented if main_data.csv / preprocessed_data.csv is present)
    # This preserves the original example logic
    if os.path.exists("epcoritamab_data.csv"):
        df_raw = pd.read_csv("epcoritamab_data.csv")
        f.write("1. Data Overview\n")
        f.write(f"   Total Records : {len(df_raw)}\n")
        f.write(f"   Total Features: {len(df_raw.columns)}\n\n")

    # Model performance table
    f.write("2. Model Performance (all metrics)\n\n")
    f.write(model_results.to_string())
    f.write("\n\n")

    # Best model (by PR-AUC)
    f.write("3. Best Model (by PR-AUC)\n")
    f.write(f"   Model : {best_model_by_prauc}\n")
    f.write(f"   PR-AUC: {best_scores.get('pr_auc', float('nan')):.4f}\n")
    if 'accuracy' in best_scores:
        f.write(f"   Accuracy: {best_scores['accuracy']:.4f}\n")
    if 'f1' in best_scores:
        f.write(f"   F1-Score: {best_scores['f1']:.4f}\n")
    if 'roc_auc' in best_scores:
        f.write(f"   ROC-AUC : {best_scores['roc_auc']:.4f}\n")
    if 'best_threshold' in best_scores:
        f.write(f"   Best Threshold (F1): {best_scores['best_threshold']:.3f}\n")
    f.write("\n")

    # Top features
    f.write("4. Top 10 Most Important Features\n\n")
    f.write(feature_imp.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("=" * 80 + "\n")
    f.write("End of Summary\n")
    f.write("=" * 80 + "\n")

print("✅ Saved: RESULTS_SUMMARY.txt\n")

# -----------------------------
# Checklists and next steps
# -----------------------------
print("=" * 80)
print("✅ Step 6 complete - visualisation finished")
print("=" * 80)
print()
print("📁 Generated visual assets:")
print("  1. model_performance_comparison.png - model performance comparison")
print("  2. top_features.png - top feature bar chart")
if os.path.exists("data_distribution.png"):
    print("  3. data_distribution.png - data distribution summary")
print("  4. RESULTS_SUMMARY.txt - textual summary\n")

print("💡 View outputs:")
print("  View image: open model_performance_comparison.png")
print("  View report: open RESULTS_SUMMARY.txt\n")

print("=" * 80)
print("🎉 Task 5 pipeline complete!")
print("=" * 80)
print()

print("📋 Complete output checklist:\n")
print("Data files:")
print("  ✓ preprocessed_data.csv (if present) - preprocessed data")
print("  ✓ X_train.csv, y_train.csv - training set")
print("  ✓ X_test.csv, y_test.csv - test set\n")

print("Model files:")
model_files = [f for f in os.listdir('.') if f.startswith('trained_model_') and f.endswith('.pkl')]
for f in model_files:
    print("  ✓", f)
print()

print("Result files:")
print("  ✓ model_comparison.csv - model scores")
print("  ✓ feature_importance.csv - Feature importance table\n")

print("Visual files:")
print("  ✓ model_performance_comparison.png")
print("  ✓ top_features.png")
if os.path.exists("data_distribution.png"):
    print("  ✓ data_distribution.png")
print()

print("Report files:")
print("  ✓ RESULTS_SUMMARY.txt\n")

print("🎯 Suggested next steps:")
print("  1. Review RESULTS_SUMMARY.txt for overall summary")
print("  2. Inspect charts to understand model performance and distributions")
print("  3. Use feature importance to refine modelling")
print("  4. Prepare project report and presentation\n")

# -----------------------------
# Advanced evaluation curves (PR/ROC/calibration): using the "best model (by PR-AUC)"
# -----------------------------
print("=" * 80)
print("📈 Advanced evaluation curves (PR/ROC/calibration)")
print("=" * 80)
print()

try:
    import pickle
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    from sklearn.calibration import calibration_curve

    best_model_file = build_model_filename(best_model_by_prauc)
    if not os.path.exists(best_model_file):
        print(f"⚠️  Best model file not found: {best_model_file}，skipping advanced curves generation")
    elif not os.path.exists("X_test.csv") or not os.path.exists("y_test.csv"):
        print("⚠️  X_test/y_test missing; skipping advanced curves")
    else:
        # Load best model and test set
        with open(best_model_file, "rb") as f:
            best_model = pickle.load(f)
        X_test = pd.read_csv("X_test.csv")
        y_test = load_y_test_auto("y_test.csv")

        # Probability or confidence score
        y_pred_proba = get_probabilities(best_model, X_test)
        if y_pred_proba is None:
            print("⚠️  Model lacks probability/decision outputs; skipping curves")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1) PR curve + F1 optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall, precision)
            if len(thresholds) > 0:
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
                best_idx = f1_scores.argmax()
                best_threshold = float(thresholds[best_idx])
                axes[0].scatter([recall[best_idx]], [precision[best_idx]], color='red', s=100, zorder=5,
                                label=f'Best threshold={best_threshold:.3f}')
            else:
                best_threshold = 0.5

            axes[0].plot(recall, precision, linewidth=2, label=f'PR curve (AUC={pr_auc:.3f})')
            axes[0].set_xlabel('Recall', fontsize=12)
            axes[0].set_ylabel('Precision', fontsize=12)
            axes[0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # 2) ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC={roc_auc:.3f})')
            axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random guess')
            axes[1].set_xlabel('False Positive Rate (FPR)', fontsize=12)
            axes[1].set_ylabel('True Positive Rate (TPR)', fontsize=12)
            axes[1].set_title('ROC curve', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # 3) Probability calibration curve
            frac_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10, strategy='uniform')
            axes[2].plot(mean_pred, frac_pos, marker='o', linewidth=2, label='Model calibration')
            axes[2].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
            axes[2].set_xlabel('Predicted probability', fontsize=12)
            axes[2].set_ylabel('Observed ratio', fontsize=12)
            axes[2].set_title('Probability calibration curve', fontsize=14, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('advanced_evaluation_curves.png', dpi=300, bbox_inches='tight')
            print("✅ Saved: advanced_evaluation_curves.png")
            print(f"   Using best model: {best_model_by_prauc}")
            print(f"   PR-AUC={pr_auc:.3f}, ROC-AUC={roc_auc:.3f}, F1Best threshold={best_threshold:.3f}\n")

except ImportError as e:
    print(f"⚠️  Library import failed: {e}")
    print("   For advanced curves ensure: pip install scikit-learn matplotlib seaborn")
except Exception as e:
    print(f"⚠️  Error generating advanced curves: {str(e)[:100]}")

print()
