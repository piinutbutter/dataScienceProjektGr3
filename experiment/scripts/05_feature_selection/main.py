"""
Feature selection: compute and inspect correlations of features with targets.

This script:
- Loads preprocessed training data (Parquet files)
- Computes Pearson correlation matrix (feature-feature)
- Computes feature-target correlations and creates ranking
- Visualizes: one correlation matrix and one ranking plot

Usage:
- Run from project root: python experiment/scripts/05_feature_selection/main.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Load configuration
params_path = PROJECT_ROOT / "experiment" / "conf" / "params.yaml"
if not params_path.exists():
    raise FileNotFoundError(f"params.yaml not found at: {params_path}")

params = yaml.safe_load(open(params_path))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]

# Use a representative horizon (15m) for analysis
representative_horizon = 15

# Create output directory for plots
output_dir = PROJECT_ROOT / "experiment" / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

symbol = "GRXEUR"

# Load feature list
features_file = processed_path / "features.txt"
if not features_file.exists():
    raise FileNotFoundError(f"Feature list not found: {features_file}")

with open(features_file, "r") as f:
    feature_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(feature_list)} features from {features_file}")

# Load training data
train_file = processed_path / f"{symbol}_train.parquet"
if not train_file.exists():
    raise FileNotFoundError(f"Training file not found: {train_file}")

print(f"\nLoading training data from {train_file}")
df = pd.read_parquet(train_file)

# Target column (using direction target as primary)
target_col = f"target_direction_{representative_horizon}m"

if target_col not in df.columns:
    raise ValueError(f"Target column {target_col} not found in data")

# Select features and target, drop NaN
analysis_cols = feature_list + [target_col]
df_clean = df[analysis_cols].dropna().reset_index(drop=True)

print(f"Clean data: {len(df_clean)} samples after dropping NaN")

# Sample data if too large (for faster computation)
max_samples = 100000
if len(df_clean) > max_samples:
    print(f"Sampling {max_samples} rows for correlation computation...")
    df_clean = df_clean.sample(n=max_samples, random_state=42).reset_index(drop=True)

# Compute correlation matrix
print("Computing correlation matrix...")
corr_matrix = df_clean.corr()

# 1. Feature-Feature correlation matrix
feature_corr = corr_matrix.loc[feature_list, feature_list]

# 2. Feature-Target correlations (sorted)
target_corr = corr_matrix.loc[feature_list, target_col].sort_values(ascending=False)

print(f"\nTop 10 features correlated with {target_col}:")
for feat, corr_val in target_corr.head(10).items():
    print(f"  {feat}: {corr_val:.6f}")

print(f"\nBottom 10 features correlated with {target_col}:")
for feat, corr_val in target_corr.tail(10).items():
    print(f"  {feat}: {corr_val:.6f}")

# Visualizations
print("\nCreating visualizations...")

# 1. Feature-Feature correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    feature_corr,
    annot=False,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={'label': 'Correlation'},
    xticklabels=False,
    yticklabels=False
)
plt.title('Feature-Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(output_dir / '06_correlations.png', dpi=150)
plt.close()
print(f"  Saved: {output_dir / '06_correlations.png'}")

# 2. Feature-Target correlation ranking
plt.figure(figsize=(8, 12))
target_corr_sorted = target_corr.sort_values(ascending=True)
colors = ['red' if x < 0 else 'green' for x in target_corr_sorted.values]
plt.barh(range(len(target_corr_sorted)), target_corr_sorted.values, color=colors, alpha=0.6)
plt.yticks(range(len(target_corr_sorted)), target_corr_sorted.index, fontsize=8)
plt.xlabel('Correlation with target_direction')
plt.title(f'Feature Correlations with Target (h{representative_horizon}m)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '06_correlations_target.png', dpi=150)
plt.close()
print(f"  Saved: {output_dir / '06_correlations_target.png'}")

# Save ranking to CSV
corr_output_dir = processed_path / "correlations"
corr_output_dir.mkdir(exist_ok=True)

target_corr_df = pd.DataFrame({
    'feature': target_corr.index,
    'correlation_with_target': target_corr.values
})
target_corr_df.to_csv(corr_output_dir / 'correlations_target_ranking.csv', index=False)
print(f"  Saved ranking to: {corr_output_dir / 'correlations_target_ranking.csv'}")

print("\nFeature selection analysis completed!")

