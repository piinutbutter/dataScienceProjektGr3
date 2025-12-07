"""
Evaluate a trained scikit-learn DecisionTree on validation and test splits.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Loads the trained DecisionTree model
- Computes overall metrics (accuracy, confusion matrix) for validation and test
- Exports per-node CSV and prints per-node subset stats using decision rules

Usage:
- Run from project root: python experiment/scripts/07_model_testing/evaluate_dt.py
- Specify horizon as command line argument: python evaluate_dt.py 15
"""

import os
import sys
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make tree_utilities importable
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))
from tree_utilities import get_tree_stats, get_decision_path, apply_decision_rules  # noqa: E402

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Load configuration
params_path = PROJECT_ROOT / "experiment" / "conf" / "params.yaml"
if not params_path.exists():
    raise FileNotFoundError(f"params.yaml not found at: {params_path}")

params = yaml.safe_load(open(params_path))
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
model_path = PROJECT_ROOT / params["MODELING"]["MODEL_PATH"]
prediction_periods = params["DATA_PREP"]["PREDICTION_PERIODS"]
data_out_dir = PROJECT_ROOT / "experiment" / "data"

# Get horizon from command line or use default
if len(sys.argv) > 1:
    horizon = int(sys.argv[1])
    if horizon not in prediction_periods:
        raise ValueError(f"Horizon {horizon} not in {prediction_periods}")
else:
    horizon = 15  # Default
    print(f"No horizon specified, using default: {horizon}m")

symbol = "GRXEUR"
os.makedirs(model_path, exist_ok=True)
os.makedirs(data_out_dir, exist_ok=True)

# -----------------------------
# Load ML-ready data
# -----------------------------
data_file = processed_path / f"{symbol}_h{horizon}m_ml_ready.npz"
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

print(f"Loading data from {data_file}")
data = np.load(data_file, allow_pickle=True)

X_val = data['X_val']
y_val_dir = data['y_val_dir']
X_test = data['X_test']
y_test_dir = data['y_test_dir']
feature_names = data['feature_names']

print(f"Validation: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")

# Convert target to binary (1 if >= 0, else 0)
y_val_binary = (y_val_dir >= 0).astype(int)
y_test_binary = (y_test_dir >= 0).astype(int)

# Create DataFrames for rule application
val_df = pd.DataFrame(X_val, columns=feature_names)
val_df["target"] = y_val_dir
val_df["target_binary"] = y_val_binary

test_df = pd.DataFrame(X_test, columns=feature_names)
test_df["target"] = y_test_dir
test_df["target_binary"] = y_test_binary

# -----------------------------
# Load trained Decision Tree
# -----------------------------
model_file = model_path / f"decision_tree_h{horizon}m.pkl"
if not model_file.exists():
    raise FileNotFoundError(f"Model file not found: {model_file}. Train it first with 02_decision_tree.py.")

with open(model_file, "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
feature_cols = bundle.get("feature_cols", list(feature_names))

# Ensure feature order matches
if list(feature_cols) != list(feature_names):
    print(f"[WARN] Feature order mismatch. Using order from model: {len(feature_cols)} features")
    # Reorder DataFrame columns to match model
    val_df = val_df[feature_cols + ["target", "target_binary"]]
    test_df = test_df[feature_cols + ["target", "target_binary"]]

# -----------------------------
# Tree stats and per-node rule evaluation
# -----------------------------
# Compute and save per-node stats
stats_df = get_tree_stats(clf, feature_cols=feature_cols)
# Filter to class==1 nodes and sort by impurity desc
stats_pos = stats_df[stats_df['class'] == 1].sort_values(by='impurity', ascending=False)

stats_csv = data_out_dir / f"tree_stats_h{horizon}m.csv"
stats_pos.to_csv(stats_csv, index=False)
print(f"[STATS] Saved node stats (class==1) to: {stats_csv}")

node_ids = stats_pos['node_id'].tolist()

# Helper to print subset stats for a split
def print_subset_stats(df: pd.DataFrame, split_name: str):
    if df.empty:
        print(f"[RULES] {split_name}: no data loaded.")
        return
    for node_id in node_ids[:200]:  # cap to avoid too verbose output
        rules = get_decision_path(clf, feature_cols, node_id)
        df_f = apply_decision_rules(df, rules)
        n = len(df_f)
        mean_bin = float(df_f['target_binary'].mean()) if n > 0 else 0.0
        mean_cont = float(df_f['target'].mean()) if n > 0 else 0.0
        print(f"Node {node_id}: rules={rules} | {split_name} samples={n} | mean_bin={mean_bin:.4f} | mean_target={mean_cont:.6f} | n_rules={len(rules)}")

# Compute and store subset stats per node as DataFrame(s)
def compute_subset_stats_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    cols = [
        "split",
        "node_id",
        "n_samples",
        "mean_target_binary",
        "mean_target",
        "n_rules",
        "rules",
    ]
    if df.empty:
        print(f"[RULES] {split_name}: no data loaded (returning empty stats df).")
        return pd.DataFrame(columns=cols)

    rows = []
    for node_id in node_ids:  # compute for all selected nodes
        rules = get_decision_path(clf, feature_cols, node_id)
        df_f = apply_decision_rules(df, rules)
        n = int(len(df_f))
        mean_bin = float(df_f['target_binary'].mean()) if n > 0 else 0.0
        mean_cont = float(df_f['target'].mean()) if n > 0 else 0.0
        rows.append({
            "split": split_name,
            "node_id": node_id,
            "n_samples": n,
            "mean_target_binary": mean_bin,
            "mean_target": mean_cont,
            "n_rules": len(rules),
            "rules": str(rules),  # Convert list to string for CSV
        })
    df_stats = pd.DataFrame(rows, columns=cols)
    return df_stats

# Print a concise console view (capped) and save full DataFrames
print("\n" + "="*80)
print("PER-NODE STATISTICS (Validation)")
print("="*80)
print_subset_stats(val_df, "validation")

print("\n" + "="*80)
print("PER-NODE STATISTICS (Test)")
print("="*80)
print_subset_stats(test_df, "test")

val_node_stats = compute_subset_stats_df(val_df, "validation")
val_stats_csv = data_out_dir / f"node_subset_stats_validation_h{horizon}m.csv"
val_node_stats.to_csv(val_stats_csv, index=False)
print(f"\n[STATS] Saved validation per-node subset stats to: {val_stats_csv}")

test_node_stats = compute_subset_stats_df(test_df, "test")
test_stats_csv = data_out_dir / f"node_subset_stats_test_h{horizon}m.csv"
test_node_stats.to_csv(test_stats_csv, index=False)
print(f"[STATS] Saved test per-node subset stats to: {test_stats_csv}")

# Optional: also save a combined view for convenience
both_node_stats = pd.concat([val_node_stats, test_node_stats], ignore_index=True)
both_stats_csv = data_out_dir / f"node_subset_stats_combined_h{horizon}m.csv"
both_node_stats.to_csv(both_stats_csv, index=False)
print(f"[STATS] Saved combined per-node subset stats to: {both_stats_csv}")

# -----------------------------
# Overall metrics (accuracy, confusion matrix) for validation and test
# -----------------------------
print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

if not val_df.empty:
    y_true_v = val_df["target_binary"].astype(int).values
    y_pred_v = clf.predict(val_df[feature_cols].values)
    acc_v = accuracy_score(y_true_v, y_pred_v)
    cm_v = confusion_matrix(y_true_v, y_pred_v)
    print(f"\n[VALIDATION]")
    print(f"  Accuracy: {acc_v:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm_v}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true_v, y_pred_v, target_names=['Down/Flat', 'Up']))
else:
    print("[VALIDATION] No data for validation metrics.")

if not test_df.empty:
    y_true_t = test_df["target_binary"].astype(int).values
    y_pred_t = clf.predict(test_df[feature_cols].values)
    acc_t = accuracy_score(y_true_t, y_pred_t)
    cm_t = confusion_matrix(y_true_t, y_pred_t)
    print(f"\n[TEST]")
    print(f"  Accuracy: {acc_t:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm_t}")
    print(f"\n  Classification Report:")
    print(classification_report(y_true_t, y_pred_t, target_names=['Down/Flat', 'Up']))
else:
    print("[TEST] No data for test metrics.")

print("\n[DONE] Decision Tree evaluation completed.")

