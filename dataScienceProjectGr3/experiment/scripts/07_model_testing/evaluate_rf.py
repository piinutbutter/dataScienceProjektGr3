"""
Evaluate a trained Random Forest on validation and test splits.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Loads the trained Random Forest model
- Computes overall metrics (accuracy, confusion matrix) for validation and test
- Provides detailed classification reports

Usage:
- Run from project root: python experiment/scripts/07_model_testing/evaluate_rf.py
- Specify horizon as command line argument: python evaluate_rf.py 15
"""

import sys
import os
import pickle
import yaml
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# -----------------------------
# Load trained Random Forest
# -----------------------------
model_file = model_path / f"random_forest_h{horizon}m.pkl"
if not model_file.exists():
    raise FileNotFoundError(f"Model file not found: {model_file}. Train it first with 03_random_forest.py.")

with open(model_file, "rb") as f:
    bundle = pickle.load(f)

clf = bundle["model"]
feature_cols = bundle.get("feature_cols", list(feature_names))

# Ensure feature order matches
if list(feature_cols) != list(feature_names):
    print(f"[WARN] Feature order mismatch. Using order from model: {len(feature_cols)} features")
    # Reorder data to match model
    feature_idx_map = {name: idx for idx, name in enumerate(feature_names)}
    X_val = X_val[:, [feature_idx_map[name] for name in feature_cols]]
    X_test = X_test[:, [feature_idx_map[name] for name in feature_cols]]

# -----------------------------
# Evaluation
# -----------------------------
print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

# Validation
if len(X_val) > 0:
    y_pred_val = clf.predict(X_val)
    val_acc = accuracy_score(y_val_binary, y_pred_val)
    val_cm = confusion_matrix(y_val_binary, y_pred_val)
    
    print(f"\n[VALIDATION]")
    print(f"  Accuracy: {val_acc:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {val_cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_val_binary, y_pred_val, target_names=['Down/Flat', 'Up']))
else:
    print("[VALIDATION] No data for validation metrics.")

# Test
if len(X_test) > 0:
    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_test_binary, y_pred_test)
    test_cm = confusion_matrix(y_test_binary, y_pred_test)
    
    print(f"\n[TEST]")
    print(f"  Accuracy: {test_acc:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {test_cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_binary, y_pred_test, target_names=['Down/Flat', 'Up']))
else:
    print("[TEST] No data for test metrics.")

# Print feature importance summary
if "feature_importance" in bundle:
    print("\n" + "="*80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*80)
    importance_list = bundle["feature_importance"]
    # Sort by importance (descending)
    sorted_importance = sorted(importance_list, key=lambda x: x['importance'], reverse=True)
    for i, item in enumerate(sorted_importance[:10], 1):
        print(f"  {i:2d}. {item['feature']:30s} : {item['importance']:.6f}")

print("\n[DONE] Random Forest evaluation completed.")

