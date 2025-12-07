"""
Train a scikit-learn DecisionTree on the ML-ready data.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Trains sklearn.tree.DecisionTreeClassifier on the data
- Saves the trained tree to disk and prints a brief summary

Usage:
- Run from project root: python experiment/scripts/06_model_training/02_decision_tree.py
- Specify horizon as command line argument: python 02_decision_tree.py 15
"""

import sys
import os
from pathlib import Path
import pickle
import yaml
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

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

# Ensure model path exists
os.makedirs(model_path, exist_ok=True)

# Load ML-ready data
data_file = processed_path / f"{symbol}_h{horizon}m_ml_ready.npz"
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

print(f"Loading data from {data_file}")
data = np.load(data_file, allow_pickle=True)

X_train = data['X_train']
y_train_dir = data['y_train_dir']
X_val = data['X_val']
y_val_dir = data['y_val_dir']
X_test = data['X_test']
y_test_dir = data['y_test_dir']
feature_names = data['feature_names']

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")

# Convert target to binary (1 if >= 0, else 0)
# Note: y_train_dir is already -1, 0, 1, so we convert to 0, 1
y_train_binary = (y_train_dir >= 0).astype(int)
y_val_binary = (y_val_dir >= 0).astype(int)
y_test_binary = (y_test_dir >= 0).astype(int)

# Train Decision Tree
max_depth = params['MODELING'].get('DT_MAX_DEPTH', 8)
min_samples_split = params['MODELING'].get('DT_MIN_SAMPLES_SPLIT', 2)
min_samples_leaf = params['MODELING'].get('DT_MIN_SAMPLES_LEAF', 1)
random_state = params['MODELING'].get('RANDOM_STATE', 42)

print(f"\nTraining DecisionTreeClassifier...")
print(f"Parameters: max_depth={max_depth}, min_samples_split={min_samples_split}, "
      f"min_samples_leaf={min_samples_leaf}")

clf = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    random_state=random_state,
)

clf.fit(X_train, y_train_binary)
print("Training completed.")

# Evaluate
train_pred = clf.predict(X_train)
train_acc = accuracy_score(y_train_binary, train_pred)
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val_binary, val_pred)
test_pred = clf.predict(X_test)
test_acc = accuracy_score(y_test_binary, test_pred)

print(f"\nMetrics:")
print(f"  Train accuracy: {train_acc:.6f}")
print(f"  Validation accuracy: {val_acc:.6f}")
print(f"  Test accuracy: {test_acc:.6f}")
print(f"  Tree depth: {clf.get_depth()}")
print(f"  Number of leaves: {clf.get_n_leaves()}")

# Classification report
print(f"\nValidation Classification Report:")
print(classification_report(y_val_binary, val_pred, target_names=['Down/Flat', 'Up']))

# Save model
out_path = model_path / f"decision_tree_h{horizon}m.pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "model": clf,
        "feature_cols": list(feature_names),
        "horizon": horizon,
        "metrics": {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "depth": int(clf.get_depth()),
            "n_leaves": int(clf.get_n_leaves())
        }
    }, f)
print(f"\nSaved DecisionTreeClassifier to: {out_path}")

# Export tree rules (text representation)
try:
    rules_txt = export_text(clf, feature_names=list(feature_names), max_depth=4)
    rules_path = model_path / f"decision_tree_h{horizon}m_rules.txt"
    with open(rules_path, "w", encoding="utf-8") as f:
        f.write(f"Decision Tree Rules for Horizon {horizon}m\n")
        f.write("=" * 80 + "\n\n")
        f.write(rules_txt)
    print(f"Exported tree rules (depth<=4) to: {rules_path}")
except Exception as e:
    print(f"Warning: Could not export tree rules: {e}")

print("\nDecision Tree training completed!")

