"""
Train a Logistic Regression baseline model on the ML-ready data.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Trains sklearn.linear_model.LogisticRegression as a baseline
- Evaluates on train/val/test sets
- Provides interpretable coefficients

Usage:
- Run from project root: python experiment/scripts/06_model_training/04_logistic_regression.py
- Specify horizon as command line argument: python 04_logistic_regression.py 15
"""

import sys
import os
from pathlib import Path
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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
y_train_binary = (y_train_dir >= 0).astype(int)
y_val_binary = (y_val_dir >= 0).astype(int)
y_test_binary = (y_test_dir >= 0).astype(int)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
C = params['MODELING'].get('LR_C', 1.0)
max_iter = params['MODELING'].get('LR_MAX_ITER', 1000)
random_state = params['MODELING'].get('RANDOM_STATE', 42)

print(f"\nTraining LogisticRegression...")
print(f"Parameters: C={C}, max_iter={max_iter}")

clf = LogisticRegression(
    C=C,
    max_iter=max_iter,
    random_state=random_state,
    n_jobs=-1,
    solver='lbfgs'  # Good default for most cases
)

clf.fit(X_train_scaled, y_train_binary)
print("Training completed.")

# Evaluate
train_pred = clf.predict(X_train_scaled)
train_acc = accuracy_score(y_train_binary, train_pred)
val_pred = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val_binary, val_pred)
test_pred = clf.predict(X_test_scaled)
test_acc = accuracy_score(y_test_binary, test_pred)

print(f"\nMetrics:")
print(f"  Train accuracy: {train_acc:.6f}")
print(f"  Validation accuracy: {val_acc:.6f}")
print(f"  Test accuracy: {test_acc:.6f}")

# Classification report
print(f"\nTest Set Classification Report:")
print(classification_report(y_test_binary, test_pred, target_names=['Down/Flat', 'Up']))

# Feature coefficients (interpretability)
coefficients = pd.DataFrame({
    'feature': feature_names,
    'coefficient': clf.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nTop 10 Features by Absolute Coefficient:")
print(coefficients.head(10).to_string(index=False))

# Save model
out_path = model_path / f"logistic_regression_h{horizon}m.pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "model": clf,
        "scaler": scaler,
        "feature_cols": list(feature_names),
        "horizon": horizon,
        "metrics": {
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "test_accuracy": float(test_acc)
        },
        "coefficients": coefficients.to_dict('records')
    }, f)
print(f"\nSaved LogisticRegression to: {out_path}")

# Save coefficients to CSV
coef_path = model_path / f"logistic_regression_h{horizon}m_coefficients.csv"
coefficients.to_csv(coef_path, index=False)
print(f"Saved coefficients to: {coef_path}")

print("\nLogistic Regression training completed!")

