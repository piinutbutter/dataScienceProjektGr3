"""
Evaluate a trained Feed Forward Neural Network on validation and test splits.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Loads the trained Neural Network model checkpoint
- Computes overall metrics (accuracy, confusion matrix) for validation and test
- Provides detailed classification reports

Usage:
- Run from project root: python experiment/scripts/07_model_testing/evaluate_nn.py
- Specify horizon as command line argument: python evaluate_nn.py 15
"""

import sys
import os
import yaml
import numpy as np
import torch
from torch import nn
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

# Convert to tensors
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# -----------------------------
# Define MLP architecture and load checkpoint
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hidden1 = params['MODELING'].get('HIDDEN1', 128)
hidden2 = params['MODELING'].get('HIDDEN2', 64)
dropout_p = params['MODELING'].get('DROPOUT', 0.1)

class MLP(nn.Module):
    """Simple 2-hidden-layer MLP regressor."""
    def __init__(self, in_dim, h1, h2, dropout_p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(h1, h2),
            nn.Dropout(dropout_p),
            nn.Linear(h2, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load checkpoint
checkpoint_path = model_path / f"best_acc_model_h{horizon}m.pt"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train it first with 01_feed_forward.py.")

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Get feature order from checkpoint if available
ckpt_features = checkpoint.get("feature_cols")
if ckpt_features and isinstance(ckpt_features, (list, tuple)):
    if list(ckpt_features) != list(feature_names):
        print(f"[WARN] Feature order mismatch. Using order from checkpoint: {len(ckpt_features)} features")
    feature_names = list(ckpt_features)

# Get config from checkpoint
ckpt_cfg = checkpoint.get("config", {})
hidden1 = ckpt_cfg.get('hidden1', hidden1)
hidden2 = ckpt_cfg.get('hidden2', hidden2)
dropout_p = ckpt_cfg.get('dropout', dropout_p)

in_dim = len(feature_names)
model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model: {in_dim} -> {hidden1} -> {hidden2} -> 1")

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_split(X, y_binary, split_name, batch_size=2048):
    """Evaluate model on a split."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(device)
            batch_pred = model(batch_X)
            # Convert to binary predictions (>= 0 -> 1, else 0)
            batch_pred_binary = (batch_pred.squeeze() >= 0).cpu().numpy().astype(int)
            predictions.extend(batch_pred_binary)
    
    predictions = np.array(predictions)
    acc = accuracy_score(y_binary, predictions)
    cm = confusion_matrix(y_binary, predictions)
    
    return acc, cm, predictions

print("\n" + "="*80)
print("OVERALL METRICS")
print("="*80)

# Validation
if len(X_val) > 0:
    val_acc, val_cm, val_pred = evaluate_split(X_val_t, y_val_binary, "validation")
    print(f"\n[VALIDATION]")
    print(f"  Accuracy: {val_acc:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {val_cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_val_binary, val_pred, target_names=['Down/Flat', 'Up']))
else:
    print("[VALIDATION] No data for validation metrics.")

# Test
if len(X_test) > 0:
    test_acc, test_cm, test_pred = evaluate_split(X_test_t, y_test_binary, "test")
    print(f"\n[TEST]")
    print(f"  Accuracy: {test_acc:.6f}")
    print(f"  Confusion Matrix:")
    print(f"    {test_cm}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_binary, test_pred, target_names=['Down/Flat', 'Up']))
else:
    print("[TEST] No data for test metrics.")

print("\n[DONE] Neural Network evaluation completed.")

