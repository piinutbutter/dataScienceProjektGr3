"""
Feed-forward neural network training for GRXEUR trend prediction.

This script:
- Loads ML-ready data from .npz files for a specific horizon
- Builds a simple MLP regressor (configurable hidden sizes/dropout)
- Trains with MSE loss and tracks both validation loss and sign-accuracy metric
- Saves best checkpoints for lowest validation loss and best validation accuracy
- Logs per-epoch metrics and saves training plots

Usage:
- Run from project root: python experiment/scripts/06_model_training/01_feed_forward.py
- Specify horizon as command line argument: python 01_feed_forward.py 15
"""

import sys
import os
from pathlib import Path
import numpy as np
import yaml
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # headless-safe backend
import matplotlib.pyplot as plt
import logging

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

# Ensure model path exists and set up logging
os.makedirs(model_path, exist_ok=True)
log_file = model_path / f'training_h{horizon}m.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    filename=str(log_file),
    filemode='a'
)
logger = logging.getLogger(__name__)

# Load ML-ready data
data_file = processed_path / f"{symbol}_h{horizon}m_ml_ready.npz"
if not data_file.exists():
    raise FileNotFoundError(f"Data file not found: {data_file}")

print(f"Loading data from {data_file}")
data = np.load(data_file, allow_pickle=True)

X_train = data['X_train']
y_train_dir = data['y_train_dir']
y_train_trend = data['y_train_trend']
X_val = data['X_val']
y_val_dir = data['y_val_dir']
y_val_trend = data['y_val_trend']
X_test = data['X_test']
y_test_dir = data['y_test_dir']
y_test_trend = data['y_test_trend']
feature_names = data['feature_names']

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Features: {len(feature_names)}")

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_trend, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val_trend, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_trend, dtype=torch.float32).unsqueeze(1)

# Create DataLoaders
batch_size = params["MODELING"].get("BATCH_SIZE", 2048)
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model
in_dim = len(feature_names)
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


model = MLP(in_dim, hidden1, hidden2, dropout_p).to(device)

# Loss & Optimizer
criterion = nn.MSELoss()
lr = params['MODELING'].get('LR', 1e-3)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=lr, 
    weight_decay=params['MODELING'].get('WEIGHT_DECAY', 1e-4)
)

# Training Loop with Early Stopping
epochs = params['MODELING'].get('EPOCHS', 25)
patience = params['MODELING'].get('PATIENCE', 5)
best_val_loss = np.inf
best_val_acc = 0.0
best_val_epoch = 0
no_improve = 0

# Metric histories
epoch_hist = []
val_acc_hist = []
val_acc_rand_hist = []
val_loss_hist = []

print(f"\nStarting training for horizon {horizon}m...")
print(f"Model: {in_dim} -> {hidden1} -> {hidden2} -> 1")
print(f"Epochs: {epochs}, Patience: {patience}\n")

for epoch in range(1, epochs + 1):
    # Train
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_correct, val_total = 0, 0
    yb_binary_sum = 0
    
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            
            # Transform to binary for accuracy calculation
            yb_binary = (yb >= 0).float()
            yb_binary_sum += yb_binary.sum().item()
            
            val_correct += (logits >= 0).eq(yb_binary).sum().item()
            val_total += yb.numel()
            
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
    
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_acc_rand = yb_binary_sum / val_total
    
    # Console output
    print(f"Epoch {epoch:03d} | Val acc={val_acc:.6f} | Best val acc={best_val_acc:.6f} | "
          f"Val acc rand={val_acc_rand:.6f} | Val loss={val_loss:.6f} | Best val loss={best_val_loss:.6f}")
    
    # Log to file
    logger.info(
        f"Epoch {epoch:03d} | Val acc={val_acc:.6f} | Best val acc={best_val_acc:.6f} | "
        f"Val acc rand={val_acc_rand:.6f} | Val loss={val_loss:.6f} | Best val loss={best_val_loss:.6f}"
    )
    
    # Save metrics
    epoch_hist.append(epoch)
    val_acc_hist.append(val_acc)
    val_acc_rand_hist.append(val_acc_rand)
    val_loss_hist.append(val_loss)
    
    # Save best accuracy model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_val_epoch = epoch
        best_acc_state = model.state_dict()
        torch.save({
            "model_state_dict": best_acc_state,
            "in_dim": in_dim,
            "feature_cols": list(feature_names),
            "config": {"hidden1": hidden1, "hidden2": hidden2, "dropout": dropout_p},
            "horizon": horizon
        }, model_path / f"best_acc_model_h{horizon}m.pt")
        logger.info(f"Saved best-accuracy checkpoint at epoch {epoch}")
    
    # Save best loss model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = model.state_dict()
        no_improve = 0
        torch.save({
            "model_state_dict": best_state,
            "in_dim": in_dim,
            "feature_cols": list(feature_names),
            "config": {"hidden1": hidden1, "hidden2": hidden2, "dropout": dropout_p},
            "horizon": horizon
        }, model_path / f"best_loss_model_h{horizon}m.pt")
        logger.info(f"Saved best-loss checkpoint at epoch {epoch}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            logger.info(f"Early stopping at epoch {epoch}")
            break

# Plot metrics
try:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Accuracies
    ax1.plot(epoch_hist, val_acc_hist, label='Val acc', color='tab:blue')
    ax1.plot(epoch_hist, val_acc_rand_hist, label='Val acc rand', color='tab:orange', linestyle='--')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, linestyle=':')
    ax1.set_title(f'Training Metrics - Horizon {horizon}m')
    
    # Loss
    ax2.plot(epoch_hist, val_loss_hist, label='Val loss', color='tab:red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':')
    
    fig.tight_layout()
    metrics_png = model_path / f'training_metrics_h{horizon}m.png'
    fig.savefig(metrics_png)
    plt.close(fig)
    print(f"\nSaved training metrics plot to: {metrics_png}")
    logger.info(f"Saved training metrics plot to: {metrics_png}")
except Exception as e:
    logger.exception(f"Failed to plot/save training metrics: {e}")

print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.6f} at epoch {best_val_epoch}")

