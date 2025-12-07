# Model Training

This directory contains scripts for training machine learning models on the GRXEUR trend prediction task.

## Scripts

### 1. Feed Forward Neural Network (`01_feed_forward.py`)

Trains a 2-hidden-layer MLP regressor.

**Usage:**
```bash
python experiment/scripts/06_model_training/01_feed_forward.py [horizon]
```

**Arguments:**
- `horizon`: Prediction horizon in minutes (5, 10, 15, 30, 60). Default: 15

**Example:**
```bash
python experiment/scripts/06_model_training/01_feed_forward.py 15
```

**Output:**
- `best_acc_model_h{horizon}m.pt` - Best accuracy checkpoint
- `best_loss_model_h{horizon}m.pt` - Best loss checkpoint
- `training_h{horizon}m.log` - Training log
- `training_metrics_h{horizon}m.png` - Training curves plot

### 2. Decision Tree (`02_decision_tree.py`)

Trains a scikit-learn DecisionTreeClassifier.

**Usage:**
```bash
python experiment/scripts/06_model_training/02_decision_tree.py [horizon]
```

**Output:**
- `decision_tree_h{horizon}m.pkl` - Trained decision tree
- `decision_tree_h{horizon}m_rules.txt` - Text representation of tree rules

### 3. Random Forest (`04_random_forest.py`)

Trains a scikit-learn RandomForestClassifier (ensemble of decision trees).

**Usage:**
```bash
python experiment/scripts/06_model_training/03_random_forest.py [horizon]
```

**Output:**
- `random_forest_h{horizon}m.pkl` - Trained random forest
- `random_forest_h{horizon}m_feature_importance.csv` - Feature importance ranking

### 4. Logistic Regression (`06_logistic_regression.py`)

Trains a Logistic Regression baseline model.

**Usage:**
```bash
python experiment/scripts/06_model_training/04_logistic_regression.py [horizon]
```

**Output:**
- `logistic_regression_h{horizon}m.pkl` - Trained logistic regression model
- `logistic_regression_h{horizon}m_coefficients.csv` - Feature coefficients (interpretable)

## Configuration

Model parameters can be adjusted in `experiment/conf/params.yaml` under the `MODELING` section:

```yaml
MODELING:
  MODEL_PATH: "experiment/models"
  HIDDEN1: 128
  HIDDEN2: 64
  DROPOUT: 0.1
  LR: 0.001
  WEIGHT_DECAY: 0.0001
  EPOCHS: 25
  PATIENCE: 5
  BATCH_SIZE: 2048
  DT_MAX_DEPTH: 8
  DT_MIN_SAMPLES_SPLIT: 2
  DT_MIN_SAMPLES_LEAF: 1
  RF_N_ESTIMATORS: 100
  RF_MAX_DEPTH: 10
  LR_C: 1.0
  LR_MAX_ITER: 1000
  RANDOM_STATE: 42
```

## Workflow

1. **Train Models (choose one or more):**
   ```bash
   # Neural Network
   python experiment/scripts/06_model_training/01_feed_forward.py 15
   
   # Decision Tree
   python experiment/scripts/06_model_training/02_decision_tree.py 15
   
   # Random Forest (recommended)
   python experiment/scripts/06_model_training/03_random_forest.py 15
   
   # Logistic Regression (baseline)
   python experiment/scripts/06_model_training/04_logistic_regression.py 15
   ```

## Notes

- Models are trained separately for each prediction horizon
- The neural network uses early stopping based on validation loss
- Both models predict the trend direction (binary classification: up vs down/flat)
- All models are saved in `experiment/models/`

