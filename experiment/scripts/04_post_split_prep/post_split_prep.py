"""
Post-split data preparation for GRXEUR trend prediction.

Dieses Skript:
- lädt Konfiguration aus experiment/conf/params.yaml
- lädt die vorverarbeiteten Parquet-Dateien (Train/Validation/Test)
- lädt die Feature-Liste aus features.txt
- wählt Feature-Spalten und Zielvariablen (Targets)
- mischt (shuffle) die Daten innerhalb jedes Splits mit EINER Permutation
- speichert ML-fertige Arrays (X, y) als .npz-Dateien
- gibt Beispielzeilen der Features und Targets im Terminal aus

Voraussetzungen:
- Step 3 (pre-split prep) wurde ausgeführt
- Dateien existieren unter: <PROCESSED_PATH>/GRXEUR_train.parquet usw.
- <PROCESSED_PATH>/features.txt enthält eine Feature-Liste (eine Zeile pro Feature)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


# 1) Konfiguration und Pfade laden

# Projektroot automatisch bestimmen:
# .../dataScienceProjektGr3/experiment/scripts/04_post_split_prep/post_split_prep.py
# -> parent.parent.parent.parent = dataScienceProjektGr3
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

params_path = PROJECT_ROOT / "experiment" / "conf" / "params.yaml"
if not params_path.exists():
    raise FileNotFoundError(f"params.yaml not found at: {params_path}")

params = yaml.safe_load(open(params_path))

# Pfad, in dem die pre-split vorbereiteten Parquet-Dateien liegen
processed_path = PROJECT_ROOT / params["DATA_PREP"]["PROCESSED_PATH"]
processed_path.mkdir(parents=True, exist_ok=True)

# Optional separater Output-Pfad für ML-Arrays
ml_output_path = PROJECT_ROOT / params["DATA_PREP"].get(
    "ML_READY_PATH", params["DATA_PREP"]["PROCESSED_PATH"]
)
ml_output_path.mkdir(parents=True, exist_ok=True)

prediction_periods = params["DATA_PREP"]["PREDICTION_PERIODS"]
random_state = params["DATA_PREP"].get("RANDOM_STATE", 42)

symbol = "GRXEUR"

# Feature-Liste laden
features_file = processed_path / "features.txt"
if not features_file.exists():
    raise FileNotFoundError(f"Feature list not found: {features_file}")

with open(features_file, "r") as f:
    feature_list = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(feature_list)} features from {features_file}")


# 2) Helper: Split laden und in X, y aufteilen
def load_split_as_xy(split_name: str, horizon: int):
    """
    Lädt einen Split (train/validation/test) und extrahiert Features und Targets
    für einen bestimmten Vorhersagehorizont.

    Args:
        split_name: 'train', 'validation' oder 'test'
        horizon: Vorhersagehorizont in Minuten (z.B. 5, 10, 15, 30, 60)

    Returns:
        X: Feature-Matrix (num_samples, num_features)
        y_dir: Klassifikationsziel (Richtung: -1, 0, 1)
        y_trend: Regressionsziel (normierte Trend-Slope)
    """
    parquet_file = processed_path / f"{symbol}_{split_name}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Split file not found: {parquet_file}")

    print(f"\nLoading {split_name} split from {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Sicherstellen, dass alle Features vorhanden sind
    missing_feats = [feat for feat in feature_list if feat not in df.columns]
    if missing_feats:
        raise ValueError(
            f"The following features from features.txt are missing in {split_name} data: {missing_feats}"
        )

    # Zielspalten für den gewünschten Horizont
    trend_col = f"target_trend_{horizon}m"
    direction_col = f"target_direction_{horizon}m"

    if trend_col not in df.columns or direction_col not in df.columns:
        raise ValueError(
            f"Missing target columns {trend_col} or {direction_col} in {split_name} data"
        )

    # Feature-Matrix X und Zielvektoren y
    X = df[feature_list].to_numpy(dtype=float)
    y_trend = df[trend_col].to_numpy(dtype=float)
    y_dir = df[direction_col].to_numpy(dtype=int)

    print(
        f"  {split_name}: {X.shape[0]} samples, {X.shape[1]} features for horizon {horizon}m"
    )

    # Beispielausgabe: erste Zeile der Features (nur erste 10, damit es lesbar bleibt)
    print("  Sample features row 0:")
    print(df[feature_list].iloc[0].head(10))

    # Beispielausgabe: Targets der ersten Zeile
    print("  Sample targets row 0:")
    print(
        {
            trend_col: df[trend_col].iloc[0],
            direction_col: df[direction_col].iloc[0],
        }
    )

    return X, y_dir, y_trend


# 3) Für jeden Horizont ML-fertige Arrays erzeugen

# Ein RNG einmal mit Seed erzeugen (für reproduzierbares Shuffling)
rng = np.random.default_rng(random_state)

for horizon in prediction_periods:
    print(f"\n=== Preparing ML data for horizon {horizon}m ===")

    # ---------------------- Train ----------------------
    X_train, y_train_dir, y_train_trend = load_split_as_xy("train", horizon)
    perm = rng.permutation(len(X_train))  # eine Permutation für alle Train-Arrays
    X_train = X_train[perm]
    y_train_dir = y_train_dir[perm]
    y_train_trend = y_train_trend[perm]

    # ------------------- Validation --------------------
    X_val, y_val_dir, y_val_trend = load_split_as_xy("validation", horizon)
    perm = rng.permutation(len(X_val))
    X_val = X_val[perm]
    y_val_dir = y_val_dir[perm]
    y_val_trend = y_val_trend[perm]

    # ---------------------- Test -----------------------
    # Hinweis: Testdaten könnte man auch ungeshuffelt lassen,
    # hier wird aber für Konsistenz ebenfalls gemischt.
    X_test, y_test_dir, y_test_trend = load_split_as_xy("test", horizon)
    perm = rng.permutation(len(X_test))
    X_test = X_test[perm]
    y_test_dir = y_test_dir[perm]
    y_test_trend = y_test_trend[perm]

    # 4) Speichern als .npz (NumPy-kompatible ML-Files)
    out_file = ml_output_path / f"{symbol}_h{horizon}m_ml_ready.npz"
    np.savez_compressed(
        out_file,
        X_train=X_train,
        y_train_dir=y_train_dir,
        y_train_trend=y_train_trend,
        X_val=X_val,
        y_val_dir=y_val_dir,
        y_val_trend=y_val_trend,
        X_test=X_test,
        y_test_dir=y_test_dir,
        y_test_trend=y_test_trend,
        feature_names=np.array(feature_list),
    )

    print(f"  Saved ML-ready arrays to {out_file}")

print("\nPost-split data preparation completed.")
