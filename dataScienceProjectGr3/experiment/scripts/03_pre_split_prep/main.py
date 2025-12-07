"""
End-to-end pre-split data preparation for the GRXEUR index feed.

This script:
- loads params.yaml for config
- loads a list of symbols (in your case: only GRXEUR, but can extend later)
- loads 1m Parquet bars
- computes forward-looking trend targets
- computes TA features (EMA, slopes, 2nd order slopes, z-norm, etc.)
- drops NaNs
- performs chronological splits
- writes train/val/test Parquet files
- writes features.txt once
"""

import os
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

import targets
import features

# -------------------------------------------------------------
# Load configuration
# -------------------------------------------------------------
# Get project root (go up from scripts/03_pre_split_prep to project root)
project_root = Path(__file__).parent.parent.parent.parent
params_path = project_root / "experiment" / "conf" / "params.yaml"

if not params_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {params_path}")

params = yaml.safe_load(open(params_path))

prediction_periods = params['DATA_PREP']['PREDICTION_PERIODS']
ema_periods = params['DATA_PREP']['EMA_PERIODS']
slope_periods = params['DATA_PREP']['SLOPE_PERIODS']
z_norm_window = params['DATA_PREP']['Z_NORM_WINDOW']

data_path = Path(project_root) / params['DATA_ACQUISITION']['DATA_PATH']
processed_path = Path(project_root) / params['DATA_PREP']['PROCESSED_PATH']
os.makedirs(processed_path, exist_ok=True)

train_date = params['DATA_PREP']['TRAIN_DATE']
validation_date = params['DATA_PREP']['VALIDATION_DATE']
test_date = params['DATA_PREP']['TEST_DATE']

# -------------------------------------------------------------
# In deinem Projekt haben wir nur 1 Symbol: GRXEUR
# (Aber die Struktur lässt Erweiterung zu)
# -------------------------------------------------------------
symbols = ["GRXEUR"]

# -------------------------------------------------------------
# Process each symbol
# -------------------------------------------------------------
for symbol in symbols:

    print(f"Processing {symbol}")

    # Load Parquet feed (actual file structure: Bars_1m_GRXEUR/GRXEUR_M1_2010_2018.parquet)
    bars_file = data_path / "Bars_1m_GRXEUR" / f"{symbol}_M1_2010_2018.parquet"
    
    if not bars_file.exists():
        print(f"WARNING: File not found: {bars_file}")
        print(f"  Trying alternative path...")
        bars_file = data_path / "Bars_1m_GRXEUR" / f"{symbol}.parquet"
        if not bars_file.exists():
            raise FileNotFoundError(f"Could not find data file for {symbol} in {data_path / 'Bars_1m_GRXEUR'}")
    
    print(f"  Loading from: {bars_file}")
    df = pd.read_parquet(bars_file)
    df["symbol"] = symbol

    # Handle timestamp: could be index or column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        elif df.index.name:
            df.rename(columns={df.index.name: "timestamp"}, inplace=True)
        else:
            df["timestamp"] = df.index
            df = df.reset_index(drop=True)
    
    # Ensure timestamp is datetime column
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a timestamp column or datetime index")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    print(f"  Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------
    df = targets.add_normalized_trend_direction(
        df,
        prediction_periods=prediction_periods,
        price_col="close"
    )

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------
    df, feature_list = features.generate_features(
        df,
        ema_periods=ema_periods,
        slope_periods=slope_periods,
        z_norm_window=z_norm_window,
        price_col="close",
        volume_col=None  # Index feed → no volume
    )

    # Remove feature/target rows with NaN
    print(f"  Dropping rows with NaN values...")
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    print(f"  Dropped {dropped} rows with NaN ({dropped/initial_len*100:.1f}%)")

    # ---------------------------------------------------------
    # Save feature names
    # ---------------------------------------------------------
    features_file = processed_path / "features.txt"
    if not features_file.exists():
        with open(features_file, "w") as f:
            for feat in feature_list:
                f.write(f"{feat}\n")
        print(f"  Saved feature list to {features_file}")

    # ---------------------------------------------------------
    # Chronological split
    # ---------------------------------------------------------
    train_date_dt = pd.to_datetime(train_date)
    validation_date_dt = pd.to_datetime(validation_date)
    test_date_dt = pd.to_datetime(test_date)
    
    train = df[df['timestamp'] <= train_date_dt].copy()
    val = df[(df['timestamp'] > train_date_dt) & (df['timestamp'] <= validation_date_dt)].copy()
    test = df[(df['timestamp'] > validation_date_dt) & (df['timestamp'] <= test_date_dt)].copy()

    # ---------------------------------------------------------
    # Persist
    # ---------------------------------------------------------
    train_file = processed_path / f"{symbol}_train.parquet"
    val_file = processed_path / f"{symbol}_validation.parquet"
    test_file = processed_path / f"{symbol}_test.parquet"
    
    train.to_parquet(train_file, index=False)
    val.to_parquet(val_file, index=False)
    test.to_parquet(test_file, index=False)

    print(f"Finished {symbol}:")
    print(f"  Train: {len(train)} rows ({train['timestamp'].min()} to {train['timestamp'].max()})")
    print(f"  Validation: {len(val)} rows ({val['timestamp'].min()} to {val['timestamp'].max()})")
    print(f"  Test: {len(test)} rows ({test['timestamp'].min()} to {test['timestamp'].max()})")
    print(f"  Files saved to: {processed_path}")

print("\nPre-split data preparation completed!")
