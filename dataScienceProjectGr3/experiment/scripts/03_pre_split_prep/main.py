# language: python
"""
End-to-end pre-split data preparation for the GRXEUR index feed.

Robustere Dateisuche:
- rekursives Suchen nach Parquet / CSV (inkl. .gz)
- prüft mehrere mögliche data_path-Varianten (z.B. experiment/data/raw)
- besseres Fehler-Reporting
- überspringen eines Symbols wenn keine Datei gefunden / lesbar
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import yaml

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

import targets
import features

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------
# Load configuration
# -------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent
params_path = project_root / "experiment" / "conf" / "params.yaml"

if not params_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {params_path}")

params = yaml.safe_load(open(params_path))

prediction_periods = params['DATA_PREP']['PREDICTION_PERIODS']
ema_periods = params['DATA_PREP']['EMA_PERIODS']
slope_periods = params['DATA_PREP']['SLOPE_PERIODS']
z_norm_window = params['DATA_PREP']['Z_NORM_WINDOW']

# Primary data path from params
data_path = Path(project_root) / params['DATA_ACQUISITION']['DATA_PATH']
# Additional candidate: data path inside the experiment directory (handles paths like experiment/data/raw)
alt_data_path = Path(project_root) / "experiment" / params['DATA_ACQUISITION']['DATA_PATH']
# Also consider project_root / data / raw as fallback
legacy_data_path = Path(project_root) / "data" / "raw"

# Build list of unique candidate base paths (keep order of preference)
candidate_base_paths = []
for p in (data_path, alt_data_path, legacy_data_path):
    if p not in candidate_base_paths:
        candidate_base_paths.append(p)

processed_path = Path(project_root) / params['DATA_PREP']['PROCESSED_PATH']
os.makedirs(processed_path, exist_ok=True)

train_date = params['DATA_PREP']['TRAIN_DATE']
validation_date = params['DATA_PREP']['VALIDATION_DATE']
test_date = params['DATA_PREP']['TEST_DATE']

# -------------------------------------------------------------
# Symbols
# -------------------------------------------------------------
symbols = ["GRXEUR"]

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def find_data_file_for_bases(symbol: str, base_paths):
    """
    Suche rekursiv nach passenden Dateien für ein Symbol in mehreren Basis-Pfaden.
    Unterstützt: .parquet, .parquet.gz, .csv, .csv.gz
    Liefert Path oder None.
    """
    patterns = [
        f"**/{symbol}*.parquet",
        f"**/{symbol}*.parquet.gz",
        f"**/{symbol}*.csv",
        f"**/{symbol}*.csv.gz",
        f"**/*{symbol}*.parquet",
        f"**/*{symbol}*.csv",
    ]

    # Try direct candidate directories and candidate filenames inside them first
    for base in base_paths:
        if not base:
            continue
        candidate_dirs = [
            base / f"Bars_1m_{symbol}",
            base / f"Bars_{symbol}",
            base / f"Bars_{symbol}_1m",
            base,
        ]
        for d in candidate_dirs:
            if d.exists() and d.is_dir():
                # try candidate filenames inside directory (non-recursive)
                candidates = [
                    d / f"{symbol}_M1_2010_2018.parquet",
                    d / f"{symbol}.parquet",
                    d / f"{symbol}_1m.parquet",
                    d / f"{symbol}_M1.parquet",
                    d / f"{symbol}_M1_2010.parquet",
                ]
                for p in candidates:
                    if p.exists():
                        return p
                # try glob within the directory (non-recursive)
                for pat in [f"{symbol}*.parquet", f"{symbol}*.csv", f"*{symbol}*.parquet", f"*{symbol}*.csv"]:
                    matches = list(d.glob(pat))
                    if matches:
                        return matches[0]

    # fallback: recursive search from each base_path
    for base in base_paths:
        if base and base.exists():
            for pat in patterns:
                matches = list(base.rglob(pat.replace("**/", ""))) if pat.startswith("**/") else list(base.rglob(pat))
                if matches:
                    return matches[0]

    return None

def read_data_file(path: Path) -> pd.DataFrame:
    """Versucht Datei als parquet oder csv zu lesen."""
    try:
        name = path.name.lower()
        if name.endswith(".parquet") or name.endswith(".parquet.gz"):
            return pd.read_parquet(path)
        elif name.endswith(".csv") or name.endswith(".csv.gz"):
            df = pd.read_csv(path, compression="infer")
            # Try to parse common datetime columns
            for col in ["timestamp", "datetime", "time", "date", "Date"]:
                if col in df.columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass
            return df
        else:
            # fallback: try parquet then csv
            try:
                return pd.read_parquet(path)
            except Exception:
                return pd.read_csv(path, compression="infer")
    except Exception as e:
        raise RuntimeError(f"Failed to read data file {path}: {e}")

# -------------------------------------------------------------
# Process each symbol
# -------------------------------------------------------------
processed_symbols = []
for symbol in symbols:
    logger.info(f"Processing {symbol}")

    bars_file = find_data_file_for_bases(symbol, candidate_base_paths)

    if bars_file is None:
        # Show available top-level files for debugging for each candidate base
        for base in candidate_base_paths:
            if base and base.exists():
                available = [p.name for p in base.glob("*") if p.is_file()]
                logger.error("Could not find data file for %s under %s. Top-level files: %s", symbol, base, available)
            else:
                logger.error("Could not find data file for %s: base path %s does not exist", symbol, base)
        logger.info("Skipping symbol %s", symbol)
        continue

    logger.info("  Loading from: %s", bars_file)
    try:
        df = read_data_file(bars_file)
    except Exception as e:
        logger.error("  Failed to read %s: %s", bars_file, e)
        logger.info("  Skipping symbol %s due to read error", symbol)
        continue

    if df is None or len(df) == 0:
        logger.error("  Loaded empty dataframe for %s from %s", symbol, bars_file)
        logger.info("  Skipping symbol %s", symbol)
        continue

    df["symbol"] = symbol

    # Handle timestamp: could be index or column
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if "datetime" in df.columns:
            df.rename(columns={"datetime": "timestamp"}, inplace=True)
        elif df.columns[0] and df.columns[0] != "timestamp" and isinstance(df.columns[0], str) and "time" in df.columns[0].lower():
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
        elif df.index.name:
            df.rename(columns={df.index.name: "timestamp"}, inplace=True)
        else:
            if "timestamp" not in df.columns:
                df["timestamp"] = pd.to_datetime(df.index)
                df = df.reset_index(drop=True)

    # Ensure timestamp is datetime column
    if "timestamp" not in df.columns:
        candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "timestamp"})
        else:
            logger.error("DataFrame must have a timestamp column or datetime index for %s (found columns: %s)", symbol, list(df.columns))
            logger.info("Skipping symbol %s", symbol)
            continue

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"].notna()].reset_index(drop=True)

    logger.info("  Loaded %d rows from %s to %s", len(df), df["timestamp"].min(), df["timestamp"].max())

    # Targets
    try:
        df = targets.add_normalized_trend_direction(df, prediction_periods=prediction_periods, price_col="close")
    except Exception as e:
        logger.error("  Failed to compute targets for %s: %s", symbol, e)
        logger.info("  Skipping symbol %s", symbol)
        continue

    # Features
    try:
        df, feature_list = features.generate_features(
            df,
            ema_periods=ema_periods,
            slope_periods=slope_periods,
            z_norm_window=z_norm_window,
            price_col="close",
            volume_col=None
        )
    except Exception as e:
        logger.error("  Failed to generate features for %s: %s", symbol, e)
        logger.info("  Skipping symbol %s", symbol)
        continue

    # Drop NaNs
    logger.info("  Dropping rows with NaN values...")
    initial_len = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = initial_len - len(df)
    pct = (dropped / initial_len * 100) if initial_len > 0 else 0.0
    logger.info("  Dropped %d rows with NaN (%.1f%%)", dropped, pct)

    # Save feature names
    features_file = processed_path / "features.txt"
    if not features_file.exists():
        try:
            with open(features_file, "w") as f:
                for feat in feature_list:
                    f.write(f"{feat}\n")
            logger.info("  Saved feature list to %s", features_file)
        except Exception as e:
            logger.warning("  Could not write features file: %s", e)

    # Chronological split
    train_date_dt = pd.to_datetime(train_date)
    validation_date_dt = pd.to_datetime(validation_date)
    test_date_dt = pd.to_datetime(test_date)

    train = df[df['timestamp'] <= train_date_dt].copy()
    val = df[(df['timestamp'] > train_date_dt) & (df['timestamp'] <= validation_date_dt)].copy()
    test = df[(df['timestamp'] > validation_date_dt) & (df['timestamp'] <= test_date_dt)].copy()

    # Persist
    train_file = processed_path / f"{symbol}_train.parquet"
    val_file = processed_path / f"{symbol}_validation.parquet"
    test_file = processed_path / f"{symbol}_test.parquet"

    try:
        train.to_parquet(train_file, index=False)
        val.to_parquet(val_file, index=False)
        test.to_parquet(test_file, index=False)
    except Exception as e:
        logger.error("  Failed to write parquet files for %s: %s", symbol, e)
        logger.info("  Skipping symbol %s", symbol)
        continue

    logger.info("Finished %s:", symbol)
    if len(train):
        logger.info("  Train: %d rows (%s to %s)", len(train), train['timestamp'].min(), train['timestamp'].max())
    else:
        logger.info("  Train: %d rows", len(train))
    if len(val):
        logger.info("  Validation: %d rows (%s to %s)", len(val), val['timestamp'].min(), val['timestamp'].max())
    else:
        logger.info("  Validation: %d rows", len(val))
    if len(test):
        logger.info("  Test: %d rows (%s to %s)", len(test), test['timestamp'].min(), test['timestamp'].max())
    else:
        logger.info("  Test: %d rows", len(test))

    logger.info("  Files saved to: %s", processed_path)
    processed_symbols.append(symbol)

if not processed_symbols:
    logger.warning("No symbols were processed. Please check that your data are available under one of: %s and that filenames contain the symbol names.", candidate_base_paths)
else:
    logger.info("Pre-split data preparation completed for: %s", processed_symbols)