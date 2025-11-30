"""
Feature engineering for GRXEUR trend prediction.

This module computes technical analysis features:
- Normalized close price and 1-minute returns
- Exponential Moving Averages (EMA) over multiple periods
- Slopes and second-order slopes of EMAs
- Z-normalized features
- Optionally: intraday time features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def generate_features(
    df: pd.DataFrame,
    ema_periods: List[int],
    slope_periods: List[int],
    z_norm_window: int,
    price_col: str = "close",
    volume_col: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate technical analysis features for trend prediction.
    
    Args:
        df: DataFrame with timestamp index and price/volume columns
        ema_periods: List of EMA periods in minutes
        slope_periods: List of periods for slope calculation in minutes
        z_norm_window: Window size for z-normalization in minutes
        price_col: Name of the price column
        volume_col: Name of the volume column (optional, can be None)
        
    Returns:
        Tuple of (DataFrame with features, list of feature names)
    """
    df = df.copy()
    feature_list = []
    
    # Handle timestamp: can be column or index
    has_timestamp_column = "timestamp" in df.columns
    has_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
    if has_timestamp_column:
        # Keep timestamp as column, use integer index for computation
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        timestamp_for_features = df["timestamp"]
    elif has_datetime_index:
        # Use datetime index for features
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        timestamp_for_features = df.index
    else:
        raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")
    
    prices = df[price_col]
    
    # 1. Normalized close price (relative to rolling mean)
    rolling_mean = prices.rolling(window=z_norm_window, min_periods=1).mean()
    df["price_normalized"] = prices / rolling_mean - 1.0
    feature_list.append("price_normalized")
    
    # 2. 1-minute returns
    df["return_1m"] = prices.pct_change()
    feature_list.append("return_1m")
    
    # 3. Exponential Moving Averages (EMA)
    for period in ema_periods:
        ema_col = f"ema_{period}m"
        df[ema_col] = prices.ewm(span=period, adjust=False).mean()
        
        # Normalized EMA (relative to price)
        ema_norm_col = f"ema_{period}m_normalized"
        df[ema_norm_col] = (df[ema_col] / prices) - 1.0
        feature_list.append(ema_norm_col)
        
        # Z-normalized EMA
        ema_z_col = f"ema_{period}m_z"
        rolling_mean_ema = df[ema_col].rolling(window=z_norm_window, min_periods=1).mean()
        rolling_std_ema = df[ema_col].rolling(window=z_norm_window, min_periods=1).std()
        df[ema_z_col] = (df[ema_col] - rolling_mean_ema) / (rolling_std_ema + 1e-8)
        feature_list.append(ema_z_col)
    
    # 4. Slopes of EMAs (first-order derivative)
    for period in slope_periods:
        ema_col = f"ema_{period}m"
        if ema_col not in df.columns:
            continue
        
        # Compute slope as first difference divided by time step
        slope_col = f"slope_ema_{period}m"
        df[slope_col] = df[ema_col].diff()
        
        # Normalize slope by price level
        slope_norm_col = f"slope_ema_{period}m_normalized"
        df[slope_norm_col] = df[slope_col] / (prices + 1e-8)
        feature_list.append(slope_norm_col)
        
        # Second-order slope (acceleration)
        slope2_col = f"slope2_ema_{period}m"
        df[slope2_col] = df[slope_col].diff()
        
        slope2_norm_col = f"slope2_ema_{period}m_normalized"
        df[slope2_norm_col] = df[slope2_col] / (prices + 1e-8)
        feature_list.append(slope2_norm_col)
    
    # 5. Z-normalized close price
    rolling_mean_price = prices.rolling(window=z_norm_window, min_periods=1).mean()
    rolling_std_price = prices.rolling(window=z_norm_window, min_periods=1).std()
    df["price_z"] = (prices - rolling_mean_price) / (rolling_std_price + 1e-8)
    feature_list.append("price_z")
    
    # 6. Additional price-based features
    if "high" in df.columns and "low" in df.columns:
        # Price range (high - low) normalized
        df["price_range"] = (df["high"] - df["low"]) / (prices + 1e-8)
        feature_list.append("price_range")
    
    # 7. Time-based features (intraday patterns)
    if has_timestamp_column:
        # Use timestamp column (Series with .dt accessor)
        # Minute of day (0-1439)
        df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
        feature_list.append("minute_of_day")
        
        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        feature_list.append("day_of_week")
        
        # Hour of day
        df["hour_of_day"] = df["timestamp"].dt.hour
        feature_list.append("hour_of_day")
    elif has_datetime_index:
        # Use datetime index
        # Minute of day (0-1439)
        df["minute_of_day"] = df.index.hour * 60 + df.index.minute
        feature_list.append("minute_of_day")
        
        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = df.index.dayofweek
        feature_list.append("day_of_week")
        
        # Hour of day
        df["hour_of_day"] = df.index.hour
        feature_list.append("hour_of_day")
    
    # Ensure timestamp is a column (not index) for downstream processing
    if not has_timestamp_column and has_datetime_index:
        df = df.reset_index()
        # Rename the datetime index column to timestamp if it exists
        if df.columns[0] != "timestamp":
            df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    elif has_timestamp_column:
        # Timestamp is already a column, make sure it's not also the index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index(drop=True)
    
    print(f"  Generated {len(feature_list)} features")
    
    return df, feature_list

