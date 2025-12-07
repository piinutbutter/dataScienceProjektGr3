"""
Feature engineering for GRXEUR trend prediction.

This module computes technical analysis features:
- Normalized close price and 1-minute returns
- Exponential Moving Averages (EMA) over multiple periods
- Slopes and second-order slopes of EMAs
- Z-normalized features
- Momentum and volatility features
- RSI (Relative Strength Index)
- ATR (Average True Range)
- Bollinger Bands (position only)
- MACD (Moving Average Convergence Divergence)
- EMA crossovers
- Lagged returns (1-2 lags)
- Intraday time features
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
        
        # Open-Close spread normalized
        if "open" in df.columns:
            df["oc_spread"] = (df["close"] - df["open"]) / (prices + 1e-8)
            feature_list.append("oc_spread")
    
    # 7. Momentum and Volatility Features (reduced periods)
    # Rolling volatility (std of returns) - only 15m, 30m, 60m
    for vol_window in [15, 30, 60]:
        vol_col = f"volatility_{vol_window}m"
        df[vol_col] = df["return_1m"].rolling(window=vol_window, min_periods=1).std()
        feature_list.append(vol_col)
    
    # Momentum (price change over different periods) - only 15m, 30m, 60m
    for mom_period in [15, 30, 60]:
        mom_col = f"momentum_{mom_period}m"
        df[mom_col] = (prices / prices.shift(mom_period) - 1.0)
        feature_list.append(mom_col)
    
    # 8. RSI (Relative Strength Index) - only one period
    rsi_period = 14
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    feature_list.append("rsi_14")
    
    # 9. ATR (Average True Range) - only one period
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        atr_period = 14
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.rolling(window=atr_period, min_periods=1).mean() / (prices + 1e-8)
        feature_list.append("atr_14")
    
    # 10. Bollinger Bands - only position, one period
    bb_period = 20
    bb_ma = prices.rolling(window=bb_period, min_periods=1).mean()
    bb_std = prices.rolling(window=bb_period, min_periods=1).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    
    # Price position within Bollinger Bands (normalized)
    df["bb_position_20"] = (prices - bb_lower) / (bb_upper - bb_lower + 1e-8)
    feature_list.append("bb_position_20")
    
    # 11. MACD (Moving Average Convergence Divergence)
    if len(ema_periods) >= 2:
        # Use fastest and slowest EMA for MACD
        fast_period = min(ema_periods)
        slow_period = max(ema_periods)
        fast_ema = df[f"ema_{fast_period}m"]
        slow_ema = df[f"ema_{slow_period}m"]
        
        df["macd"] = fast_ema - slow_ema
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        
        # Normalize MACD features
        df["macd_normalized"] = df["macd"] / (prices + 1e-8)
        df["macd_signal_normalized"] = df["macd_signal"] / (prices + 1e-8)
        df["macd_histogram_normalized"] = df["macd_histogram"] / (prices + 1e-8)
        
        feature_list.extend(["macd_normalized", "macd_signal_normalized", "macd_histogram_normalized"])
    
    # 12. EMA Crossover (only main fast-slow crossover) - DISABLED (identical to macd_normalized)
    # if len(ema_periods) >= 2:
    #     fast_period = min(ema_periods)
    #     slow_period = max(ema_periods)
    #     crossover_col = f"ema_crossover_{fast_period}m_{slow_period}m"
    #     df[crossover_col] = (df[f"ema_{fast_period}m"] - df[f"ema_{slow_period}m"]) / (prices + 1e-8)
    #     feature_list.append(crossover_col)
    
    # 13. Lagged Returns (only 1-2 lags)
    for lag in [1, 2]:
        lag_col = f"return_lag_{lag}"
        df[lag_col] = df["return_1m"].shift(lag)
        feature_list.append(lag_col)
    
    # 14. Price distance from recent high/low (only one period)
    dist_period = 30
    if "high" in df.columns and "low" in df.columns:
        high_roll = df["high"].rolling(window=dist_period, min_periods=1).max()
        low_roll = df["low"].rolling(window=dist_period, min_periods=1).min()
    else:
        high_roll = prices.rolling(window=dist_period, min_periods=1).max()
        low_roll = prices.rolling(window=dist_period, min_periods=1).min()
    
    df["dist_from_high_30m"] = (prices - high_roll) / (prices + 1e-8)
    feature_list.append("dist_from_high_30m")
    
    df["dist_from_low_30m"] = (prices - low_roll) / (prices + 1e-8)
    feature_list.append("dist_from_low_30m")
    
    # 15. Time-based features (intraday patterns)
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
        # Use datetime index - cast to DatetimeIndex for type checker
        dt_index = pd.DatetimeIndex(df.index)
        # Minute of day (0-1439)
        df["minute_of_day"] = dt_index.hour * 60 + dt_index.minute
        feature_list.append("minute_of_day")
        
        # Day of week (0=Monday, 6=Sunday)
        df["day_of_week"] = dt_index.dayofweek
        feature_list.append("day_of_week")
        
        # Hour of day
        df["hour_of_day"] = dt_index.hour
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

