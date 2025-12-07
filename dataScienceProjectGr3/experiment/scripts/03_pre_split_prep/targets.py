"""
Target computation for GRXEUR trend prediction.

This module computes forward-looking normalized trend direction targets.
For each prediction period t, we:
1. Compute linear regression slope of future price window of length t
2. Normalize by the mean price in that window
3. Use the sign as the target (upward vs downward/flat trend)
"""

import pandas as pd
import numpy as np
from typing import List


def compute_normalized_slope(prices: pd.Series, period: int) -> pd.Series:
    """
    Compute normalized linear regression slope for forward-looking price window.
    
    For each row i, we look at prices[i+1:i+1+period] and compute:
    - Linear regression slope
    - Mean price in that window
    - Normalized slope = slope / mean_price
    
    Uses vectorized operations for better performance.
    
    Args:
        prices: Series of close prices
        period: Number of periods to look ahead
        
    Returns:
        Series of normalized slopes (NaN for rows without enough future data)
    """
    n = len(prices)
    slopes = np.full(n, np.nan, dtype=float)
    
    # Pre-compute x for linear regression (0 to period-1)
    x = np.arange(period, dtype=float)
    x_mean = np.mean(x)
    x_centered = x - x_mean
    x_var = np.var(x)
    
    if x_var == 0:
        return pd.Series(slopes, index=prices.index)
    
    # Vectorized computation using rolling windows
    prices_array = prices.values
    
    for i in range(n - period):
        # Get future window starting from i+1
        y = prices_array[i+1:i+1+period]
        
        if len(y) < period:
            continue
        
        # Compute linear regression slope: slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        # Slope calculation
        slope = np.sum(x_centered * y_centered) / (x_var * (period - 1))
        
        # Normalize by mean price
        normalized_slope = slope / y_mean if y_mean > 0 else 0
        slopes[i] = normalized_slope
    
    return pd.Series(slopes, index=prices.index)


def add_normalized_trend_direction(
    df: pd.DataFrame,
    prediction_periods: List[int],
    price_col: str = "close"
) -> pd.DataFrame:
    """
    Add normalized trend direction targets for multiple prediction periods.
    
    For each prediction period, creates a target column:
    - 'target_trend_{period}m': normalized slope of future price window
    - 'target_direction_{period}m': sign of normalized slope (-1, 0, or 1)
    
    Args:
        df: DataFrame with timestamp index and price column
        prediction_periods: List of prediction periods in minutes (e.g., [5, 10, 15, 30, 60])
        price_col: Name of the price column to use
        
    Returns:
        DataFrame with added target columns
    """
    df = df.copy()
    prices = df[price_col]
    
    for period in prediction_periods:
        # Compute normalized slope
        normalized_slopes = compute_normalized_slope(prices, period)
        
        # Add normalized trend target
        target_col = f"target_trend_{period}m"
        df[target_col] = normalized_slopes
        
        # Add direction target (sign of normalized slope)
        # Use a small threshold to avoid noise
        threshold = 1e-8
        direction_col = f"target_direction_{period}m"
        df[direction_col] = np.where(
            df[target_col] > threshold, 1,
            np.where(df[target_col] < -threshold, -1, 0)
        )
        
        print(f"  Added targets for {period}m prediction period")
    
    return df

