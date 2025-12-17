# src/feature_engineer.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FeatureEngineer:
    """Creates ML-ready datasets from historical indicators and price data."""
    
    @staticmethod
    def create_target_column(price_series, indicator_series, horizon=1, method='direction'):
        """
        Creates a target variable for supervised learning.
        
        Args:
            price_series: Series of historical prices.
            indicator_series: DataFrame of historical indicators (aligned with price_series).
            horizon: Number of periods ahead to predict.
            method: 'direction' (1 if price up, 0 if down) or 'return' (continuous return).
            
        Returns:
            DataFrame with the target column added.
        """
        df = indicator_series.copy()
        
        if method == 'direction':
            # Binary classification: 1 if price goes up in next 'horizon' periods
            future_return = price_series.shift(-horizon) / price_series - 1
            df['target_direction'] = (future_return > 0).astype(int)
            
        elif method == 'return':
            # Regression: predict the future return
            future_return = price_series.shift(-horizon) / price_series - 1
            df['target_return'] = future_return
            
        # Remove the last 'horizon' rows which have NaN target
        df = df.iloc[:-horizon] if horizon > 0 else df
        
        return df
    
    @staticmethod
    def add_technical_features(df, price_series):
        """Adds common technical features to the dataset."""
        # Rolling means (momentum)
        df['price_ma_5'] = price_series.rolling(5).mean()
        df['price_ma_20'] = price_series.rolling(20).mean()
        
        # Volatility
        df['price_volatility_10'] = price_series.pct_change().rolling(10).std()
        
        # Price position relative to range
        rolling_max = price_series.rolling(20).max()
        rolling_min = price_series.rolling(20).min()
        df['price_position'] = (price_series - rolling_min) / (rolling_max - rolling_min + 1e-10)
        
        return df
