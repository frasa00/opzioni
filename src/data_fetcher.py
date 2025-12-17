"""
Data fetching module - Aggiornato per nuovi indicatori
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import requests
import json
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataFetcher:
    """Fetches and processes market data"""
    
    def __init__(self, cache_duration: int = 60):
        self.cache_duration = cache_duration
        self.cache = {}
        
        # CBOE API endpoints (alternative source)
        self.cboe_urls = {
            'vix': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv',
            'vxn': 'https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv'
        }
    
    def get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get comprehensive market data"""
        
        cache_key = f"market_{symbol}_{datetime.now().strftime('%Y%m%d')}"
        
        if cache_key in self.cache:
            if time.time() - self.cache[cache_key]['timestamp'] < self.cache_duration:
                return self.cache[cache_key]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get spot price
            hist = ticker.history(period='2d')
            if len(hist) < 2:
                return None
            
            spot_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2]
            price_change_pct = ((spot_price - prev_price) / prev_price) * 100
            
            # Get VIX/VXN
            vix_symbol = '^VIX' if symbol == '^SPX' else '^VXN'
            vix_ticker = yf.Ticker(vix_symbol)
            vix_hist = vix_ticker.history(period='2d')
            
            if len(vix_hist) >= 2:
                vix = vix_hist['Close'].iloc[-1]
                prev_vix = vix_hist['Close'].iloc[-2]
                vix_change_pct = ((vix - prev_vix) / prev_vix) * 100
            else:
                vix, vix_change_pct = 0, 0
            
            # Get volume and moving averages
            volume = hist['Volume'].iloc[-1]
            avg_volume = hist['Volume'].mean()
            
            # Calculate RSI (14-day)
            hist_14d = ticker.history(period='15d')
            if len(hist_14d) >= 14:
                delta = hist_14d['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
            else:
                rsi = 50
            
            data = {
                'symbol': symbol,
                'spot_price': spot_price,
                'price_change_pct': price_change_pct,
                'volume': volume,
                'volume_ratio': volume / avg_volume if avg_volume > 0 else 1,
                'vix': vix,
                'vix_change_pct': vix_change_pct,
                'rsi': rsi,
                'timestamp': datetime.now(),
                'support': spot_price * 0.98,  # Placeholder
                'resistance': spot_price * 1.02  # Placeholder
            }
            
            # Cache the data
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': data
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None
    
    def get_option_data(self, symbol: str, expiration: str = None) -> Optional[Dict]:
        """Get option chain data for multiple expirations"""
        
        cache_key = f"options_{symbol}_{expiration if expiration else 'all'}"
        
        if cache_key in self.cache:
            if time.time() - self.cache[cache_key]['timestamp'] < self.cache_duration:
                return self.cache[cache_key]['data']
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expirations
            expirations = ticker.options
            if not expirations:
                return None
            
            # Use nearest expiration if not specified
            if not expiration:
                expiration = expirations[0]
            
            # Get option chain
            chain = ticker.option_chain(expiration)
            
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            
            # Add additional calculations
            spot_price = self.get_market_data(symbol)['spot_price']
            
            # Calculate moneyness
            calls['moneyness'] = (calls['strike'] - spot_price) / spot_price
            puts['moneyness'] = (puts['strike'] - spot_price) / spot_price
            
            # Calculate days to expiry
            expiry_date = datetime.strptime(expiration, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            calls['days_to_expiry'] = days_to_expiry
            puts['days_to_expiry'] = days_to_expiry
            
            # Filter near-the-money options
            calls = calls[(calls['moneyness'].abs() <= 0.3)]
            puts = puts[(puts['moneyness'].abs() <= 0.3)]
            
            data = {
                'calls': calls,
                'puts': puts,
                'expiration': expiration,
                'days_to_expiry': days_to_expiry,
                'spot_price': spot_price,
                'all_expirations': expirations[:5]  # First 5 expirations
            }
            
            # Cache the data
            self.cache[cache_key] = {
                'timestamp': time.time(),
                'data': data
            }
            
            return data
            
        except Exception as e:
            print(f"Error fetching option data: {e}")
            return None
    
    def get_historical_skew(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get historical skew data"""
        
        # This would require storing historical data in a database
        # For now, return simulated data
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        skew_data = pd.DataFrame({
            'date': dates,
            'skew_25d': np.random.uniform(-0.1, 0.15, days),
            'pcr': np.random.uniform(0.5, 1.5, days),
            'vix': np.random.uniform(15, 30, days)
        })
        
        # Add some autocorrelation to make it realistic
        skew_data['skew_25d'] = skew_data['skew_25d'].rolling(window=3).mean().fillna(method='bfill')
        
        return skew_data
    
    def get_cboe_data(self, index: str = 'vix') -> Optional[pd.DataFrame]:
        """Get data from CBOE directly (alternative to Yahoo)"""
        
        try:
            url = self.cboe_urls.get(index)
            if not url:
                return None
            
            response = requests.get(url)
            if response.status_code == 200:
                data = pd.read_csv(pd.compat.StringIO(response.text))
                return data
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching CBOE data: {e}")
            return None
