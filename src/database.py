"""
Database module for storing historical data
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional

class OptionsDatabase:
    """SQLite database for storing options data"""
    
    def __init__(self, db_path: str = "data/historical/options.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp DATETIME,
            spot_price REAL,
            vix REAL,
            vix_change_pct REAL,
            price_change_pct REAL,
            volume REAL,
            rsi REAL,
            support REAL,
            resistance REAL
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS indicators (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            timestamp DATETIME,
            skew_25d REAL,
            skew_10d REAL,
            pcr_weighted REAL,
            pcr_simple REAL,
            total_gamma_exposure REAL,
            max_pain_strike REAL,
            vanna_total REAL,
            charm_total REAL,
            systemic_risk_score REAL,
            risk_level TEXT,
            data_json TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS option_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            expiration DATE,
            timestamp DATETIME,
            call_data_json TEXT,
            put_data_json TEXT,
            spot_price REAL
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_market_symbol_time 
        ON market_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_indicators_symbol_time 
        ON indicators(symbol, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def save_market_data(self, symbol: str, data: Dict):
        """Save market data to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO market_data 
        (symbol, timestamp, spot_price, vix, vix_change_pct, 
         price_change_pct, volume, rsi, support, resistance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            data.get('timestamp', datetime.now()),
            data.get('spot_price', 0),
            data.get('vix', 0),
            data.get('vix_change_pct', 0),
            data.get('price_change_pct', 0),
            data.get('volume', 0),
            data.get('rsi', 50),
            data.get('support', 0),
            data.get('resistance', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def save_indicators(self, symbol: str, indicators: Dict):
        """Save calculated indicators"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract systemic risk data
        systemic_risk = indicators.get('systemic_risk', {})
        
        cursor.execute('''
        INSERT INTO indicators 
        (symbol, timestamp, skew_25d, skew_10d, pcr_weighted, pcr_simple,
         total_gamma_exposure, max_pain_strike, vanna_total, charm_total,
         systemic_risk_score, risk_level, data_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            datetime.now(),
            indicators.get('skew_25d', 0),
            indicators.get('skew_10d', 0),
            indicators.get('pcr_weighted', 1),
            indicators.get('pcr_simple', 1),
            indicators.get('total_gamma_exposure', 0),
            indicators.get('max_pain_strike', 0),
            indicators.get('total_vanna', 0),
            indicators.get('total_charm', 0),
            systemic_risk.get('total_score', 0),
            systemic_risk.get('risk_level', 'LOW'),
            json.dumps(indicators, default=str)
        ))
        
        conn.commit()
        conn.close()
    
    def get_historical_indicators(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical indicators for specified days"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT * FROM indicators 
        WHERE symbol = ? 
        AND timestamp >= datetime('now', '-{days} days')
        ORDER BY timestamp ASC
        """
        
        df = pd.read_sql_query(query, conn, params=(symbol,))
        conn.close()
        
        return df
    
    def get_daily_summary(self, symbol: str, date: datetime = None) -> Dict:
        """Get daily summary for a specific date"""
        
        if not date:
            date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get data for the date
        date_str = date.strftime('%Y-%m-%d')
        
        cursor.execute('''
        SELECT * FROM indicators 
        WHERE symbol = ? 
        AND date(timestamp) = ?
        ORDER BY timestamp DESC
        LIMIT 1
        ''', (symbol, date_str))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            columns = ['id', 'symbol', 'timestamp', 'skew_25d', 'skew_10d', 
                      'pcr_weighted', 'pcr_simple', 'total_gamma_exposure',
                      'max_pain_strike', 'vanna_total', 'charm_total',
                      'systemic_risk_score', 'risk_level', 'data_json']
            
            return dict(zip(columns, row))
        
        return {}
    
    def cleanup_old_data(self, days_to_keep: int = 365):
        """Cleanup data older than specified days"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
        DELETE FROM market_data 
        WHERE timestamp < datetime('now', '-{days_to_keep} days')
        ''')
        
        cursor.execute(f'''
        DELETE FROM indicators 
        WHERE timestamp < datetime('now', '-{days_to_keep} days')
        ''')
        
        cursor.execute(f'''
        DELETE FROM option_chains 
        WHERE timestamp < datetime('now', '-{days_to_keep} days')
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"Cleaned up data older than {days_to_keep} days")
