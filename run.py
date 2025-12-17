#!/usr/bin/env python3
"""
Main entry point for Options Terminal
"""

import sys
import os
from src.dashboard import EnhancedOptionsDashboard

def main():
    """Main function"""
    
    print("Starting Advanced Options Terminal...")
    print("=" * 50)
    
    # Check for required modules
    try:
        import streamlit
        import yfinance
        import pandas
    except ImportError as e:
        print(f"Missing required module: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data/historical", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)
    
    # Initialize and run dashboard
    dashboard = EnhancedOptionsDashboard()
    dashboard.create_main_dashboard()

if __name__ == "__main__":
    main()
