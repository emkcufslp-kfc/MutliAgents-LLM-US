import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class MacroLoader:
    """
    Lightweight integration of the Tsvetoslav Tsachev Macro Regime concepts.
    Fetches core US macro indicators and calculates a Top-Down Regime Score.
    """
    def __init__(self):
        self.api_key = os.environ.get("FRED_API_KEY")
        self.fred = None
        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
            except ImportError:
                logger.warning("fredapi not installed. Run: pip install fredapi")
        else:
            logger.warning("FRED_API_KEY not found in .env")
            
    def get_macro_regime(self):
        """Fetches CPI and Unemployment to establish the market environment."""
        if not self.fred:
            return {"regime": "SIMULATED EXPANSION (No API Key)", "cpi": 3.2, "unemployment": 4.1}
            
        try:
            # Fetch data (simplified for dashboard)
            unrate_series = self.fred.get_series('UNRATE')
            cpi_series = self.fred.get_series('CPIAUCSL')
            
            # Simple YoY CPI calculation
            current_cpi = cpi_series.iloc[-1]
            last_year_cpi = cpi_series.iloc[-13]
            cpi_yoy = ((current_cpi / last_year_cpi) - 1) * 100
            
            unrate = unrate_series.iloc[-1]
            
            regime = "Steady Expansion"
            if cpi_yoy > 4.0 and unrate > 4.5:
                regime = "Stagflation Risk"
            elif cpi_yoy > 3.0:
                regime = "Inflationary Growth"
            elif unrate > 5.0:
                regime = "Recessionary Risk"
                
            return {
                "regime": regime,
                "cpi": round(cpi_yoy, 2),
                "unemployment": round(unrate, 1)
            }
        except Exception as e:
            logger.error(f"FRED API Error: {e}")
            return {"regime": "ERROR FETCHING DATA", "cpi": 0.0, "unemployment": 0.0}
