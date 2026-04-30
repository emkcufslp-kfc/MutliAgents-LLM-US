import yfinance as yf
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
from datetime import timedelta

from .point_in_time import PointInTimeContext

logger = logging.getLogger(__name__)

class FundamentalLoader:
    """
    Handles fetching fundamental data (ROE, FCF, Margins).
    Enforces a strict lag (e.g. 45 days) to simulate SEC filing delays
    and prevent lookahead bias when exact filing dates are unknown.
    """
    
    def __init__(self, fallback_lag_days: int = 45, cache_dir: str = "../data/cache"):
        self.fallback_lag_days = fallback_lag_days
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_fundamentals(self, ticker: str, pit_context: PointInTimeContext) -> Dict[str, Any]:
        """
        Fetches fundamental data and filters it based on the analysis_date.
        Because yfinance overrides historical fundamentals, this serves as a
        placeholder for a true point-in-time database (like Compustat/FMP).
        """
        logger.info(f"Fetching fundamental data for {ticker} as of {pit_context.analysis_date}")
        
        ticker_obj = yf.Ticker(ticker)
        
        # We try to get the income statement and balance sheet
        try:
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            cash_flow = ticker_obj.cashflow
            
            if income_stmt.empty or cash_flow.empty:
                return {"error": "Missing fundamental data"}
                
            # Transpose to get dates as rows
            inc_df = income_stmt.T.reset_index().rename(columns={"index": "ReportDate"})
            cf_df = cash_flow.T.reset_index().rename(columns={"index": "ReportDate"})
            
            # Enforce the 45-day SEC lag rule for availability
            inc_df['AvailableDate'] = pd.to_datetime(inc_df['ReportDate']) + timedelta(days=self.fallback_lag_days)
            cf_df['AvailableDate'] = pd.to_datetime(cf_df['ReportDate']) + timedelta(days=self.fallback_lag_days)
            
            # Filter based on Point In Time Context
            safe_inc = pit_context.filter_dataframe(inc_df, 'AvailableDate')
            safe_cf = pit_context.filter_dataframe(cf_df, 'AvailableDate')
            
            if safe_inc.empty or safe_cf.empty:
                logger.warning(f"No fundamental data available for {ticker} prior to {pit_context.analysis_date} (accounting for {self.fallback_lag_days} day SEC lag).")
                return {"error": "Data strictly filtered due to Point-In-Time limits."}
                
            # Get the most recent valid row
            latest_inc = safe_inc.sort_values("AvailableDate", ascending=False).iloc[0]
            latest_cf = safe_cf.sort_values("AvailableDate", ascending=False).iloc[0]
            
            # Extract key metrics needed for the Hedge Fund Screener
            net_income = latest_inc.get("Net Income", 0)
            free_cash_flow = latest_cf.get("Free Cash Flow", 0)
            operating_cf = latest_cf.get("Operating Cash Flow", 0)
            
            return {
                "ticker": ticker,
                "as_of_date": str(pit_context.analysis_date),
                "most_recent_report_date": str(latest_inc['ReportDate'].date()),
                "net_income": float(net_income) if not pd.isna(net_income) else 0.0,
                "free_cash_flow": float(free_cash_flow) if not pd.isna(free_cash_flow) else 0.0,
                "operating_cash_flow": float(operating_cf) if not pd.isna(operating_cf) else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            return {"error": str(e)}
