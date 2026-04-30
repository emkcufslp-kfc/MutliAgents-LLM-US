import yfinance as yf
import pandas as pd
import logging
from typing import Optional
from pathlib import Path

# Assuming point_in_time is in the same directory
from .point_in_time import PointInTimeContext

logger = logging.getLogger(__name__)

class PriceLoader:
    """
    Handles fetching and caching of price data, strictly enforcing
    the PointInTimeContext so that no future data leaks into the system.
    """
    
    def __init__(self, cache_dir: str = "../data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_daily_prices(self, ticker: str, pit_context: PointInTimeContext, start_date: str = "2010-01-01") -> pd.DataFrame:
        """
        Fetches historical daily prices for a ticker and truncates any
        data that occurs after the pit_context analysis_date.
        """
        logger.info(f"Fetching price data for {ticker} as of {pit_context.analysis_date}")
        
        cache_file = self.cache_dir / f"{ticker}_daily_prices.parquet"
        
        # In a production system, you'd check cache freshness. For now, we fetch fresh if missing.
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(start=start_date, end=None)  # fetch up to today
            if not df.empty:
                df.reset_index(inplace=True)
                # Convert tz-aware datetime to tz-naive date for safe comparison
                df['Date'] = pd.to_datetime(df['Date']).dt.date
                df.to_parquet(cache_file)
            else:
                logger.warning(f"No price data found for {ticker}.")
                return pd.DataFrame()
                
        # CRITICAL STEP: Prevent Lookahead Bias
        df['Date'] = pd.to_datetime(df['Date'])
        safe_df = pit_context.filter_dataframe(df, 'Date')
        
        return safe_df
