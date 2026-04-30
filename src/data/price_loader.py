import yfinance as yf
import pandas as pd
import logging
from pathlib import Path

# Assuming point_in_time is in the same directory
from .point_in_time import PointInTimeContext

logger = logging.getLogger(__name__)

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception

class PriceLoader:
    """
    Handles fetching and caching of price data, strictly enforcing
    the PointInTimeContext so that no future data leaks into the system.
    """
    
    def __init__(self, cache_dir: str | Path | None = None):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_daily_prices(self, ticker: str, pit_context: PointInTimeContext, start_date: str = "2010-01-01") -> pd.DataFrame:
        """
        Fetches historical daily prices for a ticker and truncates any
        data that occurs after the pit_context analysis_date.
        """
        logger.info(f"Fetching price data for {ticker} as of {pit_context.analysis_date}")
        
        cache_file = self.cache_dir / f"{ticker}_daily_prices.parquet"
        
        # Prefer the checked-in cache when available so the app still works under
        # Streamlit Cloud rate limits and only falls back to Yahoo when necessary.
        if cache_file.exists():
            logger.info(f"Using cached price data for {ticker} from {cache_file}")
            df = pd.read_parquet(cache_file)
        else:
            ticker_obj = yf.Ticker(ticker)
            try:
                df = ticker_obj.history(start=start_date, end=None)  # fetch up to today
            except YFRateLimitError:
                logger.warning(f"Yahoo Finance rate-limited price history for {ticker}.")
                return pd.DataFrame()
            except Exception as exc:
                logger.error(f"Failed to fetch price history for {ticker}: {exc}")
                return pd.DataFrame()

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
