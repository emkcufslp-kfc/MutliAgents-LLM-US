import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from . import alpha_vantage_loader

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

    def _cache_file(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_daily_prices.parquet"

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        prepared = df.copy()
        prepared = prepared.reset_index()

        if "Date" not in prepared.columns:
            prepared = prepared.rename(columns={prepared.columns[0]: "Date"})

        prepared["Date"] = pd.to_datetime(prepared["Date"]).dt.tz_localize(None)
        for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if column in prepared.columns:
                prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

        prepared = prepared.dropna(subset=["Date", "Close"])
        if "Volume" not in prepared.columns:
            prepared["Volume"] = 0.0
        prepared = prepared.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        return prepared

    def _filter_for_context(self, df: pd.DataFrame, pit_context: PointInTimeContext) -> pd.DataFrame:
        safe_df = pit_context.filter_dataframe(df.copy(), "Date")
        safe_df.attrs["source"] = df.attrs.get("source", "unknown")
        return safe_df

    def _load_cached_prices(self, ticker: str, pit_context: PointInTimeContext) -> pd.DataFrame:
        cache_file = self._cache_file(ticker)
        if not cache_file.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(cache_file)
        except Exception as exc:
            logger.warning(f"Failed to read cached price data for {ticker}: {exc}")
            return pd.DataFrame()

        prepared = self._prepare_dataframe(df)
        prepared.attrs["source"] = "cache"
        return self._filter_for_context(prepared, pit_context)

    def _save_cached_prices(self, ticker: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._cache_file(ticker), index=False)
        except Exception as exc:
            logger.warning(f"Failed to write cached price data for {ticker}: {exc}")

    def _download_single_ticker(self, ticker: str, start_date: str) -> pd.DataFrame:
        df = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return self._prepare_dataframe(df)

    def fetch_daily_prices(self, ticker: str, pit_context: PointInTimeContext, start_date: str = "2010-01-01") -> pd.DataFrame:
        """
        Fetches historical daily prices for a ticker and truncates any
        data that occurs after the pit_context analysis_date.
        """
        logger.info(f"Fetching price data for {ticker} as of {pit_context.analysis_date}")

        cached_df = self._load_cached_prices(ticker, pit_context)
        if not cached_df.empty:
            logger.info(f"Using cached price data for {ticker}")
            return cached_df

        try:
            df = self._download_single_ticker(ticker, start_date)
            if not df.empty:
                df.attrs["source"] = "yfinance"
                self._save_cached_prices(ticker, df)
                return self._filter_for_context(df, pit_context)
        except YFRateLimitError:
            logger.warning(f"Yahoo Finance rate-limited price history for {ticker}.")
        except Exception as exc:
            logger.error(f"Failed to fetch price history for {ticker}: {exc}")

        if alpha_vantage_loader.has_api_key():
            try:
                df = alpha_vantage_loader.fetch_daily_prices(ticker, pit_context, start_date)
                if not df.empty:
                    prepared = self._prepare_dataframe(df)
                    prepared.attrs["source"] = "alpha_vantage"
                    self._save_cached_prices(ticker, prepared)
                    return self._filter_for_context(prepared, pit_context)
            except Exception as exc:
                logger.error(f"Alpha Vantage fallback failed for {ticker}: {exc}")

        return pd.DataFrame()

    def fetch_daily_prices_batch(
        self,
        tickers: list[str],
        pit_context: PointInTimeContext,
        start_date: str = "2010-01-01",
        chunk_size: int = 50,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for ticker in dict.fromkeys(tickers):
            cached_df = self._load_cached_prices(ticker, pit_context)
            if not cached_df.empty:
                results[ticker] = cached_df
            else:
                missing.append(ticker)

        for chunk_start in range(0, len(missing), chunk_size):
            chunk = missing[chunk_start:chunk_start + chunk_size]
            if not chunk:
                continue

            try:
                batch_df = yf.download(
                    chunk,
                    start=start_date,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by="ticker",
                )
            except YFRateLimitError:
                logger.warning(f"Yahoo Finance rate-limited batch price download for chunk: {chunk}")
                batch_df = pd.DataFrame()
            except Exception as exc:
                logger.error(f"Failed batch price download for {chunk}: {exc}")
                batch_df = pd.DataFrame()

            for ticker in chunk:
                ticker_df = pd.DataFrame()

                if not batch_df.empty:
                    try:
                        if isinstance(batch_df.columns, pd.MultiIndex):
                            if ticker in batch_df.columns.get_level_values(0):
                                ticker_df = batch_df[ticker]
                        else:
                            ticker_df = batch_df
                    except Exception as exc:
                        logger.warning(f"Failed to extract {ticker} from batch price data: {exc}")

                if ticker_df is not None and not ticker_df.empty:
                    prepared = self._prepare_dataframe(ticker_df)
                    prepared.attrs["source"] = "yfinance"
                    self._save_cached_prices(ticker, prepared)
                    results[ticker] = self._filter_for_context(prepared, pit_context)
                    continue

                results[ticker] = self.fetch_daily_prices(ticker, pit_context, start_date)

        for ticker in tickers:
            results.setdefault(ticker, pd.DataFrame())

        return results
