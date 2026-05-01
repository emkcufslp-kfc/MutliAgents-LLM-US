import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf

from . import alpha_vantage_loader
from .point_in_time import PointInTimeContext

logger = logging.getLogger(__name__)

try:
    from yfinance.exceptions import YFRateLimitError
except ImportError:
    YFRateLimitError = Exception


class FundamentalLoader:
    """
    Handles fetching fundamental data (ROE, FCF, Margins).
    Enforces a strict lag (e.g. 45 days) to simulate SEC filing delays
    and prevent lookahead bias when exact filing dates are unknown.
    """

    def __init__(self, fallback_lag_days: int = 45, cache_dir: str | Path | None = None):
        self.fallback_lag_days = fallback_lag_days
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_fundamentals.json"

    def _save_cached_reports(self, ticker: str, income_df: pd.DataFrame, cash_flow_df: pd.DataFrame) -> None:
        payload = {
            "income_reports": income_df.to_dict(orient="records"),
            "cash_flow_reports": cash_flow_df.to_dict(orient="records"),
        }
        try:
            self._cache_file(ticker).write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to cache fundamentals for {ticker}: {exc}")

    def _load_cached_reports(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        cache_file = self._cache_file(ticker)
        if not cache_file.exists():
            return pd.DataFrame(), pd.DataFrame()

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read cached fundamentals for {ticker}: {exc}")
            return pd.DataFrame(), pd.DataFrame()

        inc_df = pd.DataFrame(payload.get("income_reports", []))
        cf_df = pd.DataFrame(payload.get("cash_flow_reports", []))
        if not inc_df.empty:
            inc_df["ReportDate"] = pd.to_datetime(inc_df["ReportDate"])
        if not cf_df.empty:
            cf_df["ReportDate"] = pd.to_datetime(cf_df["ReportDate"])
        return inc_df, cf_df

    def _prepare_statement_frames(
        self,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        inc_df = income_stmt.T.reset_index().rename(columns={"index": "ReportDate"})
        cf_df = cash_flow.T.reset_index().rename(columns={"index": "ReportDate"})
        inc_df["ReportDate"] = pd.to_datetime(inc_df["ReportDate"])
        cf_df["ReportDate"] = pd.to_datetime(cf_df["ReportDate"])
        return inc_df, cf_df

    def _build_snapshot(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        inc_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        source: str,
    ) -> Dict[str, Any]:
        if inc_df.empty or cf_df.empty:
            return {"error": "Missing fundamental data"}

        safe_inc = inc_df.copy()
        safe_cf = cf_df.copy()
        safe_inc["AvailableDate"] = pd.to_datetime(safe_inc["ReportDate"]) + timedelta(days=self.fallback_lag_days)
        safe_cf["AvailableDate"] = pd.to_datetime(safe_cf["ReportDate"]) + timedelta(days=self.fallback_lag_days)

        safe_inc = pit_context.filter_dataframe(safe_inc, "AvailableDate")
        safe_cf = pit_context.filter_dataframe(safe_cf, "AvailableDate")

        if safe_inc.empty or safe_cf.empty:
            logger.warning(
                f"No fundamental data available for {ticker} prior to {pit_context.analysis_date} "
                f"(accounting for {self.fallback_lag_days} day SEC lag)."
            )
            return {"error": "Data strictly filtered due to Point-In-Time limits."}

        latest_inc = safe_inc.sort_values("AvailableDate", ascending=False).iloc[0]
        latest_cf = safe_cf.sort_values("AvailableDate", ascending=False).iloc[0]

        net_income = latest_inc.get("Net Income", 0)
        free_cash_flow = latest_cf.get("Free Cash Flow", 0)
        operating_cf = latest_cf.get("Operating Cash Flow", 0)

        return {
            "ticker": ticker,
            "as_of_date": str(pit_context.analysis_date),
            "most_recent_report_date": str(pd.to_datetime(latest_inc["ReportDate"]).date()),
            "net_income": float(net_income) if not pd.isna(net_income) else 0.0,
            "free_cash_flow": float(free_cash_flow) if not pd.isna(free_cash_flow) else 0.0,
            "operating_cash_flow": float(operating_cf) if not pd.isna(operating_cf) else 0.0,
            "source": source,
        }

    def fetch_fundamentals(self, ticker: str, pit_context: PointInTimeContext) -> Dict[str, Any]:
        """
        Fetches fundamental data and filters it based on the analysis_date.
        Because yfinance overrides historical fundamentals, this serves as a
        placeholder for a true point-in-time database (like Compustat/FMP).
        """
        logger.info(f"Fetching fundamental data for {ticker} as of {pit_context.analysis_date}")

        cached_inc, cached_cf = self._load_cached_reports(ticker)
        if not cached_inc.empty and not cached_cf.empty:
            snapshot = self._build_snapshot(ticker, pit_context, cached_inc, cached_cf, "cache")
            if "error" not in snapshot:
                return snapshot

        ticker_obj = yf.Ticker(ticker)

        try:
            income_stmt = ticker_obj.income_stmt
            cash_flow = ticker_obj.cashflow
        except YFRateLimitError:
            logger.warning(f"Yahoo Finance rate-limited fundamentals for {ticker}.")
            if not cached_inc.empty and not cached_cf.empty:
                snapshot = self._build_snapshot(ticker, pit_context, cached_inc, cached_cf, "cache")
                if "error" not in snapshot:
                    return snapshot
            if alpha_vantage_loader.has_api_key():
                try:
                    fundamentals = alpha_vantage_loader.fetch_fundamentals(ticker, pit_context, self.fallback_lag_days)
                    fundamentals["source"] = "alpha_vantage"
                    return fundamentals
                except Exception as exc:
                    logger.error(f"Alpha Vantage fundamentals fallback failed for {ticker}: {exc}")
            return {"error": "Yahoo Finance rate limit exceeded"}
        except Exception as exc:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {exc}")
            if not cached_inc.empty and not cached_cf.empty:
                snapshot = self._build_snapshot(ticker, pit_context, cached_inc, cached_cf, "cache")
                if "error" not in snapshot:
                    return snapshot
            if alpha_vantage_loader.has_api_key():
                try:
                    fundamentals = alpha_vantage_loader.fetch_fundamentals(ticker, pit_context, self.fallback_lag_days)
                    fundamentals["source"] = "alpha_vantage"
                    return fundamentals
                except Exception as av_exc:
                    logger.error(f"Alpha Vantage fundamentals fallback failed for {ticker}: {av_exc}")
            return {"error": str(exc)}

        if income_stmt.empty or cash_flow.empty:
            return {"error": "Missing fundamental data"}

        inc_df, cf_df = self._prepare_statement_frames(income_stmt, cash_flow)
        self._save_cached_reports(ticker, inc_df, cf_df)
        return self._build_snapshot(ticker, pit_context, inc_df, cf_df, "yfinance")
