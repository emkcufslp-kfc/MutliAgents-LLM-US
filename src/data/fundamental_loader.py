import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yfinance as yf

from . import alpha_vantage_loader
from . import sec_edgar_loader
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

    CACHE_SCHEMA_VERSION = 2

    def __init__(
        self,
        fallback_lag_days: int = 45,
        cache_dir: str | Path | None = None,
        prefer_sec_primary: bool = True,
        yf_cooldown_seconds: int = 300,
    ):
        self.fallback_lag_days = fallback_lag_days
        self.prefer_sec_primary = prefer_sec_primary
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.yf_cooldown_seconds = max(int(yf_cooldown_seconds), 0)
        self._yf_cooldown_until: datetime | None = None

    def _cache_file(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}_fundamentals.json"

    def _needs_enrichment(self, snapshot: Dict[str, Any]) -> bool:
        return all(
            snapshot.get(key) is None
            for key in [
                "eps_ttm_yoy_pct",
                "revenue_ttm_yoy_pct",
                "roe_ttm_pct",
                "roe_stability_std_pct",
                "market_cap_b",
            ]
        )

    def _yfinance_available(self) -> bool:
        return self._yf_cooldown_until is None or datetime.now(UTC) >= self._yf_cooldown_until

    def _mark_yfinance_rate_limited(self, ticker: str) -> None:
        if self.yf_cooldown_seconds <= 0:
            return
        self._yf_cooldown_until = datetime.now(UTC) + timedelta(seconds=self.yf_cooldown_seconds)
        logger.warning(
            "Yahoo Finance rate-limited fundamentals for %s. "
            "Skipping further Yahoo fundamental requests until %s UTC.",
            ticker,
            self._yf_cooldown_until.replace(microsecond=0).isoformat(),
        )

    def _try_sec_fallback(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        current_price: float | None = None,
    ) -> Dict[str, Any]:
        try:
            return sec_edgar_loader.fetch_fundamentals(ticker, pit_context, current_price=current_price)
        except Exception as exc:
            logger.error(f"SEC companyfacts fallback failed for {ticker}: {exc}")
            return {"error": str(exc)}

    def _try_alpha_vantage_fallback(self, ticker: str, pit_context: PointInTimeContext) -> Dict[str, Any]:
        if not alpha_vantage_loader.has_api_key():
            return {"error": "ALPHA_VANTAGE_API_KEY is not configured"}
        try:
            fundamentals = alpha_vantage_loader.fetch_fundamentals(ticker, pit_context, self.fallback_lag_days)
            fundamentals["source"] = "alpha_vantage"
            return fundamentals
        except Exception as exc:
            logger.error(f"Alpha Vantage fundamentals fallback failed for {ticker}: {exc}")
            return {"error": str(exc)}

    def _merge_snapshots(self, primary: Dict[str, Any], secondary: Dict[str, Any], merged_source: str) -> Dict[str, Any]:
        merged = dict(primary)
        for key, value in secondary.items():
            if key == "source":
                continue
            if merged.get(key) in (None, "", "unknown", "unavailable") and value not in (None, ""):
                merged[key] = value
        merged["source"] = merged_source
        return merged

    def _save_cached_reports(
        self,
        ticker: str,
        income_df: pd.DataFrame,
        cash_flow_df: pd.DataFrame,
        balance_sheet_df: pd.DataFrame,
    ) -> None:
        payload = {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "income_reports": income_df.to_dict(orient="records"),
            "cash_flow_reports": cash_flow_df.to_dict(orient="records"),
            "balance_sheet_reports": balance_sheet_df.to_dict(orient="records"),
        }
        try:
            self._cache_file(ticker).write_text(json.dumps(payload, default=str), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to cache fundamentals for {ticker}: {exc}")

    def _load_cached_reports(self, ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cache_file = self._cache_file(ticker)
        if not cache_file.exists():
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read cached fundamentals for {ticker}: {exc}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        inc_df = pd.DataFrame(payload.get("income_reports", []))
        cf_df = pd.DataFrame(payload.get("cash_flow_reports", []))
        bs_df = pd.DataFrame(payload.get("balance_sheet_reports", []))
        if not inc_df.empty:
            inc_df["ReportDate"] = pd.to_datetime(inc_df["ReportDate"])
        if not cf_df.empty:
            cf_df["ReportDate"] = pd.to_datetime(cf_df["ReportDate"])
        if not bs_df.empty:
            bs_df["ReportDate"] = pd.to_datetime(bs_df["ReportDate"])
        return inc_df, cf_df, bs_df

    def _prepare_statement_frames(
        self,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame,
        balance_sheet: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        inc_df = income_stmt.T.reset_index().rename(columns={"index": "ReportDate"})
        cf_df = cash_flow.T.reset_index().rename(columns={"index": "ReportDate"})
        inc_df["ReportDate"] = pd.to_datetime(inc_df["ReportDate"])
        cf_df["ReportDate"] = pd.to_datetime(cf_df["ReportDate"])
        bs_df = pd.DataFrame()
        if balance_sheet is not None and not balance_sheet.empty:
            bs_df = balance_sheet.T.reset_index().rename(columns={"index": "ReportDate"})
            bs_df["ReportDate"] = pd.to_datetime(bs_df["ReportDate"])
        return inc_df, cf_df, bs_df

    def _first_numeric(self, row: pd.Series, candidates: list[str]) -> float | None:
        for column in candidates:
            if column not in row:
                continue
            value = pd.to_numeric(row.get(column), errors="coerce")
            if not pd.isna(value):
                return float(value)
        return None

    def _compute_ttm_series(self, df: pd.DataFrame, candidates: list[str], output_name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["ReportDate", output_name])

        series = df[["ReportDate"]].copy()
        values = []
        for _, row in df.iterrows():
            values.append(self._first_numeric(row, candidates))
        series[output_name] = values
        series = series.dropna(subset=[output_name]).sort_values("ReportDate")
        if series.empty:
            return pd.DataFrame(columns=["ReportDate", output_name])
        series[output_name] = series[output_name].rolling(window=4).sum()
        series = series.dropna(subset=[output_name]).copy()
        return series

    def _compute_point_in_time_metrics(
        self,
        inc_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        bs_df: pd.DataFrame,
        current_price: float | None,
    ) -> dict[str, float | None]:
        eps_ttm_yoy = None
        revenue_ttm_yoy = None
        roe_ttm = None
        roe_stability_std = None
        market_cap_b = None

        eps_ttm_df = self._compute_ttm_series(inc_df, ["Diluted EPS", "Basic EPS", "Normalized EPS"], "eps_ttm")
        if len(eps_ttm_df) >= 5:
            latest_eps = eps_ttm_df.iloc[-1]["eps_ttm"]
            prior_eps = eps_ttm_df.iloc[-5]["eps_ttm"]
            if prior_eps not in (None, 0):
                eps_ttm_yoy = ((latest_eps / prior_eps) - 1.0) * 100.0

        revenue_ttm_df = self._compute_ttm_series(inc_df, ["Total Revenue", "Operating Revenue", "Revenue"], "revenue_ttm")
        if len(revenue_ttm_df) >= 5:
            latest_rev = revenue_ttm_df.iloc[-1]["revenue_ttm"]
            prior_rev = revenue_ttm_df.iloc[-5]["revenue_ttm"]
            if prior_rev not in (None, 0):
                revenue_ttm_yoy = ((latest_rev / prior_rev) - 1.0) * 100.0

        if not inc_df.empty and not bs_df.empty:
            roe_samples: list[float] = []
            equity_by_date = bs_df.sort_values("ReportDate")
            for _, inc_row in inc_df.sort_values("ReportDate").iterrows():
                report_date = inc_row["ReportDate"]
                safe_equity = equity_by_date[equity_by_date["ReportDate"] <= report_date]
                if safe_equity.empty:
                    continue
                equity_row = safe_equity.iloc[-1]
                quarterly_net_income = self._first_numeric(inc_row, ["Net Income", "Net Income Common Stockholders"])
                equity = self._first_numeric(equity_row, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
                if quarterly_net_income is None or not equity:
                    continue
                annualized_roe = (quarterly_net_income * 4.0 / equity) * 100.0
                roe_samples.append(float(annualized_roe))

            if roe_samples:
                roe_ttm = float(pd.Series(roe_samples).tail(4).mean())
                roe_stability_std = float(pd.Series(roe_samples).tail(4).std(ddof=0)) if len(roe_samples) >= 2 else 0.0

            latest_equity_row = equity_by_date.iloc[-1]
            shares_outstanding = self._first_numeric(
                latest_equity_row,
                ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
            )
            if current_price and shares_outstanding:
                market_cap_b = (current_price * shares_outstanding) / 1_000_000_000.0

        return {
            "eps_ttm_yoy_pct": eps_ttm_yoy,
            "revenue_ttm_yoy_pct": revenue_ttm_yoy,
            "roe_ttm_pct": roe_ttm,
            "roe_stability_std_pct": roe_stability_std,
            "market_cap_b": market_cap_b,
        }

    def _build_snapshot(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        inc_df: pd.DataFrame,
        cf_df: pd.DataFrame,
        bs_df: pd.DataFrame,
        source: str,
        current_price: float | None = None,
    ) -> Dict[str, Any]:
        if inc_df.empty or cf_df.empty:
            return {"error": "Missing fundamental data"}

        safe_inc = inc_df.copy()
        safe_cf = cf_df.copy()
        safe_inc["AvailableDate"] = pd.to_datetime(safe_inc["ReportDate"]) + timedelta(days=self.fallback_lag_days)
        safe_cf["AvailableDate"] = pd.to_datetime(safe_cf["ReportDate"]) + timedelta(days=self.fallback_lag_days)
        safe_bs = bs_df.copy()
        if not safe_bs.empty:
            safe_bs["AvailableDate"] = pd.to_datetime(safe_bs["ReportDate"]) + timedelta(days=self.fallback_lag_days)

        safe_inc = pit_context.filter_dataframe(safe_inc, "AvailableDate")
        safe_cf = pit_context.filter_dataframe(safe_cf, "AvailableDate")
        if not safe_bs.empty:
            safe_bs = pit_context.filter_dataframe(safe_bs, "AvailableDate")

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
        point_in_time_metrics = self._compute_point_in_time_metrics(
            safe_inc.sort_values("ReportDate"),
            safe_cf.sort_values("ReportDate"),
            safe_bs.sort_values("ReportDate") if not safe_bs.empty else pd.DataFrame(),
            current_price,
        )

        return {
            "ticker": ticker,
            "as_of_date": str(pit_context.analysis_date),
            "most_recent_report_date": str(pd.to_datetime(latest_inc["ReportDate"]).date()),
            "net_income": float(net_income) if not pd.isna(net_income) else 0.0,
            "free_cash_flow": float(free_cash_flow) if not pd.isna(free_cash_flow) else 0.0,
            "operating_cash_flow": float(operating_cf) if not pd.isna(operating_cf) else 0.0,
            "source": source,
            **point_in_time_metrics,
        }

    def _best_effort_fallback_snapshot(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        current_price: float | None,
        cached_snapshot: Dict[str, Any] | None,
        sec_snapshot: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        if cached_snapshot and "error" not in cached_snapshot:
            if sec_snapshot and "error" not in sec_snapshot:
                merged = self._merge_snapshots(cached_snapshot, sec_snapshot, "cache+sec")
                if not self._needs_enrichment(merged):
                    return merged
                cached_snapshot = merged
            return cached_snapshot

        alpha_snapshot = self._try_alpha_vantage_fallback(ticker, pit_context)
        if "error" not in alpha_snapshot:
            if sec_snapshot and "error" not in sec_snapshot:
                return self._merge_snapshots(alpha_snapshot, sec_snapshot, "alpha_vantage+sec")
            return alpha_snapshot

        if sec_snapshot and "error" not in sec_snapshot:
            return sec_snapshot

        return {"error": alpha_snapshot.get("error", "Missing fundamental data")}

    def fetch_fundamentals(self, ticker: str, pit_context: PointInTimeContext, current_price: float | None = None) -> Dict[str, Any]:
        """
        Fetches fundamental data and filters it based on the analysis_date.
        Because yfinance overrides historical fundamentals, this serves as a
        placeholder for a true point-in-time database (like Compustat/FMP).
        """
        logger.info(f"Fetching fundamental data for {ticker} as of {pit_context.analysis_date}")

        cached_inc, cached_cf, cached_bs = self._load_cached_reports(ticker)
        cached_snapshot: Dict[str, Any] | None = None
        if not cached_inc.empty and not cached_cf.empty:
            cached_snapshot = self._build_snapshot(
                ticker,
                pit_context,
                cached_inc,
                cached_cf,
                cached_bs,
                "cache",
                current_price,
            )

        sec_snapshot: Dict[str, Any] | None = None
        if self.prefer_sec_primary:
            sec_snapshot = self._try_sec_fallback(ticker, pit_context, current_price)
            if cached_snapshot and "error" not in cached_snapshot and "error" not in sec_snapshot:
                merged = self._merge_snapshots(cached_snapshot, sec_snapshot, "cache+sec")
                if not self._needs_enrichment(merged):
                    return merged
                cached_snapshot = merged
            elif sec_snapshot and "error" not in sec_snapshot and not self._needs_enrichment(sec_snapshot):
                return sec_snapshot
            elif cached_snapshot and "error" not in cached_snapshot and not self._needs_enrichment(cached_snapshot):
                return cached_snapshot
        elif cached_snapshot and "error" not in cached_snapshot and not self._needs_enrichment(cached_snapshot):
            return cached_snapshot

        if not self._yfinance_available():
            logger.info(f"Skipping Yahoo Finance fundamentals for {ticker} because cooldown is active.")
            return self._best_effort_fallback_snapshot(
                ticker,
                pit_context,
                current_price,
                cached_snapshot,
                sec_snapshot,
            )

        ticker_obj = yf.Ticker(ticker)

        try:
            income_stmt = ticker_obj.income_stmt
            cash_flow = ticker_obj.cashflow
            balance_sheet = ticker_obj.balance_sheet
        except YFRateLimitError:
            self._mark_yfinance_rate_limited(ticker)
            return self._best_effort_fallback_snapshot(
                ticker,
                pit_context,
                current_price,
                cached_snapshot,
                sec_snapshot,
            )
        except Exception as exc:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {exc}")
            return self._best_effort_fallback_snapshot(
                ticker,
                pit_context,
                current_price,
                cached_snapshot,
                sec_snapshot,
            )

        if income_stmt.empty or cash_flow.empty:
            return self._best_effort_fallback_snapshot(
                ticker,
                pit_context,
                current_price,
                cached_snapshot,
                sec_snapshot,
            )

        inc_df, cf_df, bs_df = self._prepare_statement_frames(income_stmt, cash_flow, balance_sheet)
        self._save_cached_reports(ticker, inc_df, cf_df, bs_df)
        snapshot = self._build_snapshot(ticker, pit_context, inc_df, cf_df, bs_df, "yfinance", current_price)
        if "error" not in snapshot and sec_snapshot and "error" not in sec_snapshot and self._needs_enrichment(snapshot):
            return self._merge_snapshots(snapshot, sec_snapshot, "yfinance+sec")
        return snapshot
