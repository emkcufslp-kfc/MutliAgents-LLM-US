import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..data.fundamental_loader import FundamentalLoader
from ..data.point_in_time import PointInTimeContext
from ..data.price_loader import PriceLoader

logger = logging.getLogger(__name__)


class HardScreener:
    """
    Implements Phase 4: The Cost Control Gate.
    Filters out weak companies programmatically using strict rules
    before wasting expensive LLM API calls on them.
    """

    def __init__(self, price_loader: PriceLoader, fundamental_loader: FundamentalLoader):
        self.price_loader = price_loader
        self.fundamental_loader = fundamental_loader

        self.min_price = 10.0
        self.min_adv = 1_000_000
        self.require_positive_fcf = True
        self.require_positive_net_income = True
        self.require_above_200dma = True
        self.require_golden_cross = True

    def calculate_technical_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates trend, liquidity, and return metrics from daily history."""
        if len(df) < 200:
            return {"error": "Insufficient price history for 200DMA"}

        price_series = df["Close"].astype(float)
        volume_series = df["Volume"].fillna(0).astype(float)

        current_price = price_series.iloc[-1]
        dma_200 = price_series.rolling(window=200).mean().iloc[-1]
        dma_50 = price_series.rolling(window=50).mean().iloc[-1]
        adv_20 = volume_series.rolling(window=20).mean().iloc[-1]

        def pct_change(lookback: int) -> float | None:
            if len(price_series) <= lookback:
                return None
            base_price = price_series.iloc[-lookback - 1]
            if base_price == 0:
                return None
            return ((current_price / base_price) - 1.0) * 100.0

        daily_returns = price_series.pct_change().dropna()
        volatility_20d = None
        if len(daily_returns) >= 20:
            volatility_20d = daily_returns.tail(20).std() * (252 ** 0.5) * 100.0

        price_vs_50dma = ((current_price / dma_50) - 1.0) * 100.0 if dma_50 else None
        price_vs_200dma = ((current_price / dma_200) - 1.0) * 100.0 if dma_200 else None
        moving_average_spread = ((dma_50 / dma_200) - 1.0) * 100.0 if dma_200 else None

        return {
            "current_price": float(current_price),
            "dma_50": float(dma_50),
            "dma_200": float(dma_200),
            "adv_20": float(adv_20),
            "return_1m_pct": pct_change(21),
            "return_3m_pct": pct_change(63),
            "return_6m_pct": pct_change(126),
            "volatility_20d_pct": float(volatility_20d) if volatility_20d is not None else None,
            "price_vs_50dma_pct": float(price_vs_50dma) if price_vs_50dma is not None else None,
            "price_vs_200dma_pct": float(price_vs_200dma) if price_vs_200dma is not None else None,
            "ma_spread_pct": float(moving_average_spread) if moving_average_spread is not None else None,
            "data_end_date": str(pd.to_datetime(df.iloc[-1]["Date"]).date()),
            "is_above_200dma": current_price > dma_200,
            "is_golden_cross": dma_50 > dma_200,
        }

    def _build_result_row(
        self,
        ticker: str,
        passed: bool,
        reason: str,
        metrics: Dict[str, Any] | None = None,
        fundamentals: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metrics = metrics or {}
        fundamentals = fundamentals or {}
        return {
            "Ticker": ticker,
            "Status": "PASS" if passed else "FAIL",
            "Reason": reason,
            "Price": metrics.get("current_price"),
            "20D ADV": metrics.get("adv_20"),
            "50 DMA": metrics.get("dma_50"),
            "200 DMA": metrics.get("dma_200"),
            "Price vs 50DMA %": metrics.get("price_vs_50dma_pct"),
            "Price vs 200DMA %": metrics.get("price_vs_200dma_pct"),
            "50DMA vs 200DMA %": metrics.get("ma_spread_pct"),
            "1M Return %": metrics.get("return_1m_pct"),
            "3M Return %": metrics.get("return_3m_pct"),
            "6M Return %": metrics.get("return_6m_pct"),
            "20D Volatility %": metrics.get("volatility_20d_pct"),
            "FCF": fundamentals.get("free_cash_flow"),
            "Net Income": fundamentals.get("net_income"),
            "Operating CF": fundamentals.get("operating_cash_flow"),
            "Most Recent Report Date": fundamentals.get("most_recent_report_date"),
            "Price Data End": metrics.get("data_end_date"),
            "Price Source": metrics.get("price_source"),
            "Fundamental Source": fundamentals.get("source"),
        }

    def evaluate_ticker(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        price_df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a single ticker against the hard rules.
        """
        logger.info(f"Screening {ticker}...")

        if price_df is None:
            price_df = self.price_loader.fetch_daily_prices(ticker, pit_context)

        if price_df.empty:
            return {
                "ticker": ticker,
                "passed": False,
                "reason": "No price data",
                "row": self._build_result_row(ticker, False, "No price data"),
            }

        techs = self.calculate_technical_metrics(price_df)
        if "error" in techs:
            return {
                "ticker": ticker,
                "passed": False,
                "reason": techs["error"],
                "row": self._build_result_row(ticker, False, techs["error"]),
            }

        techs["price_source"] = price_df.attrs.get("source", "unknown")

        if techs["current_price"] < self.min_price:
            reason = f"Price below ${self.min_price:.2f}"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs),
            }

        if techs["adv_20"] < self.min_adv:
            reason = "Illiquid (20D ADV < 1M)"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs),
            }

        if self.require_above_200dma and not techs["is_above_200dma"]:
            reason = "Downtrend (below 200DMA)"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs),
            }

        if self.require_golden_cross and not techs["is_golden_cross"]:
            reason = "No golden cross (50DMA < 200DMA)"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs),
            }

        fundamentals = self.fundamental_loader.fetch_fundamentals(ticker, pit_context)
        if "error" in fundamentals:
            reason = f"Fundamental error: {fundamentals['error']}"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs, fundamentals),
            }

        if self.require_positive_fcf and fundamentals.get("free_cash_flow", 0) <= 0:
            reason = "Negative free cash flow"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs, fundamentals),
            }

        if self.require_positive_net_income and fundamentals.get("net_income", 0) <= 0:
            reason = "Negative net income"
            return {
                "ticker": ticker,
                "passed": False,
                "reason": reason,
                "row": self._build_result_row(ticker, False, reason, techs, fundamentals),
            }

        return {
            "ticker": ticker,
            "passed": True,
            "reason": "Met all criteria",
            "metrics": techs,
            "fundamentals": fundamentals,
            "row": self._build_result_row(ticker, True, "Met all criteria", techs, fundamentals),
        }

    def run_screen(self, universe: List[str], pit_context: PointInTimeContext) -> Tuple[List[str], pd.DataFrame]:
        """
        Runs the full universe through the hard screener.
        Returns a tuple: (List of passed tickers, DataFrame of full results).
        """
        passed_tickers: list[str] = []
        detailed_results: list[dict[str, Any]] = []

        logger.info(f"Initiating hard screener as of {pit_context.analysis_date} for {len(universe)} tickers")
        price_map = self.price_loader.fetch_daily_prices_batch(universe, pit_context)

        for ticker in universe:
            try:
                result = self.evaluate_ticker(ticker, pit_context, price_map.get(ticker))
            except Exception as exc:
                logger.exception(f"Unexpected screening failure for {ticker}: {exc}")
                result = {
                    "ticker": ticker,
                    "passed": False,
                    "reason": f"Unexpected screening failure: {exc}",
                    "row": self._build_result_row(ticker, False, f"Unexpected screening failure: {exc}"),
                }

            if result["passed"]:
                passed_tickers.append(ticker)

            detailed_results.append(result["row"])

        df_results = pd.DataFrame(detailed_results)
        return passed_tickers, df_results
