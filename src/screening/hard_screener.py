import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from ..data.fundamental_loader import FundamentalLoader
from ..data.point_in_time import PointInTimeContext
from ..data.price_loader import PriceLoader

logger = logging.getLogger(__name__)


class HardScreener:
    """
    Step 2 pre-screen with independent modules for:
    - Fundamental
    - Technical 1 (trend / structure)
    - Technical 2 (Williams VIX Fix)
    """

    def __init__(
        self,
        price_loader: PriceLoader,
        fundamental_loader: FundamentalLoader,
        screen_config: Dict[str, Any] | None = None,
    ):
        self.price_loader = price_loader
        self.fundamental_loader = fundamental_loader
        self.screen_config = screen_config or {}

    def _fundamental_config(self) -> Dict[str, Any]:
        return self.screen_config.get("fundamental", {}) or {}

    def _technical1_config(self) -> Dict[str, Any]:
        return self.screen_config.get("technical1", {}) or {}

    def _technical2_config(self) -> Dict[str, Any]:
        return self.screen_config.get("technical2", {}) or {}

    def _safe_float(self, value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric):
            return None
        return numeric

    def calculate_technical_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 252:
            return {"error": "Insufficient price history for 52-week and moving-average checks"}

        close = df["Close"].astype(float)
        low = df["Low"].astype(float) if "Low" in df.columns else close
        volume = df["Volume"].fillna(0).astype(float)

        current_price = close.iloc[-1]
        dma_50 = close.rolling(window=50).mean().iloc[-1]
        dma_150 = close.rolling(window=150).mean().iloc[-1]
        dma_200 = close.rolling(window=200).mean().iloc[-1]
        adv_20 = volume.rolling(window=20).mean().iloc[-1]
        high_52w = close.tail(252).max()
        low_52w = close.tail(252).min()

        def pct_change(lookback: int) -> float | None:
            if len(close) <= lookback:
                return None
            base_price = close.iloc[-lookback - 1]
            if base_price == 0:
                return None
            return ((current_price / base_price) - 1.0) * 100.0

        daily_returns = close.pct_change().dropna()
        volatility_20d = daily_returns.tail(20).std() * (252 ** 0.5) * 100.0 if len(daily_returns) >= 20 else None

        wvf_lookback = int(self._technical2_config().get("wvf_lookback_stddev_high", 22))
        wvf_bb_length = int(self._technical2_config().get("wvf_bollinger_band_length", 20))
        wvf_percentile_lookback = int(self._technical2_config().get("wvf_lookback_percentile_high", 50))
        bb_std_mult = float(self._technical2_config().get("wvf_bollinger_band_std_dev_mult", 2.0))

        highest_close = close.rolling(window=max(wvf_lookback, 2)).max()
        wvf_series = ((highest_close - low) / highest_close.replace(0, pd.NA)) * 100.0
        wvf_current = float(wvf_series.iloc[-1]) if not pd.isna(wvf_series.iloc[-1]) else None
        wvf_bb_mid = wvf_series.rolling(window=max(wvf_bb_length, 2)).mean()
        wvf_bb_std = wvf_series.rolling(window=max(wvf_bb_length, 2)).std()
        wvf_bb_upper = wvf_bb_mid + (wvf_bb_std * bb_std_mult)
        wvf_percentile_high = wvf_series.rolling(window=max(wvf_percentile_lookback, 2)).max() * float(
            self._technical2_config().get("wvf_highest_percentile", 0.85)
        )
        wvf_percentile_low = wvf_series.rolling(window=max(wvf_percentile_lookback, 2)).min() * float(
            self._technical2_config().get("wvf_lowest_percentile", 1.01)
        )

        within_52w_high_pct = ((high_52w - current_price) / high_52w * 100.0) if high_52w else None
        above_52w_low_pct = ((current_price / low_52w) - 1.0) * 100.0 if low_52w else None

        return {
            "current_price": float(current_price),
            "dma_50": float(dma_50),
            "dma_150": float(dma_150),
            "dma_200": float(dma_200),
            "adv_20": float(adv_20),
            "high_52w": float(high_52w),
            "low_52w": float(low_52w),
            "return_1m_pct": pct_change(21),
            "return_3m_pct": pct_change(63),
            "return_6m_pct": pct_change(126),
            "volatility_20d_pct": float(volatility_20d) if volatility_20d is not None else None,
            "price_vs_50dma_pct": ((current_price / dma_50) - 1.0) * 100.0 if dma_50 else None,
            "price_vs_150dma_pct": ((current_price / dma_150) - 1.0) * 100.0 if dma_150 else None,
            "price_vs_200dma_pct": ((current_price / dma_200) - 1.0) * 100.0 if dma_200 else None,
            "ma50_vs_ma150_pct": ((dma_50 / dma_150) - 1.0) * 100.0 if dma_150 else None,
            "ma150_vs_ma200_pct": ((dma_150 / dma_200) - 1.0) * 100.0 if dma_200 else None,
            "ma_spread_pct": ((dma_50 / dma_200) - 1.0) * 100.0 if dma_200 else None,
            "within_52w_high_pct": float(within_52w_high_pct) if within_52w_high_pct is not None else None,
            "above_52w_low_pct": float(above_52w_low_pct) if above_52w_low_pct is not None else None,
            "wvf_value": wvf_current,
            "wvf_bb_upper": float(wvf_bb_upper.iloc[-1]) if len(wvf_bb_upper) and not pd.isna(wvf_bb_upper.iloc[-1]) else None,
            "wvf_percentile_high_trigger": float(wvf_percentile_high.iloc[-1]) if len(wvf_percentile_high) and not pd.isna(wvf_percentile_high.iloc[-1]) else None,
            "wvf_percentile_low_trigger": float(wvf_percentile_low.iloc[-1]) if len(wvf_percentile_low) and not pd.isna(wvf_percentile_low.iloc[-1]) else None,
            "data_end_date": str(pd.to_datetime(df.iloc[-1]["Date"]).date()),
        }

    def _evaluate_fundamental_module(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        config = self._fundamental_config()
        eps_ttm_yoy = self._safe_float(fundamentals.get("eps_ttm_yoy_pct"))
        revenue_ttm_yoy = self._safe_float(fundamentals.get("revenue_ttm_yoy_pct"))
        roe_ttm = self._safe_float(fundamentals.get("roe_ttm_pct"))
        free_cash_flow = self._safe_float(fundamentals.get("free_cash_flow"))
        net_income = self._safe_float(fundamentals.get("net_income"))

        if all(value is None for value in [eps_ttm_yoy, revenue_ttm_yoy, roe_ttm]):
            detail_parts: list[str] = [
                "Fallback fundamental check used because PIT growth metrics were unavailable from the data source."
            ]
            legacy_checks = [
                ("Free Cash Flow", free_cash_flow, lambda value: value is not None and value > 0),
                ("Net Income", net_income, lambda value: value is not None and value > 0),
            ]
            passed_checks: list[bool] = []
            for label, value, predicate in legacy_checks:
                passed = predicate(value)
                passed_checks.append(passed)
                value_text = f"{value:.2f}" if value is not None else "N/A"
                detail_parts.append(f"{label}: {value_text} ({'PASS' if passed else 'FAIL'})")

            market_cap_b = self._safe_float(fundamentals.get("market_cap_b"))
            min_market_cap_b = float(config.get("min_market_cap_b", 10.0))
            if market_cap_b is not None:
                market_cap_ok = market_cap_b >= min_market_cap_b
                passed_checks.append(market_cap_ok)
                detail_parts.append(
                    f"Market Cap (B): {market_cap_b:.2f} vs {min_market_cap_b:.2f} (>=, {'PASS' if market_cap_ok else 'FAIL'})"
                )
            else:
                detail_parts.append("Market Cap (B): N/A (SKIP in fallback mode)")

            passed = all(passed_checks) if passed_checks else False
            applied = bool(config.get("apply_in_scan", True))
            return {
                "applied": applied,
                "passed": passed,
                "core_pass_count": sum(1 for item in passed_checks if item),
                "detail": " | ".join(detail_parts),
            }

        core_rules = [
            ("EPS TTM YoY", eps_ttm_yoy, config.get("min_eps_ttm_yoy_pct", 20.0), ">="),
            ("Revenue TTM YoY", revenue_ttm_yoy, config.get("min_revenue_ttm_yoy_pct", 20.0), ">="),
            ("ROE TTM", roe_ttm, config.get("min_roe_ttm_pct", 15.0), ">="),
        ]
        detail_parts: list[str] = []
        passed_core = 0
        for label, value, threshold, comparator in core_rules:
            numeric = self._safe_float(value)
            threshold_value = float(threshold)
            passed = numeric is not None and numeric >= threshold_value
            passed_core += 1 if passed else 0
            value_text = f"{numeric:.2f}" if numeric is not None else "N/A"
            detail_parts.append(f"{label}: {value_text} vs {threshold_value:.2f} ({'PASS' if passed else 'FAIL'})")

        aux_checks: list[tuple[str, bool, str]] = []
        roe_std = self._safe_float(fundamentals.get("roe_stability_std_pct"))
        roe_std_max = float(config.get("roe_stability_max_std_dev", 40.0))
        aux_checks.append(
            ("ROE Stability", roe_std is None or roe_std <= roe_std_max, f"{roe_std:.2f}" if roe_std is not None else "N/A")
        )
        market_cap_b = self._safe_float(fundamentals.get("market_cap_b"))
        min_market_cap_b = float(config.get("min_market_cap_b", 10.0))
        aux_checks.append(
            ("Market Cap (B)", market_cap_b is None or market_cap_b >= min_market_cap_b, f"{market_cap_b:.2f}" if market_cap_b is not None else "N/A")
        )

        for label, passed, value_text in aux_checks:
            threshold_text = f"{roe_std_max:.2f}" if label == "ROE Stability" else f"{min_market_cap_b:.2f}"
            comparator = "<=" if label == "ROE Stability" else ">="
            status = "SKIP" if value_text == "N/A" else ("PASS" if passed else "FAIL")
            detail_parts.append(f"{label}: {value_text} vs {threshold_text} ({comparator}, {status})")

        min_passed = int(config.get("min_passed_fundamental_rules", 3))
        pass_count_ok = passed_core >= min_passed
        aux_ok = all(item[1] for item in aux_checks)
        passed = pass_count_ok and aux_ok
        applied = bool(config.get("apply_in_scan", True))
        return {
            "applied": applied,
            "passed": passed,
            "core_pass_count": passed_core,
            "detail": " | ".join(detail_parts),
        }

    def _evaluate_technical1_module(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        config = self._technical1_config()
        applied = bool(config.get("apply_in_scan", False))
        detail_parts: list[str] = []
        passed_checks: list[bool] = []

        if config.get("require_close_above_all_mas", True):
            close_above = all(
                (self._safe_float(metrics.get(key)) or -10**9) > 0
                for key in ["price_vs_50dma_pct", "price_vs_150dma_pct", "price_vs_200dma_pct"]
            )
            passed_checks.append(close_above)
            detail_parts.append(f"Close > MA50/150/200 ({'PASS' if close_above else 'FAIL'})")

        if config.get("require_bullish_ma_stack", True):
            ma50 = self._safe_float(metrics.get("dma_50"))
            ma150 = self._safe_float(metrics.get("dma_150"))
            ma200 = self._safe_float(metrics.get("dma_200"))
            stack_ok = all(value is not None for value in [ma50, ma150, ma200]) and ma50 > ma150 > ma200
            passed_checks.append(stack_ok)
            detail_parts.append(f"MA50 > MA150 > MA200 ({'PASS' if stack_ok else 'FAIL'})")

        within_high = self._safe_float(metrics.get("within_52w_high_pct"))
        max_within_high = float(config.get("within_52w_high_pct", 25.0))
        within_high_ok = within_high is not None and within_high <= max_within_high
        passed_checks.append(within_high_ok)
        detail_parts.append(
            f"Within 52W High: {within_high:.2f}" if within_high is not None else "Within 52W High: N/A"
        )

        above_low = self._safe_float(metrics.get("above_52w_low_pct"))
        min_above_low = float(config.get("above_52w_low_pct", 25.0))
        above_low_ok = above_low is not None and above_low >= min_above_low
        passed_checks.append(above_low_ok)
        detail_parts.append(
            f"Above 52W Low: {above_low:.2f}" if above_low is not None else "Above 52W Low: N/A"
        )

        detail_parts[-2] = (
            f"Within 52W High: {within_high:.2f}" if within_high is not None else "Within 52W High: N/A"
        ) + f" vs {max_within_high:.2f} (<=, {'PASS' if within_high_ok else 'FAIL'})"
        detail_parts[-1] = (
            f"Above 52W Low: {above_low:.2f}" if above_low is not None else "Above 52W Low: N/A"
        ) + f" vs {min_above_low:.2f} (>=, {'PASS' if above_low_ok else 'FAIL'})"

        passed = all(passed_checks) if passed_checks else True
        return {"applied": applied, "passed": passed, "detail": " | ".join(detail_parts)}

    def _evaluate_technical2_module(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        config = self._technical2_config()
        applied = bool(config.get("apply_in_scan", False))
        enabled = bool(config.get("enable_alert", False))
        if not enabled:
            return {"applied": applied, "passed": True, "alert": False, "detail": "WVF alert disabled"}

        wvf_value = self._safe_float(metrics.get("wvf_value"))
        bb_upper = self._safe_float(metrics.get("wvf_bb_upper"))
        pct_high = self._safe_float(metrics.get("wvf_percentile_high_trigger"))
        pct_low = self._safe_float(metrics.get("wvf_percentile_low_trigger"))

        trigger_high = wvf_value is not None and (
            (bb_upper is not None and wvf_value >= bb_upper) or (pct_high is not None and wvf_value >= pct_high)
        )
        reset_low = wvf_value is not None and pct_low is not None and wvf_value <= pct_low
        alert = bool(trigger_high and not reset_low)
        detail = (
            f"WVF: {wvf_value:.2f}" if wvf_value is not None else "WVF: N/A"
        ) + (
            f" | BB Upper: {bb_upper:.2f}" if bb_upper is not None else " | BB Upper: N/A"
        ) + (
            f" | Pct High: {pct_high:.2f}" if pct_high is not None else " | Pct High: N/A"
        ) + (
            f" | Pct Low: {pct_low:.2f}" if pct_low is not None else " | Pct Low: N/A"
        ) + f" | Alert: {'YES' if alert else 'NO'}"
        return {"applied": applied, "passed": alert if applied else True, "alert": alert, "detail": detail}

    def _build_result_row(
        self,
        ticker: str,
        final_passed: bool,
        reason: str,
        metrics: Dict[str, Any] | None = None,
        fundamentals: Dict[str, Any] | None = None,
        module_results: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        metrics = metrics or {}
        fundamentals = fundamentals or {}
        module_results = module_results or {}
        fundamental = module_results.get("fundamental", {})
        technical1 = module_results.get("technical1", {})
        technical2 = module_results.get("technical2", {})
        return {
            "Ticker": ticker,
            "Status": "PASS" if final_passed else "FAIL",
            "Reason": reason,
            "Final Screen Pass": "PASS" if final_passed else "FAIL",
            "Price": metrics.get("current_price"),
            "20D ADV": metrics.get("adv_20"),
            "50 DMA": metrics.get("dma_50"),
            "150 DMA": metrics.get("dma_150"),
            "200 DMA": metrics.get("dma_200"),
            "52W High": metrics.get("high_52w"),
            "52W Low": metrics.get("low_52w"),
            "Price vs 50DMA %": metrics.get("price_vs_50dma_pct"),
            "Price vs 150DMA %": metrics.get("price_vs_150dma_pct"),
            "Price vs 200DMA %": metrics.get("price_vs_200dma_pct"),
            "50DMA vs 200DMA %": metrics.get("ma_spread_pct"),
            "Within 52W High %": metrics.get("within_52w_high_pct"),
            "Above 52W Low %": metrics.get("above_52w_low_pct"),
            "1M Return %": metrics.get("return_1m_pct"),
            "3M Return %": metrics.get("return_3m_pct"),
            "6M Return %": metrics.get("return_6m_pct"),
            "20D Volatility %": metrics.get("volatility_20d_pct"),
            "FCF": fundamentals.get("free_cash_flow"),
            "Net Income": fundamentals.get("net_income"),
            "Operating CF": fundamentals.get("operating_cash_flow"),
            "EPS TTM YoY %": fundamentals.get("eps_ttm_yoy_pct"),
            "Revenue TTM YoY %": fundamentals.get("revenue_ttm_yoy_pct"),
            "ROE TTM %": fundamentals.get("roe_ttm_pct"),
            "ROE Stability Std Dev": fundamentals.get("roe_stability_std_pct"),
            "Market Cap (B USD)": fundamentals.get("market_cap_b"),
            "Most Recent Report Date": fundamentals.get("most_recent_report_date"),
            "Price Data End": metrics.get("data_end_date"),
            "Price Source": metrics.get("price_source"),
            "Fundamental Source": fundamentals.get("source"),
            "Fundamental Applied": "YES" if fundamental.get("applied") else "NO",
            "Fundamental Pass": "PASS" if fundamental.get("passed") else "FAIL",
            "Fundamental Rules Passed": fundamental.get("core_pass_count"),
            "Fundamental Detail": fundamental.get("detail"),
            "Technical1 Applied": "YES" if technical1.get("applied") else "NO",
            "Technical1 Pass": "PASS" if technical1.get("passed") else "FAIL",
            "Technical1 Detail": technical1.get("detail"),
            "Technical2 Applied": "YES" if technical2.get("applied") else "NO",
            "Technical2 Alert": "YES" if technical2.get("alert") else "NO",
            "Technical2 Pass": "PASS" if technical2.get("passed") else "FAIL",
            "Technical2 Detail": technical2.get("detail"),
            "WVF Value": metrics.get("wvf_value"),
        }

    def evaluate_ticker(
        self,
        ticker: str,
        pit_context: PointInTimeContext,
        price_df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        logger.info(f"Screening {ticker}...")
        if price_df is None:
            price_df = self.price_loader.fetch_daily_prices(ticker, pit_context)
        if price_df.empty:
            row = self._build_result_row(ticker, False, "No price data")
            return {"ticker": ticker, "passed": False, "reason": "No price data", "row": row}

        metrics = self.calculate_technical_metrics(price_df)
        if "error" in metrics:
            row = self._build_result_row(ticker, False, metrics["error"])
            return {"ticker": ticker, "passed": False, "reason": metrics["error"], "row": row}
        metrics["price_source"] = price_df.attrs.get("source", "unknown")

        fundamentals = self.fundamental_loader.fetch_fundamentals(
            ticker,
            pit_context,
            current_price=metrics.get("current_price"),
        )
        if "error" in fundamentals:
            fundamentals = {**fundamentals, "source": fundamentals.get("source", "unavailable")}

        module_results = {
            "fundamental": self._evaluate_fundamental_module(fundamentals),
            "technical1": self._evaluate_technical1_module(metrics),
            "technical2": self._evaluate_technical2_module(metrics),
        }

        failures = [
            name
            for name, result in module_results.items()
            if result.get("applied") and not result.get("passed")
        ]
        final_passed = not failures
        reason = "Met all enabled modules" if final_passed else "Failed: " + ", ".join(failures)
        row = self._build_result_row(ticker, final_passed, reason, metrics, fundamentals, module_results)
        return {
            "ticker": ticker,
            "passed": final_passed,
            "reason": reason,
            "metrics": metrics,
            "fundamentals": fundamentals,
            "module_results": module_results,
            "row": row,
        }

    def run_screen(self, universe: List[str], pit_context: PointInTimeContext) -> Tuple[List[str], pd.DataFrame]:
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

        return passed_tickers, pd.DataFrame(detailed_results)
