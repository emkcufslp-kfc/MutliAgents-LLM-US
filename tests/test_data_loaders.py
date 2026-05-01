import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.fundamental_loader import FundamentalLoader
from src.data.point_in_time import PointInTimeContext
from src.data.price_loader import PriceLoader


TEST_CACHE_ROOT = Path("D:/Codex projects/Multi LLM/TradingAgents/us_hedgefund_agents/.codex_tmp/test_cache")


class FakeYFRateLimitError(Exception):
    pass


class PriceLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = TEST_CACHE_ROOT / self._testMethodName
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def test_batch_rate_limit_does_not_fan_out_into_single_yahoo_requests(self) -> None:
        loader = PriceLoader(cache_dir=self.test_dir, yf_cooldown_seconds=300)
        pit_context = PointInTimeContext(date(2026, 5, 1))

        with patch("src.data.price_loader.YFRateLimitError", FakeYFRateLimitError):
            with patch("src.data.price_loader.yf.download", side_effect=FakeYFRateLimitError()) as mock_download:
                with patch("src.data.price_loader.alpha_vantage_loader.has_api_key", return_value=False):
                    results = loader.fetch_daily_prices_batch(["AAPL", "MSFT"], pit_context, chunk_size=50)

        self.assertEqual(mock_download.call_count, 1)
        self.assertTrue(results["AAPL"].empty)
        self.assertTrue(results["MSFT"].empty)


class FundamentalLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = TEST_CACHE_ROOT / self._testMethodName
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def test_sec_primary_can_skip_yahoo_when_pit_metrics_are_complete(self) -> None:
        loader = FundamentalLoader(cache_dir=self.test_dir, prefer_sec_primary=True, yf_cooldown_seconds=300)
        pit_context = PointInTimeContext(date(2026, 5, 1))
        sec_snapshot = {
            "ticker": "AAPL",
            "as_of_date": "2026-05-01",
            "most_recent_report_date": "2025-12-31",
            "net_income": 1.0,
            "free_cash_flow": None,
            "operating_cash_flow": None,
            "eps_ttm_yoy_pct": 21.0,
            "revenue_ttm_yoy_pct": 22.0,
            "roe_ttm_pct": 18.0,
            "roe_stability_std_pct": 4.0,
            "market_cap_b": 100.0,
            "source": "sec",
        }

        with patch.object(loader, "_load_cached_reports", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())):
            with patch.object(loader, "_try_sec_fallback", return_value=sec_snapshot):
                with patch("src.data.fundamental_loader.yf.Ticker") as mock_ticker:
                    result = loader.fetch_fundamentals("AAPL", pit_context, current_price=200.0)

        self.assertEqual(result["source"], "sec")
        self.assertEqual(result["eps_ttm_yoy_pct"], 21.0)
        mock_ticker.assert_not_called()

    def test_yahoo_summary_fallback_enriches_missing_pit_metrics(self) -> None:
        loader = FundamentalLoader(cache_dir=self.test_dir, prefer_sec_primary=True, yf_cooldown_seconds=300)
        pit_context = PointInTimeContext(date(2026, 5, 1))
        sec_partial_snapshot = {
            "ticker": "AAPL",
            "as_of_date": "2026-05-01",
            "most_recent_report_date": "2025-12-31",
            "net_income": 1.0,
            "free_cash_flow": None,
            "operating_cash_flow": None,
            "eps_ttm_yoy_pct": None,
            "revenue_ttm_yoy_pct": None,
            "roe_ttm_pct": None,
            "roe_stability_std_pct": None,
            "market_cap_b": None,
            "source": "sec",
        }
        mock_ticker_obj = MagicMock()
        mock_ticker_obj.fast_info = {"market_cap": 250_000_000_000}
        mock_ticker_obj.get_info.return_value = {
            "earningsQuarterlyGrowth": 0.35,
            "revenueGrowth": 0.22,
            "returnOnEquity": 0.18,
            "marketCap": 250_000_000_000,
        }

        with patch.object(loader, "_load_cached_reports", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())):
            with patch.object(loader, "_try_sec_fallback", return_value=sec_partial_snapshot):
                with patch("src.data.fundamental_loader.yf.Ticker", return_value=mock_ticker_obj):
                    with patch.object(mock_ticker_obj, "income_stmt", pd.DataFrame()):
                        with patch.object(mock_ticker_obj, "cashflow", pd.DataFrame()):
                            with patch.object(mock_ticker_obj, "balance_sheet", pd.DataFrame()):
                                result = loader.fetch_fundamentals("AAPL", pit_context, current_price=200.0)

        self.assertEqual(result["source"], "sec+yfinance_info")
        self.assertEqual(result["eps_ttm_yoy_pct"], 35.0)
        self.assertEqual(result["revenue_ttm_yoy_pct"], 22.0)
        self.assertEqual(result["roe_ttm_pct"], 18.0)
        self.assertEqual(result["market_cap_b"], 250.0)


if __name__ == "__main__":
    unittest.main()
