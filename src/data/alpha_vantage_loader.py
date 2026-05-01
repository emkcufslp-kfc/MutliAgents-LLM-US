import csv
import json
import logging
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from .point_in_time import PointInTimeContext
from ..runtime_config import get_secret

logger = logging.getLogger(__name__)

API_BASE_URL = "https://www.alphavantage.co/query"


class AlphaVantageError(Exception):
    """Raised when Alpha Vantage returns an error payload."""


def has_api_key() -> bool:
    return bool(get_secret("ALPHA_VANTAGE_API_KEY"))


def _get_api_key() -> str:
    api_key = get_secret("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise AlphaVantageError("ALPHA_VANTAGE_API_KEY is not configured")
    return api_key


def _request(function_name: str, params: dict[str, Any], datatype: str = "json") -> str:
    query = {
        "function": function_name,
        "apikey": _get_api_key(),
        "datatype": datatype,
        **params,
    }
    url = f"{API_BASE_URL}?{urlencode(query)}"

    try:
        with urlopen(url, timeout=20) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise AlphaVantageError(str(exc)) from exc

    if datatype == "json":
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AlphaVantageError("Invalid JSON from Alpha Vantage") from exc

        if "Information" in data:
            raise AlphaVantageError(data["Information"])
        if "Error Message" in data:
            raise AlphaVantageError(data["Error Message"])
        if "Note" in data:
            raise AlphaVantageError(data["Note"])

    return payload


def fetch_daily_prices(
    ticker: str,
    pit_context: PointInTimeContext,
    start_date: str = "2010-01-01",
) -> pd.DataFrame:
    payload = _request(
        "TIME_SERIES_DAILY_ADJUSTED",
        {"symbol": ticker, "outputsize": "full"},
        datatype="csv",
    )

    reader = csv.DictReader(StringIO(payload))
    rows = list(reader)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    rename_map = {
        "timestamp": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)

    df["Date"] = pd.to_datetime(df["Date"])
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Date", "Close", "Volume"]).sort_values("Date")
    df = df[df["Date"] >= pd.to_datetime(start_date)]
    return pit_context.filter_dataframe(df, "Date")


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_reports(
    reports: list[dict[str, Any]],
    pit_context: PointInTimeContext,
    fallback_lag_days: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for report in reports:
        fiscal_date = report.get("fiscalDateEnding")
        if not fiscal_date:
            continue
        available_date = pd.to_datetime(fiscal_date) + timedelta(days=fallback_lag_days)
        if available_date.date() <= pit_context.analysis_date:
            filtered.append(report)
    return filtered


def fetch_fundamentals(
    ticker: str,
    pit_context: PointInTimeContext,
    fallback_lag_days: int = 45,
) -> dict[str, Any]:
    income_payload = _request("INCOME_STATEMENT", {"symbol": ticker})
    cashflow_payload = _request("CASH_FLOW", {"symbol": ticker})

    income_data = json.loads(income_payload)
    cashflow_data = json.loads(cashflow_payload)

    income_reports = income_data.get("quarterlyReports") or income_data.get("annualReports") or []
    cashflow_reports = cashflow_data.get("quarterlyReports") or cashflow_data.get("annualReports") or []

    safe_income = _filter_reports(income_reports, pit_context, fallback_lag_days)
    safe_cashflow = _filter_reports(cashflow_reports, pit_context, fallback_lag_days)

    if not safe_income or not safe_cashflow:
        return {"error": "Data strictly filtered due to Point-In-Time limits."}

    latest_income = max(safe_income, key=lambda item: item.get("fiscalDateEnding", ""))
    latest_cashflow = max(safe_cashflow, key=lambda item: item.get("fiscalDateEnding", ""))

    net_income = _safe_float(latest_income.get("netIncome"))
    operating_cf = _safe_float(latest_cashflow.get("operatingCashflow"))
    free_cash_flow = _safe_float(latest_cashflow.get("freeCashFlow"))

    if free_cash_flow is None and operating_cf is not None:
        capex = _safe_float(latest_cashflow.get("capitalExpenditures"))
        if capex is not None:
            # Alpha Vantage may report capex as a negative outflow.
            free_cash_flow = operating_cf + capex

    return {
        "ticker": ticker,
        "as_of_date": str(pit_context.analysis_date),
        "most_recent_report_date": latest_income.get("fiscalDateEnding"),
        "net_income": net_income or 0.0,
        "free_cash_flow": free_cash_flow or 0.0,
        "operating_cash_flow": operating_cf or 0.0,
    }
