import json
import logging
from datetime import date, timedelta
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .point_in_time import PointInTimeContext
from ..runtime_config import get_secret

logger = logging.getLogger(__name__)

SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = "MultiLLM-US/1.0 support@example.com"
ALLOWED_FORMS = {"10-Q", "10-Q/A", "10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}


class SecEdgarError(Exception):
    """Raised when SEC EDGAR data cannot be fetched or parsed."""


def _headers() -> dict[str, str]:
    return {
        "User-Agent": get_secret("SEC_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def _request_json(url: str) -> Any:
    request = Request(url, headers=_headers())
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError) as exc:
        raise SecEdgarError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise SecEdgarError(f"Invalid JSON payload from SEC for {url}") from exc


@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, str]:
    payload = _request_json(SEC_TICKER_URL)
    mapping: dict[str, str] = {}
    iterable = payload.values() if isinstance(payload, dict) else payload
    for item in iterable:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        cik = str(item.get("cik_str", "")).strip()
        if ticker and cik:
            mapping[ticker] = cik.zfill(10)
    return mapping


def _resolve_cik(ticker: str) -> str:
    cik = _ticker_map().get(ticker.upper())
    if not cik:
        raise SecEdgarError(f"CIK not found for ticker {ticker}")
    return cik


def _facts_payload(ticker: str) -> dict[str, Any]:
    cik = _resolve_cik(ticker)
    payload = _request_json(SEC_COMPANYFACTS_URL.format(cik=cik))
    if not isinstance(payload, dict):
        raise SecEdgarError(f"Unexpected SEC companyfacts payload for {ticker}")
    return payload


def _coerce_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _collect_entries(facts_payload: dict[str, Any], taxonomy_tags: list[tuple[str, str]]) -> list[dict[str, Any]]:
    facts = facts_payload.get("facts", {})
    collected: list[dict[str, Any]] = []
    for taxonomy, tag in taxonomy_tags:
        tag_payload = facts.get(taxonomy, {}).get(tag, {})
        units = tag_payload.get("units", {})
        for unit, entries in units.items():
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                enriched = dict(entry)
                enriched["_taxonomy"] = taxonomy
                enriched["_tag"] = tag
                enriched["_unit"] = unit
                collected.append(enriched)
    return collected


def _filter_duration_entries(
    entries: list[dict[str, Any]],
    analysis_date: date,
    min_days: int,
    max_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        form = str(entry.get("form", "")).upper()
        start = _coerce_date(entry.get("start"))
        end = _coerce_date(entry.get("end"))
        filed = _coerce_date(entry.get("filed"))
        value = _coerce_float(entry.get("val"))
        if not start or not end or not filed or value is None:
            continue
        if filed > analysis_date or end > analysis_date:
            continue
        if form not in ALLOWED_FORMS:
            continue
        duration_days = (end - start).days
        if duration_days < min_days or duration_days > max_days:
            continue
        rows.append(
            {
                "start": start,
                "end": end,
                "filed": filed,
                "value": value,
                "form": form,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["start", "end", "filed", "value", "form"])

    df = pd.DataFrame(rows).sort_values(["end", "filed"])
    return df.drop_duplicates(subset=["end"], keep="last").sort_values("end")


def _filter_instant_entries(entries: list[dict[str, Any]], analysis_date: date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        form = str(entry.get("form", "")).upper()
        end = _coerce_date(entry.get("end"))
        filed = _coerce_date(entry.get("filed"))
        value = _coerce_float(entry.get("val"))
        if not end or not filed or value is None:
            continue
        if filed > analysis_date or end > analysis_date:
            continue
        if form and form not in ALLOWED_FORMS:
            continue
        rows.append({"end": end, "filed": filed, "value": value, "form": form})

    if not rows:
        return pd.DataFrame(columns=["end", "filed", "value", "form"])

    df = pd.DataFrame(rows).sort_values(["end", "filed"])
    return df.drop_duplicates(subset=["end"], keep="last").sort_values("end")


def _latest_value(series_df: pd.DataFrame, as_of: date) -> float | None:
    if series_df.empty:
        return None
    safe = series_df[series_df["end"] <= as_of]
    if safe.empty:
        return None
    return _coerce_float(safe.iloc[-1]["value"])


def _compute_ttm_yoy(series_df: pd.DataFrame) -> float | None:
    if len(series_df) < 5:
        return None
    rolling = series_df[["end", "value"]].copy()
    rolling["ttm"] = rolling["value"].rolling(window=4).sum()
    rolling = rolling.dropna(subset=["ttm"])
    if len(rolling) < 5:
        return None
    latest = _coerce_float(rolling.iloc[-1]["ttm"])
    prior = _coerce_float(rolling.iloc[-5]["ttm"])
    if latest is None or prior in (None, 0):
        return None
    return ((latest / prior) - 1.0) * 100.0


def _compute_roe_metrics(net_income_q_df: pd.DataFrame, equity_df: pd.DataFrame) -> tuple[float | None, float | None]:
    if net_income_q_df.empty or equity_df.empty:
        return None, None

    roe_samples: list[float] = []
    for _, row in net_income_q_df.iterrows():
        equity = _latest_value(equity_df, row["end"])
        if equity in (None, 0):
            continue
        annualized_roe = (float(row["value"]) * 4.0 / equity) * 100.0
        roe_samples.append(annualized_roe)

    if not roe_samples:
        return None, None

    series = pd.Series(roe_samples).tail(4)
    roe_ttm = float(series.mean())
    roe_std = float(series.std(ddof=0)) if len(series) >= 2 else 0.0
    return roe_ttm, roe_std


def fetch_fundamentals(
    ticker: str,
    pit_context: PointInTimeContext,
    current_price: float | None = None,
) -> dict[str, Any]:
    payload = _facts_payload(ticker)
    analysis_date = pit_context.analysis_date

    revenue_entries = _collect_entries(
        payload,
        [
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),
            ("us-gaap", "SalesRevenueNet"),
            ("us-gaap", "Revenues"),
        ],
    )
    net_income_entries = _collect_entries(
        payload,
        [
            ("us-gaap", "NetIncomeLoss"),
            ("us-gaap", "ProfitLoss"),
        ],
    )
    eps_entries = _collect_entries(
        payload,
        [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasicAndDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
        ],
    )
    equity_entries = _collect_entries(
        payload,
        [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
            ("us-gaap", "CommonStockEquity"),
        ],
    )
    shares_entries = _collect_entries(
        payload,
        [
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
            ("dei", "EntityPublicFloat"),
        ],
    )

    revenue_q_df = _filter_duration_entries(revenue_entries, analysis_date, 75, 120)
    net_income_q_df = _filter_duration_entries(net_income_entries, analysis_date, 75, 120)
    eps_q_df = _filter_duration_entries(eps_entries, analysis_date, 75, 120)
    equity_df = _filter_instant_entries(equity_entries, analysis_date)
    shares_df = _filter_instant_entries(shares_entries, analysis_date)

    eps_ttm_yoy = _compute_ttm_yoy(eps_q_df)
    revenue_ttm_yoy = _compute_ttm_yoy(revenue_q_df)
    roe_ttm, roe_stability_std = _compute_roe_metrics(net_income_q_df, equity_df)

    latest_net_income = _latest_value(net_income_q_df, analysis_date)
    latest_revenue_end = revenue_q_df.iloc[-1]["end"] if not revenue_q_df.empty else None
    shares_outstanding = _latest_value(shares_df, analysis_date)
    market_cap_b = None
    if current_price and shares_outstanding:
        market_cap_b = (current_price * shares_outstanding) / 1_000_000_000.0

    if (
        eps_ttm_yoy is None
        and revenue_ttm_yoy is None
        and roe_ttm is None
        and latest_net_income is None
        and market_cap_b is None
    ):
        return {"error": f"SEC companyfacts did not provide usable PIT metrics for {ticker}"}

    return {
        "ticker": ticker,
        "as_of_date": str(analysis_date),
        "most_recent_report_date": str(latest_revenue_end) if latest_revenue_end else None,
        "net_income": latest_net_income if latest_net_income is not None else 0.0,
        "free_cash_flow": None,
        "operating_cash_flow": None,
        "eps_ttm_yoy_pct": eps_ttm_yoy,
        "revenue_ttm_yoy_pct": revenue_ttm_yoy,
        "roe_ttm_pct": roe_ttm,
        "roe_stability_std_pct": roe_stability_std,
        "market_cap_b": market_cap_b,
        "source": "sec",
    }
