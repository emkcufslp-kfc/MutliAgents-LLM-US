import ast
import io
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import HedgeFundGraph
from src.data.fundamental_loader import FundamentalLoader
from src.data.macro_loader import MacroLoader
from src.data.point_in_time import PointInTimeContext
from src.data.price_loader import PriceLoader
from src.data import alpha_vantage_loader
from src.runtime_config import get_secret
from src.screening.hard_screener import HardScreener

st.set_page_config(page_title="Multi-Agent Hedge Fund", layout="wide")

APP_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = APP_ROOT / "data" / "cache"
INDEX_SOURCES = {
    "SP500": {
        "csv_url": "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv",
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "candidate_columns": ["Symbol", "Ticker symbol", "Ticker"],
        "min_count": 450,
        "cache_file": CACHE_DIR / "sp500_members.csv",
    },
    "Nasdaq 100": {
        "csv_url": "https://github.com/Gary-Strauss/NASDAQ100_Constituents/raw/refs/heads/master/data/nasdaq100_constituents.csv",
        "url": "https://en.wikipedia.org/wiki/Nasdaq-100",
        "candidate_columns": ["Ticker", "Symbol"],
        "min_count": 90,
        "cache_file": CACHE_DIR / "nasdaq100_members.csv",
    },
}
UNIVERSE_CHOICES = {
    "S&P 500 (503 stocks)": ["SP500"],
    "Nasdaq 100 (100 stocks)": ["Nasdaq 100"],
    "S&P 500 + Nasdaq 100": ["SP500", "Nasdaq 100"],
}


def inject_app_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg: #0a0f1a;
  --panel: #111827;
  --panel-2: #182131;
  --panel-3: #0f1725;
  --line: rgba(160, 174, 192, 0.18);
  --text: #f6f7fb;
  --muted: #9aa6bd;
  --soft: #c9d2e3;
  --accent: #7cc4ff;
  --accent-2: #33e0c4;
  --danger: #ff7f8f;
  --warning: #ffc86b;
}

html, body, [class*="css"]  {
  font-family: "Space Grotesk", "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at top left, rgba(51, 224, 196, 0.09), transparent 28%),
    radial-gradient(circle at top right, rgba(124, 196, 255, 0.10), transparent 24%),
    linear-gradient(180deg, #0a0f1a 0%, #08111b 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f1420 0%, #121a28 100%);
  border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--text);
  letter-spacing: 0.02em;
}

.side-label {
  color: #bcc7da;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-top: 1.25rem;
  margin-bottom: 0.45rem;
}

.side-note {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.55;
}

.hero {
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(17,24,39,0.96), rgba(24,33,49,0.92));
  border-radius: 22px;
  padding: 1.4rem 1.5rem 1.3rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: 0 20px 40px rgba(2, 6, 23, 0.28);
}

.hero-kicker {
  color: var(--accent-2);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  margin-bottom: 0.4rem;
}

.hero-title {
  color: var(--text);
  font-size: 2.35rem;
  font-weight: 700;
  line-height: 1.08;
  margin: 0;
}

.hero-subtitle {
  color: var(--soft);
  font-size: 1rem;
  margin-top: 0.85rem;
  max-width: 62rem;
  line-height: 1.7;
}

.section-card {
  border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(16,23,37,0.95), rgba(12,18,30,0.95));
  border-radius: 20px;
  padding: 1.1rem 1.15rem;
  margin: 0.85rem 0 1rem 0;
}

.section-kicker {
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 0.35rem;
}

.section-title {
  color: var(--text);
  font-size: 1.24rem;
  font-weight: 700;
  margin-bottom: 0.2rem;
}

.section-copy {
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.6;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 0.85rem;
  margin: 0.6rem 0 1.0rem 0;
}

.status-card {
  border: 1px solid var(--line);
  background: rgba(14, 21, 34, 0.88);
  border-radius: 18px;
  padding: 1rem 1rem 0.95rem 1rem;
}

.status-label {
  color: var(--muted);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-bottom: 0.35rem;
}

.status-value {
  color: var(--text);
  font-size: 1.35rem;
  font-weight: 700;
}

.status-detail {
  color: var(--soft);
  font-size: 0.86rem;
  margin-top: 0.35rem;
}

.mini-note {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.55;
  margin-top: 0.5rem;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stDateInput"] > div,
div[data-testid="stTextArea"] textarea,
div[data-testid="stFileUploader"] section {
  border-radius: 14px !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stDateInput"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stRadio"] label {
  color: #d8e1f0 !important;
  font-weight: 600 !important;
}

div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stDateInput"] input,
div[data-testid="stTextArea"] textarea {
  background: #162032 !important;
  border: 1px solid rgba(124, 196, 255, 0.28) !important;
  color: var(--text) !important;
  font-family: "IBM Plex Mono", monospace !important;
}

div[data-testid="stFileUploader"] section {
  background: #162032 !important;
  border: 1px dashed rgba(124, 196, 255, 0.42) !important;
  padding: 0.85rem !important;
}

.stButton > button,
.stDownloadButton > button {
  width: 100%;
  border-radius: 14px !important;
  border: 1px solid rgba(124, 196, 255, 0.34) !important;
  background: linear-gradient(135deg, #20304a 0%, #172335 100%) !important;
  color: #f7fbff !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  box-shadow: 0 10px 24px rgba(2, 6, 23, 0.22);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: rgba(51, 224, 196, 0.55) !important;
  color: white !important;
}

div[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 0.85rem 1rem;
  background: rgba(15, 23, 37, 0.9);
}

div[data-testid="stMetricLabel"] {
  color: var(--muted) !important;
}

div[data-testid="stMetricValue"] {
  color: var(--text) !important;
}

div[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 18px;
  overflow: hidden;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_hero(screen_state: dict | None = None) -> None:
    universe_count = len(screen_state["universe"]) if screen_state else 0
    survivor_count = len(screen_state["survivors"]) if screen_state else 0
    effective_date = screen_state["effective_date"] if screen_state else get_latest_us_trading_day()
    selected = st.session_state.selected_ticker or "None"

    st.markdown(
        f"""
<div class="hero">
  <div class="hero-kicker">Institutional Research Workflow</div>
  <h1 class="hero-title">Multi-Agent US Equity Hedge Fund</h1>
  <div class="hero-subtitle">
    A live screening and analysis workspace for stock investors. Build the universe, run the point-in-time filter,
    then click into qualified names to inspect the actual data, scoring logic, execution framework, and risk allocation.
  </div>
</div>
<div class="status-grid">
  <div class="status-card">
    <div class="status-label">Effective Screen Date</div>
    <div class="status-value">{effective_date}</div>
    <div class="status-detail">Latest valid US trading day is used automatically.</div>
  </div>
  <div class="status-card">
    <div class="status-label">Universe Size</div>
    <div class="status-value">{universe_count}</div>
    <div class="status-detail">Combined from index source, CSV upload, and direct tickers.</div>
  </div>
  <div class="status-card">
    <div class="status-label">Qualified Names</div>
    <div class="status-value">{survivor_count}</div>
    <div class="status-detail">Names that passed the hard institutional gate.</div>
  </div>
  <div class="status-card">
    <div class="status-label">Selected Ticker</div>
    <div class="status-value">{selected}</div>
    <div class="status-detail">Click a qualified ticker to trigger Step 3 analysis.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
<div class="section-card">
  <div class="section-kicker">{kicker}</div>
  <div class="section-title">{title}</div>
  <div class="section-copy">{copy}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "app_log": [],
        "last_screen_results": None,
        "last_survivors": [],
        "last_export_df": None,
        "screen_state": None,
        "selected_ticker": None,
        "analysis_cache": {},
        "analysis_order": [],
        "fundamental_filter_config": {},
        "technical1_filter_config": {},
        "technical2_filter_config": {},
        "filter_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.app_log.insert(0, f"[{timestamp}] {message}")
    st.session_state.app_log = st.session_state.app_log[:100]


def mask_secret(secret: str | None) -> str:
    if not secret:
        return "missing"
    if len(secret) <= 6:
        return "*" * len(secret)
    return f"{secret[:3]}***{secret[-2:]}"


@st.cache_data(ttl=300, show_spinner=False)
def run_api_diagnostics() -> list[dict]:
    diagnostics: list[dict] = []

    xai_key = get_secret("XAI_API_KEY")
    diagnostics.append(
        {
            "API": "xAI / Grok",
            "Secret": "XAI_API_KEY",
            "Configured": "Yes" if xai_key else "No",
            "Status": "Configured" if xai_key else "Missing secret",
            "Detail": f"Key preview: {mask_secret(xai_key)}. Live auth test is skipped to avoid unnecessary paid calls.",
        }
    )

    fred_key = get_secret("FRED_API_KEY")
    if not fred_key:
        diagnostics.append(
            {
                "API": "FRED",
                "Secret": "FRED_API_KEY",
                "Configured": "No",
                "Status": "Missing secret",
                "Detail": "Macro regime will fall back to unavailable mode until the FRED key is added.",
            }
        )
    else:
        try:
            macro_loader = MacroLoader()
            if not macro_loader.fred:
                raise RuntimeError("fred client did not initialize")
            series = macro_loader.fred.get_series("UNRATE")
            latest = float(series.iloc[-1]) if len(series) else None
            diagnostics.append(
                {
                    "API": "FRED",
                    "Secret": "FRED_API_KEY",
                    "Configured": "Yes",
                    "Status": "Working",
                    "Detail": f"UNRATE test fetch succeeded. Latest value: {latest:.1f}" if latest is not None else "UNRATE test fetch returned no rows.",
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "API": "FRED",
                    "Secret": "FRED_API_KEY",
                    "Configured": "Yes",
                    "Status": "Error",
                    "Detail": str(exc),
                }
            )

    alpha_key = get_secret("ALPHA_VANTAGE_API_KEY")
    if not alpha_key:
        diagnostics.append(
            {
                "API": "Alpha Vantage",
                "Secret": "ALPHA_VANTAGE_API_KEY",
                "Configured": "No",
                "Status": "Missing secret",
                "Detail": "Yahoo Finance rate-limit fallback is disabled until this key is added.",
            }
        )
    else:
        try:
            payload = alpha_vantage_loader._request(
                "TIME_SERIES_DAILY_ADJUSTED",
                {"symbol": "SPY", "outputsize": "compact"},
                datatype="csv",
            )
            header = payload.splitlines()[0] if payload else ""
            if "timestamp" not in header.lower():
                raise RuntimeError("unexpected Alpha Vantage CSV response")
            diagnostics.append(
                {
                    "API": "Alpha Vantage",
                    "Secret": "ALPHA_VANTAGE_API_KEY",
                    "Configured": "Yes",
                    "Status": "Working",
                    "Detail": "SPY daily-price test fetch succeeded.",
                }
            )
        except Exception as exc:
            diagnostics.append(
                {
                    "API": "Alpha Vantage",
                    "Secret": "ALPHA_VANTAGE_API_KEY",
                    "Configured": "Yes",
                    "Status": "Error",
                    "Detail": str(exc),
                }
            )

    return diagnostics


def render_api_diagnostics() -> None:
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Refresh API Diagnostics"):
            run_api_diagnostics.clear()
    with c2:
        st.caption("Checks are cached for 5 minutes to avoid unnecessary API traffic.")

    diagnostics_df = pd.DataFrame(run_api_diagnostics())
    st.dataframe(diagnostics_df, hide_index=True)

    failing = diagnostics_df[~diagnostics_df["Status"].isin(["Working", "Configured"])]
    if not failing.empty:
        for _, row in failing.iterrows():
            st.warning(f"{row['API']}: {row['Status']} - {row['Detail']}")


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def parse_direct_tickers(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    parts = re.split(r"[\s,;]+", raw_text)
    return list(dict.fromkeys(normalize_ticker(part) for part in parts if normalize_ticker(part)))


def load_uploaded_tickers(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []

    raw = uploaded_file.getvalue()
    if not raw:
        return []

    df = pd.read_csv(io.BytesIO(raw))
    if df.empty:
        return []

    candidate_columns = ["ticker", "symbol", "Ticker", "Symbol"]
    selected_column = next((column for column in candidate_columns if column in df.columns), df.columns[0])
    tickers = [normalize_ticker(value) for value in df[selected_column].dropna().astype(str)]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def build_export_filename(target_date: date) -> str:
    return f"US-{target_date.strftime('%d-%b-%y')}.csv"


def try_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_number(value, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    numeric = try_float(value)
    if numeric is None:
        return "N/A"
    return f"{prefix}{numeric:,.{decimals}f}{suffix}"


def format_compact_currency(value) -> str:
    numeric = try_float(value)
    if numeric is None:
        return "N/A"
    absolute = abs(numeric)
    if absolute >= 1_000_000_000:
        return f"${numeric / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"${numeric / 1_000_000:,.2f}M"
    if absolute >= 1_000:
        return f"${numeric / 1_000:,.2f}K"
    return f"${numeric:,.2f}"


def format_confidence_pct(value) -> str:
    numeric = try_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric * 100:,.2f}%"


def default_fundamental_config() -> dict:
    return {
        "preset": "TradingView match",
        "apply_in_scan": True,
        "min_eps_ttm_yoy_pct": 20.0,
        "min_revenue_ttm_yoy_pct": 20.0,
        "min_roe_ttm_pct": 15.0,
        "roe_stability_max_std_dev": 40.0,
        "min_market_cap_b": 10.0,
        "min_passed_fundamental_rules": 3,
    }


def default_technical1_config() -> dict:
    return {
        "preset": "Technical-Lazy",
        "apply_in_scan": False,
        "require_close_above_all_mas": True,
        "require_bullish_ma_stack": True,
        "within_52w_high_pct": 25.0,
        "above_52w_low_pct": 25.0,
    }


def default_technical2_config() -> dict:
    return {
        "preset": "Technical-VIX",
        "apply_in_scan": False,
        "enable_alert": False,
        "wvf_lookback_stddev_high": 22,
        "wvf_bollinger_band_length": 20,
        "wvf_bollinger_band_std_dev_mult": 2.0,
        "wvf_lookback_percentile_high": 50,
        "wvf_highest_percentile": 0.85,
        "wvf_lowest_percentile": 1.01,
    }


def build_screening_config(fundamental_config: dict, technical1_config: dict, technical2_config: dict) -> dict:
    return {
        "fundamental": dict(fundamental_config),
        "technical1": dict(technical1_config),
        "technical2": dict(technical2_config),
    }


def config_signature(config: dict) -> str:
    return str(tuple(sorted((key, str(value)) for key, value in config.items())))


def get_latest_us_trading_day(reference_date: date | None = None) -> date:
    if reference_date is None:
        reference_date = datetime.now(ZoneInfo("America/New_York")).date()

    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        sessions = calendar.sessions_in_range(
            pd.Timestamp(reference_date - timedelta(days=10)),
            pd.Timestamp(reference_date),
        )
        if len(sessions) > 0:
            return sessions[-1].date()
    except Exception:
        pass

    resolved = reference_date
    while resolved.weekday() >= 5:
        resolved -= timedelta(days=1)
    return resolved


def resolve_analysis_date(selected_date: date) -> date:
    return get_latest_us_trading_day(selected_date)


@st.cache_data(ttl=86400, show_spinner=False)
def load_index_members(index_name: str) -> tuple[list[str], str]:
    source = INDEX_SOURCES[index_name]
    cache_file = source["cache_file"]
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    csv_url = source.get("csv_url")
    if csv_url:
        try:
            csv_df = pd.read_csv(csv_url)
            for column in source["candidate_columns"]:
                if column not in csv_df.columns:
                    continue
                tickers = [
                    normalize_ticker(value)
                    for value in csv_df[column].dropna().astype(str).tolist()
                ]
                tickers = list(dict.fromkeys(ticker for ticker in tickers if ticker))
                if len(tickers) >= source["min_count"]:
                    pd.DataFrame({"Ticker": tickers}).to_csv(cache_file, index=False)
                    return tickers, "live-csv"
        except Exception:
            pass

    try:
        tables = pd.read_html(source["url"])
        for table in tables:
            for column in source["candidate_columns"]:
                if column not in table.columns:
                    continue
                tickers = [
                    normalize_ticker(value)
                    for value in table[column].dropna().astype(str).tolist()
                ]
                tickers = list(dict.fromkeys(ticker for ticker in tickers if ticker))
                if len(tickers) >= source["min_count"]:
                    pd.DataFrame({"Ticker": tickers}).to_csv(cache_file, index=False)
                    return tickers, "live"
    except Exception:
        pass

    if cache_file.exists():
        cached_df = pd.read_csv(cache_file)
        tickers = [normalize_ticker(value) for value in cached_df["Ticker"].astype(str).tolist()]
        tickers = list(dict.fromkeys(ticker for ticker in tickers if ticker))
        if tickers:
            return tickers, "cache"

    return [], "unavailable"


def build_universe(
    selected_indices: list[str],
    uploaded_tickers: list[str],
    manual_tickers: list[str],
) -> tuple[list[str], list[str], list[str]]:
    universe: list[str] = []
    notes: list[str] = []
    errors: list[str] = []

    for index_name in selected_indices:
        tickers, source = load_index_members(index_name)
        if tickers:
            universe.extend(tickers)
            notes.append(f"{index_name}: {len(tickers)} symbols ({source})")
        else:
            errors.append(f"{index_name} universe could not be loaded automatically.")

    if uploaded_tickers:
        universe.extend(uploaded_tickers)
        notes.append(f"Uploaded CSV: {len(uploaded_tickers)} symbols")

    if manual_tickers:
        universe.extend(manual_tickers)
        notes.append(f"Direct entry: {len(manual_tickers)} symbols")

    return list(dict.fromkeys(universe)), notes, errors


def build_export_dataframe(
    df_results: pd.DataFrame,
    target_date: date,
    selected_universe_size: int,
    final_decisions: list[dict],
) -> pd.DataFrame:
    export_df = df_results.copy()
    export_df.insert(0, "Run Date", target_date.strftime("%Y-%m-%d"))
    export_df.insert(1, "Universe Size", selected_universe_size)
    if final_decisions:
        decisions_df = pd.DataFrame(final_decisions)
        export_df = export_df.merge(decisions_df, how="left", on="Ticker")
    return export_df


def build_research_scorecard(metrics: dict, macro_data: dict) -> dict:
    price = try_float(metrics.get("Price")) or 0.0
    adv = try_float(metrics.get("20D ADV")) or 0.0
    price_vs_200 = try_float(metrics.get("Price vs 200DMA %")) or 0.0
    ma_spread = try_float(metrics.get("50DMA vs 200DMA %")) or 0.0
    ret_3m = try_float(metrics.get("3M Return %")) or 0.0
    vol_20d = try_float(metrics.get("20D Volatility %")) or 0.0
    fcf = try_float(metrics.get("FCF")) or 0.0
    net_income = try_float(metrics.get("Net Income")) or 0.0
    regime = str(macro_data.get("regime", "Unknown"))

    fundamental_score = 0
    fundamental_reasons = []
    if fcf > 0:
        fundamental_score += 5
        fundamental_reasons.append(f"free cash flow is positive at {format_compact_currency(fcf)}")
    if net_income > 0:
        fundamental_score += 5
        fundamental_reasons.append(f"net income is positive at {format_compact_currency(net_income)}")

    technical_score = 0
    if price > 0:
        technical_score += 2
    if price_vs_200 > 0:
        technical_score += 4
    if ma_spread > 0:
        technical_score += 2
    if ret_3m > 0:
        technical_score += 2
    technical_reasons = [
        f"price is {format_number(price_vs_200, suffix='%')} above the 200DMA",
        f"50DMA is {format_number(ma_spread, suffix='%')} above the 200DMA",
        f"3M return is {format_number(ret_3m, suffix='%')}",
    ]

    liquidity_score = 0
    if adv >= 1_000_000:
        liquidity_score += 5
    if adv >= 5_000_000:
        liquidity_score += 3
    if vol_20d <= 45:
        liquidity_score += 2
    liquidity_reasons = [
        f"20D ADV is {format_number(adv, decimals=0)} shares",
        f"20D annualized volatility is {format_number(vol_20d, suffix='%')}",
    ]

    if regime == "Steady Expansion":
        macro_score = 9
    elif regime == "Inflationary Growth":
        macro_score = 7
    elif regime == "Stagflation Risk":
        macro_score = 4
    elif regime == "Recessionary Risk":
        macro_score = 3
    else:
        macro_score = 5
    macro_reasons = [f"current macro regime is {regime}"]

    criteria = [
        {
            "criterion": "Fundamental Quality",
            "score": min(10, fundamental_score),
            "reason": "; ".join(fundamental_reasons) if fundamental_reasons else "fundamental data is incomplete",
        },
        {
            "criterion": "Technical Setup",
            "score": min(10, technical_score),
            "reason": "; ".join(technical_reasons),
        },
        {
            "criterion": "Liquidity and Volatility",
            "score": min(10, liquidity_score),
            "reason": "; ".join(liquidity_reasons),
        },
        {
            "criterion": "Macro Regime Fit",
            "score": min(10, macro_score),
            "reason": "; ".join(macro_reasons),
        },
    ]

    total_score = sum(item["score"] for item in criteria)
    if total_score >= 32:
        verdict = "APPROVE"
    elif total_score >= 26:
        verdict = "WATCHLIST"
    else:
        verdict = "REJECT"

    principal_risks = []
    if vol_20d > 40:
        principal_risks.append("elevated short-term volatility")
    if ret_3m > 25:
        principal_risks.append("strong recent run may make the entry extended")
    if regime in {"Stagflation Risk", "Recessionary Risk"}:
        principal_risks.append(f"macro regime is {regime.lower()}")
    if not principal_risks:
        principal_risks.append("the trend still needs to hold above the moving averages")

    confidence = max(0.35, min(0.95, total_score / 40))
    return {
        "criteria": criteria,
        "total_score": total_score,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "principal_risks": principal_risks,
    }


def build_execution_plan(metrics: dict, scorecard: dict) -> dict:
    price = try_float(metrics.get("Price")) or 0.0
    dma_50 = try_float(metrics.get("50 DMA")) or price
    dma_200 = try_float(metrics.get("200 DMA")) or price
    total_score = scorecard["total_score"]

    if total_score >= 32:
        action = "BUY"
    elif total_score >= 26:
        action = "HOLD"
    else:
        action = "AVOID"

    lower_entry = min(price, max(dma_50, price * 0.97))
    upper_entry = max(price, dma_50)
    stop_loss = min(dma_200 * 0.98, price * 0.92) if price else None
    risk_distance_pct = (((price - stop_loss) / price) * 100.0) if price and stop_loss else None

    return {
        "action": action,
        "entry_zone": f"${lower_entry:,.2f} to ${upper_entry:,.2f}" if price else "N/A",
        "entry_logic": "Scale in near the current price / 50DMA support band and avoid chasing a breakout far above trend support.",
        "stop_loss": f"${stop_loss:,.2f}" if stop_loss else "N/A",
        "risk_distance_pct": round(risk_distance_pct, 2) if risk_distance_pct is not None else None,
        "time_horizon": "4-12 weeks",
    }


def build_risk_plan(metrics: dict, macro_data: dict, scorecard: dict, execution_plan: dict) -> dict:
    total_score = scorecard["total_score"]
    vol_20d = try_float(metrics.get("20D Volatility %")) or 0.0
    regime = str(macro_data.get("regime", "Unknown"))

    if execution_plan["action"] == "BUY" and total_score >= 34 and vol_20d <= 35 and regime in {"Steady Expansion", "Inflationary Growth"}:
        capital_allocation = "10% of total capital"
        approval = "APPROVED"
        reason = "Score is strong, volatility is still manageable, and the macro regime is supportive."
    elif execution_plan["action"] == "BUY":
        capital_allocation = "5% of total capital"
        approval = "APPROVED"
        reason = "The setup passed, but sizing is capped because either volatility or macro conditions argue for a smaller starter position."
    elif execution_plan["action"] == "HOLD":
        capital_allocation = "0% new capital"
        approval = "HOLD"
        reason = "The name is worth monitoring, but the scorecard does not justify a fresh risk allocation yet."
    else:
        capital_allocation = "0% of total capital"
        approval = "VETO"
        reason = "The setup did not clear the score threshold for a new position."

    primary_risk = "; ".join(scorecard.get("principal_risks", []))
    return {
        "approval": approval,
        "capital_allocation": capital_allocation,
        "allocation_reason": reason,
        "primary_risk": primary_risk,
    }


def build_screen_state(
    target_date: date,
    effective_date: date,
    universe: list[str],
    notes: list[str],
    macro_data: dict,
    survivors: list[str],
    df_results: pd.DataFrame,
    screening_config: dict,
) -> dict:
    return {
        "target_date": target_date,
        "effective_date": effective_date,
        "universe": universe,
        "universe_notes": notes,
        "macro_data": macro_data,
        "survivors": survivors,
        "df_results": df_results,
        "screening_config": screening_config,
    }


def get_display_dataframe(df_results: pd.DataFrame) -> pd.DataFrame:
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    display_df = df_results.copy()
    currency_columns = ["Price", "50 DMA", "150 DMA", "200 DMA", "52W High", "52W Low"]
    volume_columns = ["20D ADV"]
    percent_columns = [
        "Price vs 50DMA %",
        "Price vs 150DMA %",
        "Price vs 200DMA %",
        "50DMA vs 200DMA %",
        "Within 52W High %",
        "Above 52W Low %",
        "1M Return %",
        "3M Return %",
        "6M Return %",
        "20D Volatility %",
        "EPS TTM YoY %",
        "Revenue TTM YoY %",
        "ROE TTM %",
        "ROE Stability Std Dev",
        "WVF Value",
    ]
    fundamental_columns = ["FCF", "Net Income", "Operating CF"]
    market_cap_columns = ["Market Cap (B USD)"]

    for column in currency_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, prefix="$"))
    for column in volume_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, decimals=2))
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, suffix="%"))
    for column in fundamental_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_compact_currency)
    for column in market_cap_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_number)

    return display_df


def parse_trader_plan(plan_value) -> dict:
    if isinstance(plan_value, dict):
        return plan_value
    if not plan_value:
        return {}

    try:
        parsed = ast.literal_eval(str(plan_value))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {"raw_plan": str(plan_value)}


def get_agent_label(agent_models: dict, agent_key: str) -> str:
    metadata = (agent_models or {}).get(agent_key, {})
    model = metadata.get("model", "unknown")
    provider = metadata.get("provider", "unknown")
    runtime_mode = metadata.get("runtime_mode", "unknown")
    return f"{provider} / {model} ({runtime_mode})"


def render_agent_model_badge(agent_models: dict, agent_key: str) -> None:
    st.caption(f"Model: {get_agent_label(agent_models, agent_key)}")


def render_process_audit(final_state: dict) -> None:
    analysis_protocol = final_state.get("analysis_protocol", {})
    agent_models = final_state.get("agent_models", {})
    max_rounds = analysis_protocol.get("max_debate_rounds", final_state.get("debate_round", 0))

    st.markdown("#### Process Audit")
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Analysis Date", str(final_state.get("analysis_date", "N/A")))
    a2.metric("Debate Rounds", format_number(final_state.get("debate_round"), decimals=2))
    a3.metric("Bull Model", get_agent_label(agent_models, "bull_researcher"))
    a4.metric("Bear Model", get_agent_label(agent_models, "bear_researcher"))
    a5.metric("Audit Status", final_state.get("audit_status", "N/A"))

    st.info(
        f"Evidence basis: {analysis_protocol.get('analysis_basis', 'N/A')}. "
        f"Debate method: {analysis_protocol.get('debate_style', 'N/A')}. "
        f"Configured max rounds: {format_number(max_rounds, decimals=2)}."
    )
    st.caption(analysis_protocol.get("independence_note", ""))


def render_agent_brief(title: str, content: str, agent_models: dict, agent_key: str, description: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(description)
    render_agent_model_badge(agent_models, agent_key)
    st.markdown(content)


def render_research_manager_section(transcript: dict, scorecard: dict, agent_models: dict) -> None:
    st.markdown("#### Research Manager")
    st.caption("This manager converts the bull/bear evidence into a weighted scorecard and a go / watchlist / reject ruling.")
    render_agent_model_badge(agent_models, "research_manager")

    summary1, summary2, summary3 = st.columns(3)
    summary1.metric("Score", f"{format_number(scorecard.get('total_score'))}/40.00")
    summary2.metric("Verdict", scorecard.get("verdict", "N/A"))
    summary3.metric("Confidence", format_confidence_pct(scorecard.get("confidence")))

    st.markdown(transcript.get("research_manager_ruling", "No ruling generated."))


def render_risk_manager_section(risk_plan: dict, agent_models: dict) -> None:
    st.markdown("#### Risk Manager")
    st.caption("This manager explains how sizing is approved, reduced, or vetoed from the draft execution plan.")
    render_agent_model_badge(agent_models, "risk_manager")

    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Approval", risk_plan.get("approval", "N/A"))
    r2.metric("Capital Allocation", risk_plan.get("capital_allocation", "N/A"))
    r3.metric("Primary Risk", risk_plan.get("primary_risk", "N/A"))

    st.info(risk_plan.get("allocation_reason", ""))
    st.markdown(risk_plan.get("risk_ruling", "No risk ruling generated."))


def render_audit_manager_section(final_state: dict, agent_models: dict) -> None:
    transcript = final_state.get("debate_transcript", {})
    audit_notes = final_state.get("audit_notes", []) or []

    st.markdown("#### Audit Manager")
    st.caption("This final control verifies the date boundary, required evidence, and whether the final decision is consistent with the score and risk outputs.")
    render_agent_model_badge(agent_models, "audit_manager")

    a1, a2 = st.columns(2)
    a1.metric("Audit Status", final_state.get("audit_status", "N/A"))
    a2.metric("Log Entries", format_number(len(audit_notes)))

    st.markdown(transcript.get("audit_manager_ruling", "No audit ruling generated."))

    if audit_notes:
        audit_df = pd.DataFrame({"Verification Log": audit_notes})
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


def render_reference_page() -> None:
    render_hero(st.session_state.screen_state)
    render_section_header(
        "Reference",
        "System Workflow and Control Logic",
        "This page explains how the screen, scoring model, and rate-limit protection work so the investor can audit the system design.",
    )

    st.subheader("Current Flow")
    st.markdown(
        """
1. Choose the analysis date plus the universe source (`SP500`, `Nasdaq 100`, both, uploaded CSV, and/or direct tickers).
2. Click `Run AI Swarm` to run the macro pass and the point-in-time hard screen.
3. Review the Step 2 qualified list and click a ticker to trigger Step 3 analysis.
4. Inspect the factual analyst notes, the research manager scorecard, the execution plan, and the portfolio manager risk allocation.
"""
    )

    st.subheader("Scoring")
    st.markdown(
        """
- `Fundamental Quality`: positive free cash flow and positive net income.
- `Technical Setup`: price vs 200DMA, 50DMA vs 200DMA, and medium-term return profile.
- `Liquidity and Volatility`: 20-day average daily volume and recent realized volatility.
- `Macro Regime Fit`: whether the current FRED regime is supportive or defensive.
"""
    )

    st.subheader("Rate-Limit Handling")
    st.markdown(
        """
- Price history now prefers local parquet cache before live Yahoo calls.
- Batch price downloads reduce the number of Yahoo requests for large universes.
- Fundamental snapshots are cached locally after the first successful fetch.
- If Yahoo or Alpha Vantage fails for a ticker, the screen marks that ticker as failed instead of crashing the entire app.
"""
    )

    st.subheader("API Diagnostics")
    st.caption("Confirms whether the current deployment has working Grok, FRED, and Alpha Vantage credentials.")
    render_api_diagnostics()

    st.subheader("Runtime Log")
    if st.session_state.app_log:
        st.code("\n".join(st.session_state.app_log), language="text")
    else:
        st.info("No runtime events yet.")


def render_macro_section(macro_data: dict) -> None:
    render_section_header(
        "Step 1",
        "Top-Down Macro Regime",
        "Macro context frames the opportunity set before any bottom-up stock selection is allowed to proceed.",
    )
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Current Regime", macro_data.get("regime", "N/A"))
    m_col2.metric("YoY Inflation (CPI)", f"{macro_data.get('cpi', 'N/A')}%")
    m_col3.metric("Unemployment Rate", f"{macro_data.get('unemployment', 'N/A')}%")


def render_step2_section(screen_state: dict) -> None:
    df_results = screen_state["df_results"]
    survivors = screen_state["survivors"]
    effective_date = screen_state["effective_date"]
    screening_config = screen_state.get("screening_config", {})
    fundamental_config = screening_config.get("fundamental", {})
    technical1_config = screening_config.get("technical1", {})
    technical2_config = screening_config.get("technical2", {})

    render_section_header(
        "Step 2",
        f"Point-In-Time Pre-Screening (As of {effective_date})",
        "This screen evaluates independent Fundamental, Technical 1, and Technical 2 modules using only data available as of the selected date.",
    )
    if screen_state["universe_notes"]:
        st.caption("Universe sources: " + " | ".join(screen_state["universe_notes"]))

    st.markdown(
        f"Fundamental applied: {'Yes' if fundamental_config.get('apply_in_scan') else 'No'} | "
        f"Technical 1 applied: {'Yes' if technical1_config.get('apply_in_scan') else 'No'} | "
        f"Technical 2 applied: {'Yes' if technical2_config.get('apply_in_scan') else 'No'} | "
        f"WVF alert enabled: {'Yes' if technical2_config.get('enable_alert') else 'No'}"
    )
    st.dataframe(get_display_dataframe(df_results), use_container_width=True, hide_index=True)

    audit_columns = [
        "Ticker",
        "Reason",
        "Fundamental Source",
        "Fundamental Pass",
        "Fundamental Rules Passed",
        "Fundamental Detail",
        "Technical1 Detail",
        "Technical2 Detail",
    ]
    available_audit_columns = [column for column in audit_columns if column in df_results.columns]
    if available_audit_columns:
        with st.expander("Step 2 Audit Detail", expanded=not survivors):
            audit_df = df_results[available_audit_columns].copy()
            st.dataframe(audit_df, use_container_width=True, hide_index=True)

    st.success(f"Qualified for further research: {len(survivors)} out of {len(screen_state['universe'])}.")

    if not survivors:
        st.warning("No names qualified for Step 3. Review the failed reasons in Step 2 or try a narrower universe.")


def render_survivor_picker(screen_state: dict) -> None:
    survivors = screen_state["survivors"]
    if not survivors:
        return

    render_section_header(
        "Step 3",
        "Click a Qualified Ticker",
        "The system now waits for the user to choose where to spend research time. Click any qualified name below to launch the detailed analysis.",
    )

    survivor_df = get_display_dataframe(screen_state["df_results"])
    if "Ticker" not in survivor_df.columns:
        st.warning("Qualified tickers were found, but the Step 2 summary table is missing ticker labels.")
        return

    survivor_df = survivor_df[survivor_df["Ticker"].isin(survivors)].copy()
    summary_columns = ["Ticker", "Price", "3M Return %", "Price vs 200DMA %", "FCF", "Net Income"]
    for column in summary_columns:
        if column not in survivor_df.columns:
            survivor_df[column] = "N/A"
    survivor_df = survivor_df[summary_columns]
    selection = st.dataframe(
        survivor_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="survivor_table",
    )
    st.caption(f"Click a ticker row to launch the multi-agent review using only data available as of {screen_state['effective_date']}.")

    selected_rows = getattr(getattr(selection, "selection", None), "rows", [])
    if selected_rows:
        selected_ticker = survivor_df.iloc[selected_rows[0]]["Ticker"]
        if st.session_state.selected_ticker != selected_ticker:
            st.session_state.selected_ticker = selected_ticker
            append_log(f"User selected {selected_ticker} for Step 3 analysis.")


def render_fact_sheet(metrics: dict) -> None:
    st.markdown("#### Historical Data Fact Sheet")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", format_number(metrics.get("Price"), prefix="$"))
    c2.metric("20D ADV", format_number(metrics.get("20D ADV"), decimals=0))
    c3.metric("50DMA", format_number(metrics.get("50 DMA"), prefix="$"))
    c4.metric("200DMA", format_number(metrics.get("200 DMA"), prefix="$"))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("1M Return", format_number(metrics.get("1M Return %"), suffix="%"))
    c6.metric("3M Return", format_number(metrics.get("3M Return %"), suffix="%"))
    c7.metric("6M Return", format_number(metrics.get("6M Return %"), suffix="%"))
    c8.metric("20D Volatility", format_number(metrics.get("20D Volatility %"), suffix="%"))

    st.caption(
        f"Price data end date: {metrics.get('Price Data End', 'N/A')} | "
        f"Price source: {metrics.get('Price Source', 'N/A')} | "
        f"Fundamental source: {metrics.get('Fundamental Source', 'N/A')}"
    )


def render_scorecard(scorecard: dict) -> None:
    st.markdown("#### Scorecard")
    criteria = scorecard.get("criteria") or []
    if criteria:
        score_df = pd.DataFrame(criteria)
        if "score" in score_df.columns:
            score_df["score"] = score_df["score"].apply(lambda value: format_number(value))
        st.dataframe(score_df, use_container_width=True, hide_index=True)
    else:
        st.info("Scorecard criteria are not available for this run, so the view is showing the fallback summary only.")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Score", f"{format_number(scorecard.get('total_score'))}/40.00")
    s2.metric("Verdict", scorecard.get("verdict", "N/A"))
    s3.metric("Confidence", format_confidence_pct(scorecard.get("confidence")))


def render_execution_and_risk(execution_plan: dict, risk_plan: dict) -> None:
    st.markdown("#### Execution Plan Details")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Action", execution_plan.get("action", "N/A"))
    e2.metric("Entry Zone", execution_plan.get("entry_zone", "N/A"))
    e3.metric("Stop Loss", execution_plan.get("stop_loss", "N/A"))
    e4.metric("Time Horizon", execution_plan.get("time_horizon", "N/A"))
    st.write(execution_plan.get("entry_logic", ""))
    risk_distance = execution_plan.get("risk_distance_pct")
    if risk_distance is not None:
        st.caption(f"Stop distance from current price: {format_number(risk_distance, suffix='%')}")


def render_decision_view(ticker: str, final_state: dict, metrics: dict) -> None:
    transcript = final_state.get("debate_transcript", {})
    scorecard = final_state.get("research_scorecard", {})
    execution_plan = final_state.get("execution_plan", {})
    risk_plan = final_state.get("risk_assessment") or final_state.get("risk_plan", {})
    agent_models = final_state.get("agent_models", {})
    trader_plan = parse_trader_plan(final_state.get("trader_plan", {}).get("plan_details"))
    if trader_plan and "action" in trader_plan:
        execution_plan = {**execution_plan, **trader_plan}

    render_section_header(
        "Step 3 Analysis",
        f"{ticker} Research Workspace",
        "The views below expose how the system thinks: the factual analyst notes, the scorecard logic, the execution framework, and the final risk-aware portfolio recommendation.",
    )
    render_fact_sheet(metrics)
    render_process_audit(final_state)
    render_scorecard(scorecard)

    col1, col2 = st.columns(2)
    with col1:
        render_agent_brief(
            "Fundamental Analyst",
            final_state.get("analyst_reports", {}).get("fundamental", "No report generated."),
            agent_models,
            "fundamental_analyst",
            "What: point-in-time fundamentals. Why: validate profitability. How: use only published metrics available on the analysis date.",
        )
        render_agent_brief(
            "Bull Case",
            transcript.get("bull", "No bull case generated."),
            agent_models,
            "bull_researcher",
            "What: strongest long thesis. Why: identify upside drivers. How: argue from the same historical dataset without future data.",
        )
    with col2:
        render_agent_brief(
            "Technical Analyst",
            final_state.get("analyst_reports", {}).get("technical", "No report generated."),
            agent_models,
            "technical_analyst",
            "What: trend, momentum, liquidity, and volatility. Why: validate the tape. How: use only historical market structure as of the selected date.",
        )
        render_agent_brief(
            "Bear Case",
            transcript.get("bear", "No bear case generated."),
            agent_models,
            "bear_researcher",
            "What: strongest downside thesis. Why: surface hidden risks. How: challenge the bull case using the same historical evidence only.",
        )

    render_research_manager_section(transcript, scorecard, agent_models)

    render_execution_and_risk(execution_plan, risk_plan)
    render_risk_manager_section(risk_plan, agent_models)
    render_audit_manager_section(final_state, agent_models)

    st.markdown("#### Portfolio Manager Final Decision")
    d1, d2 = st.columns(2)
    d1.metric("Final Action", final_state.get("final_decision", "UNKNOWN"))
    d2.metric("Confidence", format_confidence_pct(final_state.get("confidence", 0)))

    st.markdown("#### Detailed Thesis")
    st.markdown(str(final_state.get("pm_reasoning", "No portfolio manager reasoning generated.")))


def build_initial_state(ticker: str, analysis_date: date, metrics: dict, macro_data: dict) -> dict:
    scorecard = build_research_scorecard(metrics, macro_data)
    execution_plan = build_execution_plan(metrics, scorecard)
    risk_plan = build_risk_plan(metrics, macro_data, scorecard, execution_plan)

    return {
        "ticker": ticker,
        "analysis_date": str(analysis_date),
        "market_metrics": metrics,
        "macro_context": macro_data,
        "research_scorecard": scorecard,
        "execution_plan": execution_plan,
        "risk_plan": risk_plan,
        "analyst_reports": {},
        "debate_transcript": {},
        "debate_round": 0,
        "trader_plan": {},
        "risk_assessment": {},
        "final_decision": "",
        "confidence": 0.0,
        "pm_reasoning": "",
        "audit_status": "PENDING",
        "audit_notes": [],
        "agent_models": {},
        "analysis_protocol": {},
    }


def get_agent_setup_metadata(agent_graph) -> dict:
    describe = getattr(agent_graph, "describe_agent_setup", None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            pass

    return {
        "fundamental_analyst": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "technical_analyst": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "bull_researcher": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "bear_researcher": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "research_manager": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "trader_agent": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "risk_manager": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "portfolio_manager": {"provider": "unknown", "model": "unknown", "runtime_mode": "unknown"},
        "audit_manager": {"provider": "system", "model": "fallback", "runtime_mode": "deterministic-validation"},
    }


def get_analysis_protocol_metadata(agent_graph) -> dict:
    describe = getattr(agent_graph, "describe_analysis_protocol", None)
    if callable(describe):
        try:
            return describe()
        except Exception:
            pass

    return {
        "analysis_basis": "point-in-time-only",
        "evidence_scope": ["market_metrics", "macro_context", "research_scorecard"],
        "debate_style": "compatibility-fallback",
        "max_debate_rounds": 0,
        "independence_note": "Compatibility fallback metadata was used because the graph object does not expose protocol details.",
    }


def render_selected_analysis(screen_state: dict) -> list[dict]:
    survivors = screen_state["survivors"]
    ticker = st.session_state.selected_ticker
    final_decisions: list[dict] = []

    if not ticker or ticker not in survivors:
        return final_decisions

    df_results = screen_state["df_results"]
    row = df_results[df_results["Ticker"] == ticker]
    if row.empty:
        st.warning(f"No Step 2 metrics found for {ticker}.")
        return final_decisions

    metrics = row.iloc[0].to_dict()
    cache_key = f"{screen_state['effective_date']}::{ticker}"
    final_state = st.session_state.analysis_cache.get(cache_key)

    if final_state is None:
        if ticker not in st.session_state.analysis_order:
            st.session_state.analysis_order.append(ticker)

        if not st.session_state.get("agent_graph"):
            st.session_state.agent_graph = HedgeFundGraph()

        if not (st.session_state.agent_graph.tier1_llm and st.session_state.agent_graph.tier2_llm):
            st.info("Live LLM credentials are not available in this runtime, so Step 3 is using deterministic analysis built from actual historical and fundamental data.")

        append_log(f"Running Step 3 analysis for {ticker}.")
        initial_state = build_initial_state(ticker, screen_state["effective_date"], metrics, screen_state["macro_data"])
        initial_state["agent_models"] = get_agent_setup_metadata(st.session_state.agent_graph)
        initial_state["analysis_protocol"] = get_analysis_protocol_metadata(st.session_state.agent_graph)
        initial_state["audit_notes"] = [
            f"Historical guardrail active for analysis date {screen_state['effective_date']}.",
            "Bull and Bear researchers are independent nodes and may only use point-in-time inputs supplied by the app.",
        ]
        with st.spinner(f"Running multi-agent analysis for {ticker}..."):
            final_state = st.session_state.agent_graph.run_analysis(initial_state)

        st.session_state.analysis_cache[cache_key] = final_state
        append_log(
            f"{ticker} decision complete: {final_state.get('final_decision', 'UNKNOWN')} at "
            f"{final_state.get('confidence', 0) * 100:.0f}% confidence."
        )

    render_decision_view(ticker, final_state, metrics)

    final_decisions.append(
        {
            "Ticker": ticker,
            "Final Decision": final_state.get("final_decision", "UNKNOWN"),
            "Confidence": round(final_state.get("confidence", 0.0), 2),
            "Audit Status": final_state.get("audit_status", "N/A"),
            "Risk Approval": (final_state.get("risk_assessment") or {}).get("approval", "N/A"),
            "Capital Allocation": (final_state.get("risk_assessment") or {}).get("capital_allocation", "N/A"),
        }
    )

    return final_decisions


def collect_cached_final_decisions(screen_state: dict) -> list[dict]:
    decisions: list[dict] = []
    for ticker in st.session_state.analysis_order:
        cache_key = f"{screen_state['effective_date']}::{ticker}"
        final_state = st.session_state.analysis_cache.get(cache_key)
        if not final_state:
            continue
        decisions.append(
            {
                "Ticker": ticker,
                "Final Decision": final_state.get("final_decision", "UNKNOWN"),
                "Confidence": round(final_state.get("confidence", 0.0), 2),
                "Audit Status": final_state.get("audit_status", "N/A"),
                "Risk Approval": (final_state.get("risk_assessment") or {}).get("approval", "N/A"),
                "Capital Allocation": (final_state.get("risk_assessment") or {}).get("capital_allocation", "N/A"),
            }
        )
    return decisions


def render_analysis_page() -> None:
    default_date = get_latest_us_trading_day()
    if not st.session_state.fundamental_filter_config:
        st.session_state.fundamental_filter_config = default_fundamental_config()
    if not st.session_state.technical1_filter_config:
        st.session_state.technical1_filter_config = default_technical1_config()
    if not st.session_state.technical2_filter_config:
        st.session_state.technical2_filter_config = default_technical2_config()

    st.sidebar.markdown('<div class="side-label">Step 1</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Choose Universe**")
    universe_choice = st.sidebar.selectbox("Universe Source", options=list(UNIVERSE_CHOICES.keys()), index=0)
    selected_indices = UNIVERSE_CHOICES[universe_choice]
    uploaded_universe = st.sidebar.file_uploader("Upload CSV Universe", type=["csv"])
    tickers_input = st.sidebar.text_area(
        "Direct Tickers",
        value="",
        height=90,
        help="Enter one ticker or multiple tickers with commas, spaces, or new lines.",
        placeholder="AAPL, MSFT, NVDA",
    )

    uploaded_tickers = load_uploaded_tickers(uploaded_universe)
    manual_tickers = parse_direct_tickers(tickers_input)
    universe_ready = bool(selected_indices or uploaded_tickers or manual_tickers)

    st.sidebar.markdown('<div class="side-label">Step 2</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Screening Setup**")

    with st.sidebar.expander("Fundamental Setup", expanded=True):
        st.caption("Matches TradingView-style core rules. The 3 growth / quality rules are counted independently.")
        fundamental_preset = st.selectbox("Preset", ["TradingView match"], key="fundamental_preset")
        fundamental_apply = st.checkbox(
            "Require fundamentals for pass",
            value=st.session_state.fundamental_filter_config.get("apply_in_scan", True),
            key="fundamental_apply",
        )
        min_eps = st.number_input("Min EPS TTM YoY (%)", value=float(st.session_state.fundamental_filter_config.get("min_eps_ttm_yoy_pct", 20.0)), step=1.0, format="%.2f")
        min_revenue = st.number_input("Min Revenue TTM YoY (%)", value=float(st.session_state.fundamental_filter_config.get("min_revenue_ttm_yoy_pct", 20.0)), step=1.0, format="%.2f")
        min_roe = st.number_input("Min ROE TTM (%)", value=float(st.session_state.fundamental_filter_config.get("min_roe_ttm_pct", 15.0)), step=1.0, format="%.2f")
        max_roe_std = st.number_input("ROE Stability Max Std-Dev", value=float(st.session_state.fundamental_filter_config.get("roe_stability_max_std_dev", 40.0)), step=1.0, format="%.2f")
        min_market_cap_b = st.number_input("Min Market Cap (B USD)", value=float(st.session_state.fundamental_filter_config.get("min_market_cap_b", 10.0)), step=1.0, format="%.2f")
        min_fundamental_rules = st.number_input("Min Passed Fundamental Rules (Out of 3)", min_value=1, max_value=3, value=int(st.session_state.fundamental_filter_config.get("min_passed_fundamental_rules", 3)), step=1)
        st.session_state.fundamental_filter_config = {
            "preset": fundamental_preset,
            "apply_in_scan": fundamental_apply,
            "min_eps_ttm_yoy_pct": float(min_eps),
            "min_revenue_ttm_yoy_pct": float(min_revenue),
            "min_roe_ttm_pct": float(min_roe),
            "roe_stability_max_std_dev": float(max_roe_std),
            "min_market_cap_b": float(min_market_cap_b),
            "min_passed_fundamental_rules": int(min_fundamental_rules),
        }

    with st.sidebar.expander("Technical Indicator 1", expanded=False):
        st.caption("MA checks use MA50, MA150, and MA200. This module works independently from fundamentals.")
        technical1_preset = st.selectbox("Preset", ["Technical-Lazy"], key="technical1_preset")
        technical1_apply = st.checkbox(
            "Apply technical-lazy filter in scan",
            value=st.session_state.technical1_filter_config.get("apply_in_scan", False),
            key="technical1_apply",
        )
        require_close_above = st.checkbox(
            "Require Close > MA50, MA150, MA200",
            value=st.session_state.technical1_filter_config.get("require_close_above_all_mas", True),
            key="technical1_close_above",
        )
        require_stack = st.checkbox(
            "Require MA50 > MA150 > MA200",
            value=st.session_state.technical1_filter_config.get("require_bullish_ma_stack", True),
            key="technical1_ma_stack",
        )
        within_high = st.number_input("Within 52W High (%)", value=float(st.session_state.technical1_filter_config.get("within_52w_high_pct", 25.0)), step=1.0, format="%.2f")
        above_low = st.number_input("Above 52W Low (%)", value=float(st.session_state.technical1_filter_config.get("above_52w_low_pct", 25.0)), step=1.0, format="%.2f")
        st.session_state.technical1_filter_config = {
            "preset": technical1_preset,
            "apply_in_scan": technical1_apply,
            "require_close_above_all_mas": require_close_above,
            "require_bullish_ma_stack": require_stack,
            "within_52w_high_pct": float(within_high),
            "above_52w_low_pct": float(above_low),
        }

    with st.sidebar.expander("Technical Indicator 2", expanded=False):
        st.caption("Williams VIX Fix runs independently. You can keep it as an alert or also apply it in the scan.")
        technical2_preset = st.selectbox("Preset", ["Technical-VIX"], key="technical2_preset")
        technical2_apply = st.checkbox(
            "Apply technical-vix filter in scan",
            value=st.session_state.technical2_filter_config.get("apply_in_scan", False),
            key="technical2_apply",
        )
        technical2_alert = st.checkbox(
            "Enable Williams VIX Fix alert",
            value=st.session_state.technical2_filter_config.get("enable_alert", False),
            key="technical2_alert",
        )
        wvf_lookback = st.number_input("WVF Lookback Period Standard Deviation High", value=int(st.session_state.technical2_filter_config.get("wvf_lookback_stddev_high", 22)), step=1)
        wvf_bb_length = st.number_input("WVF Bollinger Band Length", value=int(st.session_state.technical2_filter_config.get("wvf_bollinger_band_length", 20)), step=1)
        wvf_bb_mult = st.number_input("WVF Bollinger Band Std Dev Mult", value=float(st.session_state.technical2_filter_config.get("wvf_bollinger_band_std_dev_mult", 2.0)), step=0.1, format="%.2f")
        wvf_pct_lookback = st.number_input("WVF Look Back Period Percentile High", value=int(st.session_state.technical2_filter_config.get("wvf_lookback_percentile_high", 50)), step=1)
        wvf_high_pct = st.number_input("WVF Highest Percentile", value=float(st.session_state.technical2_filter_config.get("wvf_highest_percentile", 0.85)), step=0.01, format="%.2f")
        wvf_low_pct = st.number_input("WVF Lowest Percentile", value=float(st.session_state.technical2_filter_config.get("wvf_lowest_percentile", 1.01)), step=0.01, format="%.2f")
        st.session_state.technical2_filter_config = {
            "preset": technical2_preset,
            "apply_in_scan": technical2_apply,
            "enable_alert": technical2_alert,
            "wvf_lookback_stddev_high": int(wvf_lookback),
            "wvf_bollinger_band_length": int(wvf_bb_length),
            "wvf_bollinger_band_std_dev_mult": float(wvf_bb_mult),
            "wvf_lookback_percentile_high": int(wvf_pct_lookback),
            "wvf_highest_percentile": float(wvf_high_pct),
            "wvf_lowest_percentile": float(wvf_low_pct),
        }

    screening_config = build_screening_config(
        st.session_state.fundamental_filter_config,
        st.session_state.technical1_filter_config,
        st.session_state.technical2_filter_config,
    )
    screening_ready = True
    st.sidebar.caption(
        f"Fundamental: {'required' if screening_config['fundamental']['apply_in_scan'] else 'optional'} | "
        f"Tech 1: {'applied' if screening_config['technical1']['apply_in_scan'] else 'not applied'} | "
        f"Tech 2: {'applied' if screening_config['technical2']['apply_in_scan'] else 'not applied'}"
    )

    st.sidebar.markdown('<div class="side-label">Step 3</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Analysis Date**")
    target_date = st.sidebar.date_input("Analysis Date", value=default_date)
    effective_date = resolve_analysis_date(target_date)
    date_ready = bool(effective_date)

    if effective_date != target_date:
        st.sidebar.info(f"Selected date is not a US trading day. The app will use {effective_date}.")

    st.sidebar.markdown('<div class="side-label">Step 4</div>', unsafe_allow_html=True)
    st.sidebar.markdown("**Run Screen**")
    run_analysis = st.sidebar.button("Run AI Swarm", use_container_width=True)

    st.sidebar.markdown('<div class="side-label">Workflow Progress</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        f'<div class="side-note">Step 1 Universe: {"Done" if universe_ready else "Pending"}<br>'
        f'Step 2 Screening Setup: {"Done" if screening_ready else "Pending"}<br>'
        f'Step 3 Analysis Date: {"Done" if date_ready else "Pending"}<br>'
        f'Step 4 Run Screen: {"Ready" if universe_ready and screening_ready and date_ready else "Locked"}</div>',
        unsafe_allow_html=True,
    )

    signature = config_signature(screening_config)
    if st.session_state.filter_signature and st.session_state.filter_signature != signature and st.session_state.screen_state:
        st.session_state.screen_state = None
        st.session_state.selected_ticker = None
        st.session_state.analysis_cache = {}
        st.session_state.analysis_order = []
        st.session_state.last_screen_results = None
        st.session_state.last_survivors = []
        st.session_state.last_export_df = None
        append_log("Step 2 screening setup changed. Cached screen results were cleared to avoid stale output.")
    st.session_state.filter_signature = signature

    if run_analysis:
        universe, notes, errors = build_universe(selected_indices, uploaded_tickers, manual_tickers)
        if errors:
            for error in errors:
                st.sidebar.warning(error)

        if not universe:
            st.error("No valid universe was provided. Choose SP500, Nasdaq 100, upload a CSV, or enter tickers directly.")
            return

        append_log(f"Analysis started for {len(universe)} tickers as of {effective_date}.")
        with st.spinner("Fetching macro context..."):
            macro_loader = MacroLoader()
            macro_data = macro_loader.get_macro_regime()
        append_log(f"Macro regime loaded: {macro_data.get('regime', 'N/A')}.")

        with st.spinner("Running point-in-time hard screen..."):
            pit_context = PointInTimeContext(effective_date)
            price_loader = PriceLoader()
            fundamental_loader = FundamentalLoader(fallback_lag_days=45)
            try:
                screener = HardScreener(price_loader, fundamental_loader, screening_config)
            except TypeError as exc:
                # Compatibility fallback for environments still loading an older
                # HardScreener constructor that only accepts the two loader args.
                append_log(f"HardScreener constructor fallback engaged: {exc}")
                screener = HardScreener(price_loader, fundamental_loader)
                setattr(screener, "screen_config", screening_config)
            survivors, df_results = screener.run_screen(universe, pit_context)

        st.session_state.screen_state = build_screen_state(
            target_date=target_date,
            effective_date=effective_date,
            universe=universe,
            notes=notes,
            macro_data=macro_data,
            survivors=survivors,
            df_results=df_results,
            screening_config=screening_config,
        )
        st.session_state.last_screen_results = df_results
        st.session_state.last_survivors = survivors
        st.session_state.selected_ticker = None
        st.session_state.analysis_cache = {}
        st.session_state.analysis_order = []
        append_log(f"Pre-screen complete: {len(survivors)} out of {len(universe)} tickers survived.")

    screen_state = st.session_state.screen_state
    render_hero(screen_state)
    if not screen_state:
        render_section_header(
            "Ready",
            "Build the Universe and Run the Screen",
            "Use the left control panel to choose the universe source, add optional custom tickers, and launch the institutional pre-screen. The main workspace will update here once the run completes.",
        )
        st.markdown(
            '<div class="mini-note">Suggested starting setup: use <strong>S&amp;P 500 + Nasdaq 100</strong>, keep the default screen date, and click <strong>Run AI Swarm</strong>. After Step 2 completes, click a qualified ticker to inspect the full research workflow.</div>',
            unsafe_allow_html=True,
        )
        return

    render_macro_section(screen_state["macro_data"])
    render_step2_section(screen_state)
    render_survivor_picker(screen_state)

    render_selected_analysis(screen_state)
    final_decisions = collect_cached_final_decisions(screen_state)
    st.session_state.last_export_df = build_export_dataframe(
        screen_state["df_results"],
        screen_state["effective_date"],
        len(screen_state["universe"]),
        final_decisions,
    )

    st.download_button(
        "Export Latest Result (.csv)",
        data=st.session_state.last_export_df.to_csv(index=False).encode("utf-8"),
        file_name=build_export_filename(screen_state["effective_date"]),
        mime="text/csv",
    )


def main() -> None:
    init_session_state()
    inject_app_styles()
    st.sidebar.markdown('<div class="side-label">Navigation</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Go To", ["Analysis", "Reference"], index=0, label_visibility="collapsed")

    if page == "Reference":
        render_reference_page()
    else:
        render_analysis_page()


if __name__ == "__main__":
    main()
