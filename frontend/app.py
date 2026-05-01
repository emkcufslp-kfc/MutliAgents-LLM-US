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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.app_log.insert(0, f"[{timestamp}] {message}")
    st.session_state.app_log = st.session_state.app_log[:100]


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
) -> dict:
    return {
        "target_date": target_date,
        "effective_date": effective_date,
        "universe": universe,
        "universe_notes": notes,
        "macro_data": macro_data,
        "survivors": survivors,
        "df_results": df_results,
    }


def get_display_dataframe(df_results: pd.DataFrame) -> pd.DataFrame:
    if df_results is None or df_results.empty:
        return pd.DataFrame()

    display_df = df_results.copy()
    currency_columns = ["Price", "50 DMA", "200 DMA"]
    volume_columns = ["20D ADV"]
    percent_columns = ["Price vs 50DMA %", "Price vs 200DMA %", "50DMA vs 200DMA %", "1M Return %", "3M Return %", "6M Return %", "20D Volatility %"]
    fundamental_columns = ["FCF", "Net Income", "Operating CF"]

    for column in currency_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, prefix="$"))
    for column in volume_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, decimals=0))
    for column in percent_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(lambda value: format_number(value, suffix="%"))
    for column in fundamental_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(format_compact_currency)

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

    render_section_header(
        "Step 2",
        f"Point-In-Time Pre-Screening (As of {effective_date})",
        "This screen uses actual historical price structure, liquidity, and fundamental profitability to decide which names deserve deeper research.",
    )
    if screen_state["universe_notes"]:
        st.caption("Universe sources: " + " | ".join(screen_state["universe_notes"]))

    st.markdown("Rules applied: minimum $10 share price, 20D ADV above 1M shares, price above 200DMA, 50DMA above 200DMA, positive free cash flow, and positive net income.")
    st.dataframe(get_display_dataframe(df_results), use_container_width=True, hide_index=True)
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
    st.markdown("#### Research Manager Scoring Criteria and Result")
    criteria = scorecard.get("criteria") or []
    if criteria:
        score_df = pd.DataFrame(criteria)
        st.dataframe(score_df, use_container_width=True, hide_index=True)
    else:
        st.info("Scorecard criteria are not available for this run, so the view is showing the fallback summary only.")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Score", f"{scorecard.get('total_score', 'N/A')}/40")
    s2.metric("Verdict", scorecard.get("verdict", "N/A"))
    confidence = scorecard.get("confidence")
    confidence_text = f"{confidence * 100:.0f}%" if isinstance(confidence, (int, float)) else "N/A"
    s3.metric("Confidence", confidence_text)


def render_execution_and_risk(execution_plan: dict, risk_plan: dict) -> None:
    st.markdown("#### Execution Plan Details")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Action", execution_plan.get("action", "N/A"))
    e2.metric("Entry Zone", execution_plan.get("entry_zone", "N/A"))
    e3.metric("Stop Loss", execution_plan.get("stop_loss", "N/A"))
    e4.metric("Time Horizon", execution_plan.get("time_horizon", "N/A"))
    st.write(execution_plan.get("entry_logic", ""))

    st.markdown("#### Portfolio Manager Risk Allocation")
    r1, r2, r3 = st.columns(3)
    r1.metric("Risk Approval", risk_plan.get("approval", "N/A"))
    r2.metric("Capital Allocation", risk_plan.get("capital_allocation", "N/A"))
    r3.metric("Primary Risk", risk_plan.get("primary_risk", "N/A"))
    st.info(risk_plan.get("allocation_reason", ""))


def render_decision_view(ticker: str, final_state: dict, metrics: dict) -> None:
    transcript = final_state.get("debate_transcript", {})
    scorecard = final_state.get("research_scorecard", {})
    execution_plan = final_state.get("execution_plan", {})
    risk_plan = final_state.get("risk_assessment") or final_state.get("risk_plan", {})
    trader_plan = parse_trader_plan(final_state.get("trader_plan", {}).get("plan_details"))
    if trader_plan and "action" in trader_plan:
        execution_plan = {**execution_plan, **trader_plan}

    render_section_header(
        "Step 3 Analysis",
        f"{ticker} Research Workspace",
        "The views below expose how the system thinks: the factual analyst notes, the scorecard logic, the execution framework, and the final risk-aware portfolio recommendation.",
    )
    render_fact_sheet(metrics)
    render_scorecard(scorecard)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Fundamental Analyst")
        st.markdown(final_state.get("analyst_reports", {}).get("fundamental", "No report generated."))
        st.markdown("#### Bull Case")
        st.markdown(transcript.get("bull", "No bull case generated."))
    with col2:
        st.markdown("#### Technical Analyst")
        st.markdown(final_state.get("analyst_reports", {}).get("technical", "No report generated."))
        st.markdown("#### Bear Case")
        st.markdown(transcript.get("bear", "No bear case generated."))

    st.markdown("#### Research Manager Ruling")
    st.markdown(transcript.get("research_manager_ruling", "No ruling generated."))

    render_execution_and_risk(execution_plan, risk_plan)

    st.markdown("#### Portfolio Manager Final Decision")
    d1, d2 = st.columns(2)
    d1.metric("Final Action", final_state.get("final_decision", "UNKNOWN"))
    d2.metric("Confidence", f"{final_state.get('confidence', 0) * 100:.0f}%")

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
            "Confidence": round(final_state.get("confidence", 0.0), 4),
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
                "Confidence": round(final_state.get("confidence", 0.0), 4),
                "Risk Approval": (final_state.get("risk_assessment") or {}).get("approval", "N/A"),
                "Capital Allocation": (final_state.get("risk_assessment") or {}).get("capital_allocation", "N/A"),
            }
        )
    return decisions


def render_analysis_page() -> None:
    default_date = get_latest_us_trading_day()
    st.sidebar.markdown('<div class="side-label">Universe Source</div>', unsafe_allow_html=True)
    universe_choice = st.sidebar.selectbox(
        "Universe Source",
        options=list(UNIVERSE_CHOICES.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_indices = UNIVERSE_CHOICES[universe_choice]

    st.sidebar.markdown('<div class="side-label">Or Upload Tickers CSV</div>', unsafe_allow_html=True)
    uploaded_universe = st.sidebar.file_uploader(
        "Upload CSV Universe",
        type=["csv"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown('<div class="side-label">Or Enter Tickers</div>', unsafe_allow_html=True)
    tickers_input = st.sidebar.text_area(
        "Direct Tickers",
        value="",
        height=90,
        help="Enter one ticker or multiple tickers with commas, spaces, or new lines.",
        label_visibility="collapsed",
        placeholder="AAPL, MSFT, NVDA",
    )

    st.sidebar.markdown('<div class="side-label">Screen Date</div>', unsafe_allow_html=True)
    target_date = st.sidebar.date_input("Analysis Date", value=default_date, label_visibility="collapsed")
    run_analysis = st.sidebar.button("Run AI Swarm", use_container_width=True)

    st.sidebar.markdown('<div class="side-label">Workflow</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="side-note">1. Build the universe from S&amp;P 500, Nasdaq 100, both, CSV, and/or direct tickers.<br>'
        '2. Run the point-in-time screen using actual historical and fundamental data.<br>'
        '3. Click a qualified ticker to open the analyst, scoring, execution, and portfolio views.</div>',
        unsafe_allow_html=True,
    )

    uploaded_tickers = load_uploaded_tickers(uploaded_universe)
    manual_tickers = parse_direct_tickers(tickers_input)
    effective_date = resolve_analysis_date(target_date)

    if effective_date != target_date:
        st.sidebar.info(f"Selected date is not a US trading day. The app will use {effective_date}.")

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
            screener = HardScreener(price_loader, fundamental_loader)
            survivors, df_results = screener.run_screen(universe, pit_context)

        st.session_state.screen_state = build_screen_state(
            target_date=target_date,
            effective_date=effective_date,
            universe=universe,
            notes=notes,
            macro_data=macro_data,
            survivors=survivors,
            df_results=df_results,
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
