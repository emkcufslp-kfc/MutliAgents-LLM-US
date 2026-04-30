import ast
import os
import re
import sys
from datetime import date, datetime

import pandas as pd
import streamlit as st

# Ensure the parent directory is in the system path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.graph import HedgeFundGraph
from src.data.fundamental_loader import FundamentalLoader
from src.data.macro_loader import MacroLoader
from src.data.point_in_time import PointInTimeContext
from src.data.price_loader import PriceLoader
from src.screening.hard_screener import HardScreener

st.set_page_config(page_title="Multi-Agent Hedge Fund", layout="wide", page_icon="📈")


def init_session_state() -> None:
    if "app_log" not in st.session_state:
        st.session_state.app_log = []
    if "last_screen_results" not in st.session_state:
        st.session_state.last_screen_results = None
    if "last_survivors" not in st.session_state:
        st.session_state.last_survivors = []


def append_log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.app_log.insert(0, f"[{timestamp}] {message}")
    st.session_state.app_log = st.session_state.app_log[:50]


def render_reference_page() -> None:
    st.title("📚 System Reference")
    st.caption("Workflow, logic, control philosophy, and runtime notes for the US Equity Hedge Fund app.")

    st.subheader("Flow")
    st.markdown(
        """
1. **Step 1: Macro regime**
   Pull FRED macro context first so the system understands the top-down environment.
2. **Step 2: Point-in-time pre-screen**
   Run hard rules on a broad universe before the LLM layer touches anything.
3. **Step 2.5: Optional technical filter**
   Apply an additional technical indicator only when the user wants a stricter shortlist.
4. **Step 3: User selection**
   Let the user choose which pre-screen survivors should go into the multi-agent debate.
5. **Step 4: AI swarm orchestration**
   Run analysts, debate, portfolio management, and execution logic only on the selected names.
"""
    )

    st.subheader("Logic")
    logic_df = pd.DataFrame(
        [
            {
                "Stage": "Step 1",
                "Name": "Macro Regime",
                "Current Logic": "Uses FRED inflation and unemployment to classify the environment.",
                "Why It Exists": "Prevents bottom-up stock ideas from ignoring the broader macro backdrop.",
            },
            {
                "Stage": "Step 2",
                "Name": "Base Pre-Screen",
                "Current Logic": "Checks liquidity, price trend, 50/200DMA structure, positive FCF, and positive net income.",
                "Why It Exists": "Removes low-quality names before expensive downstream analysis.",
            },
            {
                "Stage": "Step 2 Proposal",
                "Name": "600-Ticker Core Universe",
                "Current Logic": "Planned source: uploaded S&P 500 plus Nasdaq 100 list as the default institutional universe.",
                "Why It Exists": "Keeps the universe large enough for opportunity discovery while staying curated and understandable.",
            },
            {
                "Stage": "Step 2.5 Proposal",
                "Name": "Optional Technical Filter",
                "Current Logic": "Planned optional indicator layer before user selection.",
                "Why It Exists": "Lets the user tighten the shortlist without forcing that constraint on every run.",
            },
            {
                "Stage": "Step 3 Proposal",
                "Name": "User Choice",
                "Current Logic": "User selects which survivors deserve full multi-agent analysis.",
                "Why It Exists": "Keeps humans in control of final research allocation.",
            },
            {
                "Stage": "Step 4",
                "Name": "AI Swarm",
                "Current Logic": "Run fundamental analyst, technical analyst, bull/bear debate, research manager, trader, risk manager, and portfolio manager.",
                "Why It Exists": "Converts a filtered candidate list into actionable trade theses with structured reasoning.",
            },
        ]
    )
    st.dataframe(logic_df, use_container_width=True, hide_index=True)

    st.subheader("Description")
    st.markdown(
        """
The system is designed as a cost-control funnel:

- **Macro first** to avoid blind bottom-up stock picking.
- **Rules before models** so deterministic filters remove weak names cheaply.
- **Optional extra filter** so stricter technical confirmation is available without over-constraining every workflow.
- **User selection before debate** so the final LLM budget is spent only on names the user actually wants to study.
- **Multi-agent synthesis last** so debate, risk, and execution are used as a precision layer rather than a screening engine.
"""
    )

    st.subheader("Reason")
    st.markdown(
        """
- **Broad uploaded universe** is more scalable than manual ticker typing.
- **Point-in-time screening** reduces lookahead bias.
- **Optional technical indicator** gives flexibility across swing, momentum, and fundamental-first styles.
- **User-gated Step 3** keeps the app explainable and interactive instead of fully opaque.
- **Reference view on the left** makes the workflow visible to non-technical users and easier to maintain over time.
"""
    )

    st.subheader("Data Source Priority")
    st.markdown(
        """
1. **Checked-in cache files** for deterministic fallback behavior.
2. **yfinance** as the primary live retail-market data source.
3. **Alpha Vantage** as the automatic backup when a key is configured and Yahoo fails.
4. **Uploaded universe lists** for symbol scope, not for price-history screening math.
"""
    )

    st.subheader("Reference Log")
    if st.session_state.app_log:
        st.code("\n".join(st.session_state.app_log), language="text")
    else:
        st.info("No runtime events yet. Run a screen in the Analysis view and the event log will appear here.")


def render_decision_view(final_state: dict, transcript: dict) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🕵️ Independent Analysts")
        reports = final_state.get("analyst_reports", {})
        st.info(f"**Fundamental Analyst:**\n{reports.get('fundamental', 'N/A')}")
        st.info(f"**Technical Analyst:**\n{reports.get('technical', 'N/A')}")

        st.markdown("#### ⚖️ Research Manager Scoring")
        st.warning(f"**Judge Ruling:**\n{transcript.get('research_manager_ruling', 'N/A')}")

    with col2:
        st.markdown(f"#### ⚔️ The Bull/Bear Debate (Round {final_state.get('debate_round', 0)})")
        st.success(f"**🟢 Bull Case:**\n{transcript.get('bull', 'N/A')}")
        st.error(f"**🔴 Bear Case:**\n{transcript.get('bear', 'N/A')}")

    st.divider()
    st.markdown("#### 🏦 Portfolio Manager Final Decision")

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    metrics_col1.metric("Final Action", final_state.get("final_decision", "UNKNOWN"))
    metrics_col2.metric("Confidence", f"{final_state.get('confidence', 0) * 100:.0f}%")
    metrics_col3.metric("Risk Approval", final_state.get("risk_assessment", {}).get("approval", "APPROVED"))

    st.divider()
    st.markdown("#### 🎯 Execution Plan Details")
    plan_str = final_state.get("trader_plan", {}).get("plan_details", "{}")
    clean_str = re.sub(r"\(.*?\)", "", str(plan_str)).strip()

    try:
        plan_dict = ast.literal_eval(clean_str)
        if isinstance(plan_dict, dict):
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Action", plan_dict.get("action", "N/A"))
            ec2.metric("Entry Zone", plan_dict.get("entry_zone", "N/A"))
            ec3.metric("Stop Loss", plan_dict.get("stop_loss", "N/A"))
            ec4.metric("Pos Size", plan_dict.get("position_size", "N/A"))
        else:
            st.info(plan_str)
    except Exception:
        st.info(plan_str)

    st.markdown("#### 🧠 Portfolio Manager Reasoning")
    pm_reason = str(final_state.get("pm_reasoning", "No reasoning provided."))

    pros_match = re.search(r"PROS:\s*(.*?)(?=CONS:|CONCERNS:|REASON:|$)", pm_reason, re.IGNORECASE | re.DOTALL)
    cons_match = re.search(r"CONS:\s*(.*?)(?=CONCERNS:|REASON:|$)", pm_reason, re.IGNORECASE | re.DOTALL)
    concerns_match = re.search(r"CONCERNS:\s*(.*?)(?=REASON:|$)", pm_reason, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r"REASON:\s*(.*?)$", pm_reason, re.IGNORECASE | re.DOTALL)

    if pros_match and cons_match:
        rc1, rc2 = st.columns(2)
        with rc1:
            st.success(f"**👍 PROS:**\n{pros_match.group(1).strip()}")
        with rc2:
            st.error(f"**👎 CONS:**\n{cons_match.group(1).strip()}")

    if concerns_match:
        st.warning(f"**⚠️ CONCERNS:** {concerns_match.group(1).strip()}")

    final_thesis = reason_match.group(1).strip() if reason_match else pm_reason
    st.info(f"**Detailed Thesis:**\n{final_thesis}")


def render_analysis_page() -> None:
    st.title("📈 Multi-Agent US Equity Hedge Fund")
    st.markdown("An institutional-grade, rule-based autonomous AI investment platform.")

    st.sidebar.header("Control Panel")
    target_date = st.sidebar.date_input("Analysis Date (Point-in-Time)", value=date(2024, 1, 15))
    tickers_input = st.sidebar.text_input("Universe (Comma separated)", value="AAPL,NVDA,AMC,GME,PLTR")
    run_analysis = st.sidebar.button("Run AI Swarm")

    st.sidebar.markdown("### Workflow")
    st.sidebar.caption("Current app: Step 1 macro → Step 2 pre-screen → Step 3 AI swarm")
    st.sidebar.caption("Planned workflow: uploaded 600-ticker universe → optional technical filter → user selection → AI swarm")

    if not run_analysis:
        st.info("Use the left panel to run the current screening flow, or open Reference to review the system design and planned workflow.")
        return

    universe = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    append_log(f"Analysis started for {len(universe)} tickers as of {target_date}.")

    st.subheader("🌍 Step 1: Top-Down Macro Regime (Powered by FRED)")
    with st.spinner("Fetching Federal Reserve Economic Data..."):
        macro_loader = MacroLoader()
        macro_data = macro_loader.get_macro_regime()
    append_log(f"Macro regime loaded: {macro_data.get('regime', 'N/A')}.")

    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Current Regime", macro_data.get("regime", "N/A"))
    m_col2.metric("YoY Inflation (CPI)", f"{macro_data.get('cpi', 'N/A')}%")
    m_col3.metric("Unemployment Rate", f"{macro_data.get('unemployment', 'N/A')}%")

    st.divider()
    st.subheader(f"📊 Step 2: Point-In-Time Pre-Screening (As of {target_date})")

    with st.spinner("Fetching historical data and applying strict institutional rules..."):
        pit_context = PointInTimeContext(target_date)
        price_loader = PriceLoader()
        fundamental_loader = FundamentalLoader(fallback_lag_days=45)
        screener = HardScreener(price_loader, fundamental_loader)
        survivors, df_results = screener.run_screen(universe, pit_context)

    st.session_state.last_screen_results = df_results
    st.session_state.last_survivors = survivors
    append_log(f"Pre-screen complete: {len(survivors)} out of {len(universe)} tickers survived.")

    st.markdown("#### 🎯 Institutional Rules Applied")
    st.markdown("- **Liquidity Constraint:** Minimum $10.00 Price & > 1M Shares ADV")
    st.markdown("- **Trend Filter:** Price > 200-Day MA & Golden Cross (50DMA > 200DMA)")
    st.markdown("- **Quality Check:** Positive Free Cash Flow (FCF) & Positive Net Income")
    st.markdown("- **Planned Optional Filter Before Step 3:** User-selectable technical indicator layer")

    st.dataframe(df_results, use_container_width=True, hide_index=True)
    st.success(f"Screening Complete: {len(survivors)} out of {len(universe)} passed the fundamental/technical gates.")
    st.write(f"**Survivors Proceeding to AI Swarm:** {', '.join(survivors) if survivors else 'None'}")

    if not survivors:
        st.warning("No names advanced to the AI swarm. In the planned workflow, this is where the optional technical filter and user selection stage will sit.")
        return

    st.subheader("🧠 Step 3: LangGraph Multi-Agent Orchestration")

    with st.spinner("Initializing AI Swarm..."):
        agent_graph = HedgeFundGraph()
    append_log(f"AI swarm initialized for {len(survivors)} surviving tickers.")

    for ticker in survivors:
        with st.expander(f"Deep Dive Analysis: {ticker}", expanded=False):
            st.write(f"Executing parallel agents for **{ticker}**...")
            append_log(f"Running multi-agent analysis for {ticker}.")

            ticker_metrics = {}
            if not df_results.empty:
                row = df_results[df_results["Ticker"] == ticker]
                if not row.empty:
                    ticker_metrics = row.iloc[0].to_dict()

            initial_state = {
                "ticker": ticker,
                "analysis_date": str(target_date),
                "market_metrics": ticker_metrics,
                "analyst_reports": {},
                "debate_transcript": {},
                "debate_round": 0,
                "trader_plan": {},
                "risk_assessment": {},
                "final_decision": "",
                "confidence": 0.0,
                "pm_reasoning": "",
                "audit_status": "PENDING",
            }

            with st.spinner(f"Agents debating {ticker}..."):
                final_state = agent_graph.compiled_graph.invoke(initial_state)

            transcript = final_state.get("debate_transcript", {})
            append_log(
                f"{ticker} decision complete: {final_state.get('final_decision', 'UNKNOWN')} at {final_state.get('confidence', 0) * 100:.0f}% confidence."
            )
            render_decision_view(final_state, transcript)


def main() -> None:
    init_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go To", ["Analysis", "Reference"], index=0)

    if page == "Reference":
        render_reference_page()
    else:
        render_analysis_page()


if __name__ == "__main__":
    main()
