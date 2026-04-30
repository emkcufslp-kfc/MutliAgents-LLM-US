import streamlit as st
import sys
import os
import pandas as pd
from datetime import date
import ast
import re

# Ensure the parent directory is in the system path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.point_in_time import PointInTimeContext
from src.data.price_loader import PriceLoader
from src.data.fundamental_loader import FundamentalLoader
from src.data.macro_loader import MacroLoader
from src.screening.hard_screener import HardScreener
from src.agents.graph import HedgeFundGraph

st.set_page_config(page_title="Multi-Agent Hedge Fund", layout="wide", page_icon="📈")

st.title("📈 Multi-Agent US Equity Hedge Fund")
st.markdown("An institutional-grade, rule-based autonomous AI investment platform.")

# --- Sidebar Controls ---
st.sidebar.header("Control Panel")
target_date = st.sidebar.date_input("Analysis Date (Point-in-Time)", value=date(2024, 1, 15))
tickers_input = st.sidebar.text_input("Universe (Comma separated)", value="AAPL,NVDA,AMC,GME,PLTR")
run_analysis = st.sidebar.button("Run AI Swarm")

# --- Main Application Logic ---
if run_analysis:
    universe = [t.strip().upper() for t in tickers_input.split(",")]
    
    st.subheader("🌍 Step 1: Top-Down Macro Regime (Powered by FRED)")
    with st.spinner("Fetching Federal Reserve Economic Data..."):
        macro_loader = MacroLoader()
        macro_data = macro_loader.get_macro_regime()
        
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
        
    st.markdown("#### 🎯 Institutional Rules Applied:")
    st.markdown("- **Price Constraint:** Must be trading > $5.00")
    st.markdown("- **Trend Filter:** Must be trading above the 200-Day Moving Average")
    st.markdown("- **Quality Check:** Must generate Positive Free Cash Flow (FCF)")
    
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.success(f"Screening Complete: {len(survivors)} out of {len(universe)} passed the fundamental/technical gates.")
    st.write(f"**Survivors Proceeding to AI Swarm:** {', '.join(survivors) if survivors else 'None'}")
    
    if survivors:
        st.subheader("🧠 Step 3: LangGraph Multi-Agent Orchestration")
        
        with st.spinner("Initializing AI Swarm..."):
            agent_graph = HedgeFundGraph()
            
        for ticker in survivors:
            with st.expander(f"Deep Dive Analysis: {ticker}", expanded=False):
                st.write(f"Executing parallel agents for **{ticker}**...")
                
                initial_state = {
                    "ticker": ticker,
                    "analysis_date": str(target_date),
                    "market_metrics": {},
                    "analyst_reports": {},
                    "debate_transcript": {},
                    "debate_round": 0,
                    "trader_plan": {},
                    "risk_assessment": {},
                    "final_decision": "",
                    "confidence": 0.0,
                    "pm_reasoning": "",
                    "audit_status": "PENDING"
                }
                
                # We use a spinner for each ticker as the LLM requests can take time
                with st.spinner(f"Agents debating {ticker}..."):
                    final_state = agent_graph.compiled_graph.invoke(initial_state)
                
                # --- Present the Output Beautifully ---
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🕵️ Independent Analysts")
                    reports = final_state.get('analyst_reports', {})
                    st.info(f"**Fundamental Analyst:**\n{reports.get('fundamental', 'N/A')}")
                    st.info(f"**Technical Analyst:**\n{reports.get('technical', 'N/A')}")
                    
                    st.markdown("#### ⚖️ Research Manager Scoring")
                    transcript = final_state.get('debate_transcript', {})
                    st.warning(f"**Judge Ruling:**\n{transcript.get('research_manager_ruling', 'N/A')}")
                    
                with col2:
                    st.markdown(f"#### ⚔️ The Bull/Bear Debate (Round {final_state.get('debate_round', 0)})")
                    st.success(f"**🟢 Bull Case:**\n{transcript.get('bull', 'N/A')}")
                    st.error(f"**🔴 Bear Case:**\n{transcript.get('bear', 'N/A')}")
                    
                st.divider()
                
                # Execution & Decision
                st.markdown("#### 🏦 Portfolio Manager Final Decision")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                metrics_col1.metric("Final Action", final_state.get("final_decision", "UNKNOWN"))
                metrics_col2.metric("Confidence", f"{final_state.get('confidence', 0)*100}%")
                metrics_col3.metric("Risk Approval", final_state.get("risk_assessment", {}).get("approval", "APPROVED"))
                
                st.divider()
                st.markdown("#### 🎯 Execution Plan Details")
                plan_str = final_state.get('trader_plan', {}).get('plan_details', "{}")
                
                clean_str = re.sub(r'\(.*?\)', '', str(plan_str)).strip()
                try:
                    plan_dict = ast.literal_eval(clean_str)
                    if isinstance(plan_dict, dict):
                        ec1, ec2, ec3, ec4 = st.columns(4)
                        ec1.metric("Action", plan_dict.get('action', 'N/A'))
                        ec2.metric("Entry Zone", plan_dict.get('entry_zone', 'N/A'))
                        ec3.metric("Stop Loss", plan_dict.get('stop_loss', 'N/A'))
                        ec4.metric("Pos Size", plan_dict.get('position_size', 'N/A'))
                    else:
                        st.info(plan_str)
                except:
                    st.info(plan_str)
                
                st.markdown("#### 🧠 Portfolio Manager Reasoning")
                pm_reason = str(final_state.get('pm_reasoning', 'No reasoning provided.'))
                
                # Extract fields if they exist using regex
                pros_match = re.search(r'PROS:\s*(.*?)(?=CONS:|CONCERNS:|REASON:|$)', pm_reason, re.IGNORECASE | re.DOTALL)
                cons_match = re.search(r'CONS:\s*(.*?)(?=CONCERNS:|REASON:|$)', pm_reason, re.IGNORECASE | re.DOTALL)
                concerns_match = re.search(r'CONCERNS:\s*(.*?)(?=REASON:|$)', pm_reason, re.IGNORECASE | re.DOTALL)
                reason_match = re.search(r'REASON:\s*(.*?)$', pm_reason, re.IGNORECASE | re.DOTALL)
                
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
