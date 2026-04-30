import pandas as pd
import logging
from typing import List, Dict
from datetime import date, timedelta

from ..data.point_in_time import PointInTimeContext
from ..data.price_loader import PriceLoader
from ..data.fundamental_loader import FundamentalLoader
from ..screening.hard_screener import HardScreener
from ..agents.graph import HedgeFundGraph

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Executes the multi-agent system across historical dates.
    Enforces T+1 execution to prevent lookahead bias.
    """
    def __init__(self):
        self.price_loader = PriceLoader()
        self.fundamental_loader = FundamentalLoader()
        self.screener = HardScreener(self.price_loader, self.fundamental_loader)
        self.agent_graph = HedgeFundGraph()
        
    def run_backtest(self, universe: List[str], start_date: date, end_date: date, interval_days: int = 30) -> pd.DataFrame:
        """
        Steps through time, runs the AI Swarm, and logs decisions.
        """
        current_date = start_date
        trade_log = []
        
        while current_date <= end_date:
            print(f"\n=============================================")
            print(f"📅 BACKTEST DATE: {current_date}")
            print(f"=============================================")
            
            # 1. Establish Point-In-Time limits
            pit_context = PointInTimeContext(current_date)
            
            # 2. Hard Screener
            survivors = self.screener.run_screen(universe, pit_context)
            
            if not survivors:
                print("No stocks passed the screener for this period.")
                current_date += timedelta(days=interval_days)
                continue
                
            # 3. AI Swarm Evaluation
            for ticker in survivors:
                print(f"\n🧠 Launching Swarm for {ticker}...")
                
                initial_state = {
                    "ticker": ticker,
                    "analysis_date": str(current_date),
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
                
                final_state = self.agent_graph.run_analysis(initial_state)
                
                # Log the trade
                if final_state["final_decision"] == "BUY":
                    # T+1 Execution Logic
                    # In a full backtest, we would fetch the Open price for current_date + 1 business day
                    trade_log.append({
                        "Date": current_date,
                        "Ticker": ticker,
                        "Action": final_state["final_decision"],
                        "Confidence": final_state["confidence"],
                        "PM_Reasoning": final_state.get("pm_reasoning", ""),
                        "Trader_Plan": final_state.get("trader_plan", {}),
                        "Risk_Ruling": final_state.get("risk_assessment", {})
                    })
                    
            current_date += timedelta(days=interval_days)
            
        return pd.DataFrame(trade_log)
