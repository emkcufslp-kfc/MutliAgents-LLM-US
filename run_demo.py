import os
import sys
from datetime import date

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.point_in_time import PointInTimeContext
from src.data.price_loader import PriceLoader
from src.data.fundamental_loader import FundamentalLoader
from src.screening.hard_screener import HardScreener
from src.agents.graph import HedgeFundGraph

def main():
    # 1. Define our Analysis Date (e.g., simulating we are running this on Jan 15, 2024)
    # The system will strictly hide any data published after this date.
    analysis_date = date(2024, 1, 15)
    pit_context = PointInTimeContext(analysis_date)
    
    # 2. Initialize our Data Foundation (Phase 2)
    price_loader = PriceLoader()
    fundamental_loader = FundamentalLoader(fallback_lag_days=45)
    
    # 3. Initialize the Hard Screener (Phase 4)
    screener = HardScreener(price_loader, fundamental_loader)
    
    # 4. Define a small test universe
    test_universe = [
        "AAPL",  # Usually passes
        "NVDA",  # Usually passes
        "AMC",   # Should fail (negative FCF or below 200DMA)
        "GME",   # Should fail
        "PLTR"
    ]
    
    # 5. Execute the screen
    survivors, _ = screener.run_screen(test_universe, pit_context)
    
    print("\n--- Phase 5: LangGraph AI Orchestrator ---")
    print(f"Passing {len(survivors)} surviving tickers to the AI swarm...\n")
    
    # 6. Initialize the LangGraph Orchestrator (Phase 5)
    orchestrator = HedgeFundGraph()
    
    for ticker in survivors:
        initial_state = {
            "ticker": ticker,
            "analysis_date": str(analysis_date),
            "market_metrics": {"fcf_growth": "Positive", "trend": "Bullish"},
            "analyst_reports": {},
            "debate_transcript": {},
            "risk_assessment": {},
            "final_decision": "",
            "confidence": 0.0,
            "pm_reasoning": "",
            "audit_status": "PENDING",
            "audit_notes": []
        }
        orchestrator.run_analysis(initial_state)

if __name__ == "__main__":
    main()
