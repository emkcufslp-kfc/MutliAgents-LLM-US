import os
import sys
from datetime import date

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.backtest.engine import BacktestEngine

def pretty_print_decision(ticker, final_state):
    print(f"\n==================================================")
    print(f"[REPORT] FULL AI DEBATE & DECISION LOG FOR {ticker}")
    print(f"==================================================")
    
    # 1. Analyst Reports
    print("\n[PHASE 5: INDEPENDENT ANALYSTS]")
    for analyst, report in final_state.get('analyst_reports', {}).items():
        print(f"  -> {analyst.capitalize()}: {report}")
        
    # 2. Bull vs Bear Debate
    print(f"\n[PHASE 6: 2-ROUND DEBATE (Final Round: {final_state.get('debate_round', 0)})]")
    transcript = final_state.get('debate_transcript', {})
    print(f"  [BULL]: {transcript.get('bull', 'No data')}")
    print(f"  [BEAR]: {transcript.get('bear', 'No data')}")
    
    # 3. Research Manager
    print("\n[PHASE 6.5: RESEARCH MANAGER RUBRIC SCORING]")
    print(f"  [JUDGE RULING]: {transcript.get('research_manager_ruling', transcript.get('research_manager', 'No data'))}")
    
    # 4. Execution Plan
    print("\n[PHASE 7 & 8: EXECUTION & RISK]")
    trader_plan = final_state.get('trader_plan', {}).get('plan_details', final_state.get('trader_plan', {}))
    print(f"  [TRADER PLAN]: {trader_plan}")
    risk_ruling = final_state.get('risk_assessment', {}).get('risk_ruling', final_state.get('risk_assessment', {}))
    print(f"  [RISK VETO]: {risk_ruling}")
    
    # 5. Final PM Decision
    print("\n[PHASE 9: PORTFOLIO MANAGER FINAL DECISION]")
    print(f"  [DECISION]: {final_state.get('final_decision')} (Confidence: {final_state.get('confidence', 0)*100}%)")
    print(f"  [REASONING]: {final_state.get('pm_reasoning')}")
    print("==================================================\n")

def main():
    print("Initializing Multi-Agent Backtest Engine...")
    
    # To run a full test on AAPL, TSLA, MSFT
    universe = ["AAPL", "TSLA", "MSFT"]
    
    # For demonstration, we run a single date backtest to see the deep logic.
    test_date = date(2024, 1, 15)
    
    engine = BacktestEngine()
    
    # We will hook into the engine or just run the graph directly for the detailed printout
    print(f"Running deep-dive test for Date: {test_date}")
    
    from src.data.point_in_time import PointInTimeContext
    pit_context = PointInTimeContext(test_date)
    
    survivors, _ = engine.screener.run_screen(universe, pit_context)
    
    for ticker in survivors:
        initial_state = {
            "ticker": ticker,
            "analysis_date": str(test_date),
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
        
        final_state = engine.agent_graph.compiled_graph.invoke(initial_state)
        pretty_print_decision(ticker, final_state)

if __name__ == "__main__":
    main()
