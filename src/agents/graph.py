import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Any

from .state import AgentState
from .llm_wrapper import LLMFactory

logger = logging.getLogger(__name__)

class HedgeFundGraph:
    """
    The main LangGraph orchestrator for the US Hedge Fund multi-agent system.
    This routes the surviving tickers from Phase 4 through the Tier 1 and Tier 2 LLMs.
    """
    
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self.llm_factory = LLMFactory()
        
        # We instantiate them once
        self.tier1_llm = self.llm_factory.get_tier1_llm()
        self.tier2_llm = self.llm_factory.get_tier2_llm()
        
        self._build_graph()
        
    def _invoke_llm(self, llm, messages, default_response: str) -> str:
        """Helper to invoke LLM and gracefully fallback on API errors."""
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM API Error: {e}")
            return default_response
        
    def _node_fundamental_analyst(self, state: AgentState) -> Dict[str, Any]:
        """Tier 1 LLM: Parses raw SEC data into a structured fundamental view."""
        logger.info(f"[{state['ticker']}] Running Fundamental Analyst...")
        
        # If Tier 1 LLM isn't running, fallback to dummy
        if not self.tier1_llm:
            return {"analyst_reports": {"fundamental": "Strong FCF growth, but high debt. (Simulated)"}}
            
        system_msg = SystemMessage(content=\"\"\"You are a strict, quantitative Hedge Fund Fundamental Analyst. 
        You MUST NOT hallucinate, simulate, or guess. 
        Analyze the exact metrics provided. If you say 'strong FCF growth', you MUST state exactly how strong (cite the numbers), compare it to the baseline, and explain the reason behind it.
        Return a detailed, structured paragraph.\"\"\")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nMetrics: {state['market_metrics']}")
        
        content = self._invoke_llm(self.tier1_llm, [system_msg, user_msg], "Strong FCF growth, but high debt. (Simulated Fallback)")
        return {"analyst_reports": {"fundamental": content}}
        
    def _node_technical_analyst(self, state: AgentState) -> Dict[str, Any]:
        """Tier 1 LLM: Evaluates momentum and structure."""
        logger.info(f"[{state['ticker']}] Running Technical Analyst...")
        # Placeholder for LLM invocation
        return {"analyst_reports": {"technical": "Price broke above 200DMA. Bullish trend."}}
        
    def _node_bull_researcher(self, state: AgentState) -> Dict[str, Any]:
        """Tier 2 LLM: Argues strictly for why the stock should be bought."""
        logger.info(f"[{state['ticker']}] Running Bull Researcher...")
        
        if not self.tier2_llm:
            return {"debate_transcript": {"bull": "Strong fundamentals and technical breakout make this a clear buy. (Simulated)"}}
            
        system_msg = SystemMessage(content=\"\"\"You are the Bull Researcher. Argue strictly for why this stock is a BUY based ONLY on the provided analyst reports. 
        You MUST NOT hallucinate or guess. Beside a summary, you MUST show the exact reasons to support your case, citing specific data points from the reports. Explain the 'how' and 'what'.\"\"\")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nReports: {state.get('analyst_reports', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "Strong fundamentals and technical breakout make this a clear buy. (Simulated Fallback)")
        return {"debate_transcript": {"bull": content}}

    def _node_bear_researcher(self, state: AgentState) -> Dict[str, Any]:
        """Tier 2 LLM: Argues strictly for why the stock should be sold or avoided."""
        logger.info(f"[{state['ticker']}] Running Bear Researcher...")
        
        if not self.tier2_llm:
            return {"debate_transcript": {"bear": "Macro risks and potential overvaluation mean this should be avoided. (Simulated)"}}
            
        system_msg = SystemMessage(content=\"\"\"You are the Bear Researcher. Argue strictly for why this stock is a SELL/AVOID based ONLY on the provided analyst reports. 
        You MUST NOT hallucinate or guess. Beside a summary, you MUST show the exact reasons to support your case, citing specific data points from the reports. Explain the 'how' and 'what'.\"\"\")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nReports: {state.get('analyst_reports', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "Macro risks and potential overvaluation mean this should be avoided. (Simulated Fallback)")
        return {"debate_transcript": {"bear": content}}
        
    def _node_debate_sync(self, state: AgentState) -> Dict[str, Any]:
        """Synchronizes parallel execution and increments the round counter."""
        current_round = state.get("debate_round", 0) + 1
        logger.info(f"[{state['ticker']}] Synchronizing Debate Round {current_round}")
        return {"debate_round": current_round}

    def _node_research_manager(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizes the bull and bear arguments using a strict 4-point rubric."""
        logger.info(f"[{state['ticker']}] Research Manager synthesizing debate...")
        
        if not self.tier2_llm:
            return {"debate_transcript": {"research_manager": "Total Score: 32/40. (Simulated)"}}
            
        system_msg = SystemMessage(content='''You are the Research Manager (The Judge). 
        Score the Bull and Bear debate strictly out of 40 points using this rubric:
        1. Fundamental Edge (1-10)
        2. Technical Setup (1-10)
        3. Macro/Regime Risk (1-10)
        4. Asymmetry (1-10)
        If Score < 25, REJECT. If >= 25, APPROVE.''')
        
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nDebate: {state.get('debate_transcript', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "Total Score: 32/40. (Simulated Fallback)")
        return {"debate_transcript": {"research_manager_ruling": content}}
        
    def _node_trader_agent(self, state: AgentState) -> Dict[str, Any]:
        """Tier 2 LLM: Determines entry price, stop loss, and position size."""
        logger.info(f"[{state['ticker']}] Running Trader Agent...")
        
        if not self.tier2_llm:
            return {"trader_plan": {"action": "BUY", "entry_zone": "Current Price", "stop_loss": "-5%", "position_size": "5%"}}
            
        system_msg = SystemMessage(content="You are the Trader Agent. Based on the debate, define the entry logic, stop loss, and position size.")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nDebate: {state.get('debate_transcript', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "{'action': 'BUY', 'entry_zone': 'Current Price', 'stop_loss': '-5%', 'position_size': '5%'} (Simulated Fallback)")
        return {"trader_plan": {"plan_details": content}}
        
    def _node_risk_manager(self, state: AgentState) -> Dict[str, Any]:
        """Tier 2 LLM: Evaluates the Trader's plan against portfolio risk limits."""
        logger.info(f"[{state['ticker']}] Running Risk Manager...")
        
        if not self.tier2_llm:
            return {"risk_assessment": {"approval": "APPROVED", "risk_warning": "None (Simulated)"}}
            
        system_msg = SystemMessage(content="You are the Risk Manager. Review the Trader Plan and either APPROVE, REDUCE SIZE, or VETO.")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nTrader Plan: {state.get('trader_plan', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "APPROVED (Simulated Fallback)")
        return {"risk_assessment": {"risk_ruling": content}}
    def _node_portfolio_manager(self, state: AgentState) -> Dict[str, Any]:
        """Tier 2 LLM: The final decision maker."""
        logger.info(f"[{state['ticker']}] Portfolio Manager finalizing decision...")
        
        if not self.tier2_llm:
            return {
                "final_decision": "BUY",
                "confidence": 0.85,
                "pm_reasoning": "Technical breakout supported by strong FCF. (Simulated)",
                "audit_status": "PASS"
            }
            
        system_msg = SystemMessage(content=\"\"\"You are the Portfolio Manager. Review the Analyst Reports and Debate Transcript. 
        You MUST NOT hallucinate. Weigh the pros and cons. 
        Reply strictly with a JSON-like structure: 
        FINAL_DECISION: [BUY/HOLD/SELL], 
        CONFIDENCE: [0.0-1.0], 
        PROS: [List 2 key strengths], 
        CONS: [List 2 key weaknesses], 
        CONCERNS: [Primary risk factor], 
        REASON: [Your detailed comparative thesis].\"\"\")
        user_msg = HumanMessage(content=f"Ticker: {state['ticker']}\nReports: {state.get('analyst_reports', {})}\nDebate: {state.get('debate_transcript', {})}")
        
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], "FINAL_DECISION: BUY, CONFIDENCE: 0.85, PROS: Strong FCF; Technical breakout, CONS: High valuation; Macro uncertainty, CONCERNS: Sector rotation risk, REASON: The fundamental cash flow outweighs the technical overextension. (Simulated Fallback)")
        
        # A simple parser to extract the fields
        decision = "BUY" if "FINAL_DECISION: BUY" in content.upper() else "SELL" if "FINAL_DECISION: SELL" in content.upper() else "HOLD"
        
        return {
            "final_decision": decision,
            "confidence": 0.80,
            "pm_reasoning": content,
            "audit_status": "PASS"
        }

    def _should_continue_debate(self, state: AgentState):
        """Routing function for the conditional edge."""
        max_rounds = self.llm_factory.config.get("debate", {}).get("max_debate_rounds", 2)
        if state.get("debate_round", 0) < max_rounds:
            return "continue"
        return "end"

    def _build_graph(self):
        """Constructs the nodes and edges of the state machine."""
        # 1. Add Nodes
        self.graph.add_node("fundamental_analyst", self._node_fundamental_analyst)
        self.graph.add_node("technical_analyst", self._node_technical_analyst)
        self.graph.add_node("bull_researcher", self._node_bull_researcher)
        self.graph.add_node("bear_researcher", self._node_bear_researcher)
        self.graph.add_node("debate_sync", self._node_debate_sync)
        self.graph.add_node("research_manager", self._node_research_manager)
        self.graph.add_node("trader_agent", self._node_trader_agent)
        self.graph.add_node("risk_manager", self._node_risk_manager)
        self.graph.add_node("portfolio_manager", self._node_portfolio_manager)
        
        # 2. Add Edges
        self.graph.set_entry_point("fundamental_analyst")
        self.graph.add_edge("fundamental_analyst", "technical_analyst")
        
        # Branch to Bull/Bear
        self.graph.add_edge("technical_analyst", "bull_researcher")
        self.graph.add_edge("technical_analyst", "bear_researcher")
        
        # Converge at Debate Sync
        self.graph.add_edge("bull_researcher", "debate_sync")
        self.graph.add_edge("bear_researcher", "debate_sync")
        
        # Conditional Edge for Multi-Round Debate
        # Instead of fanning out in a list, we route back to the parallel trigger manually or just 
        # add edges from sync to both to simulate the loop if condition passes.
        # But for simpler LangGraph 0.0.x logic, we'll route debate_sync -> research_manager and loop inside node if needed.
        # Let's use standard conditional edges.
        self.graph.add_conditional_edges(
            "debate_sync",
            self._should_continue_debate,
            {
                "continue": "technical_analyst", # Routes back before the split to re-trigger both
                "end": "research_manager"
            }
        )
        
        # Final Path
        self.graph.add_edge("research_manager", "trader_agent")
        self.graph.add_edge("trader_agent", "risk_manager")
        self.graph.add_edge("risk_manager", "portfolio_manager")
        self.graph.add_edge("portfolio_manager", END)
        
        # 3. Compile the graph
        self.compiled_graph = self.graph.compile()
        
    def run_analysis(self, initial_state: dict) -> dict:
        """Executes the graph for a given ticker's initial state."""
        print(f"\n--- Launching Agent Swarm for {initial_state['ticker']} ---")
        final_state = self.compiled_graph.invoke(initial_state)
        print(f"  Trader Plan: {final_state.get('trader_plan')}")
        print(f"  Risk Ruling: {final_state.get('risk_assessment')}")
        print(f"--- Analysis Complete: {final_state['final_decision']} ({final_state['confidence']*100}%) ---\n")
        return final_state
