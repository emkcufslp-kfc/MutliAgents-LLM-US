import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from .llm_wrapper import LLMFactory
from .state import AgentState

logger = logging.getLogger(__name__)


class HedgeFundGraph:
    """
    The main LangGraph orchestrator for the US Hedge Fund multi-agent system.
    This routes the surviving tickers from Phase 4 through the Tier 1 and Tier 2 LLMs.
    """

    def __init__(self):
        self.graph = StateGraph(AgentState)
        self.llm_factory = LLMFactory()
        self.tier1_llm = self.llm_factory.get_tier1_llm()
        self.tier2_llm = self.llm_factory.get_tier2_llm()
        self._build_graph()

    def _invoke_llm(self, llm, messages, default_response: str) -> str:
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as exc:
            logger.error(f"LLM API error: {exc}")
            return default_response

    def _metrics(self, state: AgentState) -> Dict[str, Any]:
        return state.get("market_metrics", {}) or {}

    def _scorecard(self, state: AgentState) -> Dict[str, Any]:
        return state.get("research_scorecard", {}) or {}

    def _execution_plan(self, state: AgentState) -> Dict[str, Any]:
        return state.get("execution_plan", {}) or {}

    def _risk_plan(self, state: AgentState) -> Dict[str, Any]:
        return state.get("risk_plan", {}) or {}

    def describe_agent_setup(self) -> Dict[str, Any]:
        llm_config = self.llm_factory.config.get("llm", {})
        tier1_provider = llm_config.get("tier1_provider", "unknown")
        tier1_model = llm_config.get("tier1_model", "unknown")
        tier2_provider = llm_config.get("tier2_provider", "unknown")
        tier2_model = llm_config.get("tier2_model", "unknown")
        tier1_runtime = "live-llm" if self.tier1_llm else "deterministic-fallback"
        tier2_runtime = "live-llm" if self.tier2_llm else "deterministic-fallback"

        return {
            "fundamental_analyst": {
                "provider": tier1_provider,
                "model": tier1_model,
                "runtime_mode": tier1_runtime,
            },
            "technical_analyst": {
                "provider": tier1_provider,
                "model": tier1_model,
                "runtime_mode": tier1_runtime,
            },
            "bull_researcher": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "bear_researcher": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "research_manager": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "trader_agent": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "risk_manager": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "portfolio_manager": {
                "provider": tier2_provider,
                "model": tier2_model,
                "runtime_mode": tier2_runtime,
            },
            "audit_manager": {
                "provider": "system",
                "model": "deterministic-audit",
                "runtime_mode": "deterministic-validation",
            },
        }

    def _historical_guardrail(self, state: AgentState) -> str:
        return (
            f"Treat {state['analysis_date']} as the current date for this task. "
            "Use only the provided metrics, reports, scorecard, and macro context. "
            "Do not introduce facts, prices, earnings, macro releases, or events published after that date."
        )

    def _analysis_protocol(self) -> Dict[str, Any]:
        max_rounds = self.llm_factory.config.get("debate", {}).get("max_debate_rounds", 2)
        return {
            "analysis_basis": "point-in-time-only",
            "evidence_scope": [
                "market_metrics",
                "macro_context",
                "research_scorecard",
                "fundamental_report",
                "technical_report",
            ],
            "debate_style": "parallel adversarial memos with rebuttal context",
            "max_debate_rounds": max_rounds,
            "independence_note": (
                "Bull and Bear agents are run as separate nodes. They may rebut each other using prior transcript text, "
                "but they are only allowed to cite the supplied point-in-time dataset."
            ),
        }

    def describe_analysis_protocol(self) -> Dict[str, Any]:
        return self._analysis_protocol()

    def _deterministic_fundamental_report(self, state: AgentState) -> str:
        metrics = self._metrics(state)
        return (
            f"Free cash flow is {metrics.get('FCF')} and net income is {metrics.get('Net Income')} "
            f"based on the most recent report dated {metrics.get('Most Recent Report Date', 'N/A')}. "
            f"Operating cash flow is {metrics.get('Operating CF')}. The name passed the hard screen because "
            f"cash generation and earnings were both positive as of {state['analysis_date']}."
        )

    def _deterministic_technical_report(self, state: AgentState) -> str:
        metrics = self._metrics(state)
        return (
            f"- Last close: {metrics.get('Price')}\n"
            f"- 50DMA: {metrics.get('50 DMA')}\n"
            f"- 200DMA: {metrics.get('200 DMA')}\n"
            f"- Price vs 200DMA: {metrics.get('Price vs 200DMA %')}%\n"
            f"- 1M / 3M / 6M returns: {metrics.get('1M Return %')}% / {metrics.get('3M Return %')}% / {metrics.get('6M Return %')}%\n"
            f"- 20D average daily volume: {metrics.get('20D ADV')}\n"
            f"- 20D annualized volatility: {metrics.get('20D Volatility %')}%"
        )

    def _deterministic_bull_case(self, state: AgentState) -> str:
        metrics = self._metrics(state)
        scorecard = self._scorecard(state)
        return (
            f"## What\n{state['ticker']} passed the point-in-time screen and earned {scorecard.get('total_score', 'N/A')}/40 "
            f"on the research scorecard.\n\n"
            f"## Why\nThe long case is supported by positive profitability, supportive trend structure, and adequate liquidity.\n\n"
            f"## How\nThis argument uses only the supplied historical metrics and analyst notes as of {state['analysis_date']}.\n\n"
            f"## Evidence\n"
            f"- Price remains {metrics.get('Price vs 200DMA %')}% above the 200DMA.\n"
            f"- The 50DMA sits {metrics.get('50DMA vs 200DMA %')}% above the 200DMA, confirming a positive trend stack.\n"
            f"- Free cash flow is positive at {metrics.get('FCF')} and net income is {metrics.get('Net Income')}.\n"
            f"- 20D ADV is {metrics.get('20D ADV')}, which supports execution liquidity."
        )

    def _deterministic_bear_case(self, state: AgentState) -> str:
        metrics = self._metrics(state)
        scorecard = self._scorecard(state)
        risks = scorecard.get("principal_risks", [])
        risk_text = "; ".join(risks) if risks else "Trend or macro conditions could weaken after the screen date."
        return (
            f"## What\nThe setup is investable, but it is not risk-free.\n\n"
            f"## Why\nThe bear case focuses on volatility, extension risk, and the possibility that trend support fails.\n\n"
            f"## How\nThis counter-case is built from the same point-in-time dataset as the bull case, without any future knowledge.\n\n"
            f"## Evidence\n"
            f"- 20D annualized volatility is {metrics.get('20D Volatility %')}%, which can widen stop-loss needs.\n"
            f"- Momentum can reverse if price loses the 50DMA ({metrics.get('50 DMA')}) or 200DMA ({metrics.get('200 DMA')}).\n"
            f"- Primary risks noted by the scorecard: {risk_text}"
        )

    def _deterministic_research_ruling(self, state: AgentState) -> str:
        scorecard = self._scorecard(state)
        verdict = scorecard.get("verdict", "REVIEW")
        criteria_lines = []
        for item in scorecard.get("criteria", []):
            criteria_lines.append(f"- {item['criterion']}: {item['score']}/10. {item['reason']}")
        joined = "\n".join(criteria_lines) if criteria_lines else "- Scorecard not available."
        return (
            f"## What\nThe research manager reviewed the bull case, bear case, and the explicit scorecard.\n\n"
            f"## Why\nThe ruling is driven by the factor scores and whether the evidence is internally consistent.\n\n"
            f"## How\nThe decision is made by weighting the four scorecard buckets equally and checking whether the debate evidence supports the score.\n\n"
            f"## Final Decision\n"
            f"- Total score: {scorecard.get('total_score', 'N/A')}/40\n"
            f"- Verdict: {verdict}\n"
            f"- Criteria:\n{joined}"
        )

    def _deterministic_risk_ruling(self, state: AgentState) -> str:
        risk_plan = self._risk_plan(state)
        execution_plan = self._execution_plan(state)
        return (
            "## What\n"
            f"The risk manager reviewed the proposed trade action of {execution_plan.get('action', 'N/A')} and the draft allocation plan.\n\n"
            "## Why\n"
            f"The allocation is capped by the score, volatility, and macro regime, with primary concern on {risk_plan.get('primary_risk', 'N/A')}.\n\n"
            "## How\n"
            "Risk sizing starts from the deterministic risk plan and is then approved, reduced, or vetoed using the same point-in-time evidence set.\n\n"
            "## Allocation Decision\n"
            f"- Approval: {risk_plan.get('approval', 'N/A')}\n"
            f"- Capital allocation: {risk_plan.get('capital_allocation', 'N/A')}\n"
            f"- Reason: {risk_plan.get('allocation_reason', 'N/A')}"
        )

    def _collect_audit_findings(self, state: AgentState) -> list[str]:
        notes = list(state.get("audit_notes", []) or [])
        metrics = self._metrics(state)
        scorecard = self._scorecard(state)
        execution_plan = self._execution_plan(state)
        risk_assessment = state.get("risk_assessment", {}) or {}
        analysis_date = str(state.get("analysis_date", ""))

        notes.append(f"Audit scope: validate all evidence against analysis date {analysis_date}.")

        price_end = str(metrics.get("Price Data End", "N/A"))
        report_date = str(metrics.get("Most Recent Report Date", "N/A"))

        if price_end != "N/A" and price_end > analysis_date:
            notes.append(f"FAIL: price data end date {price_end} is later than analysis date {analysis_date}.")
        else:
            notes.append(f"PASS: price data end date {price_end} is not later than analysis date {analysis_date}.")

        if report_date != "N/A" and report_date > analysis_date:
            notes.append(f"FAIL: most recent report date {report_date} is later than analysis date {analysis_date}.")
        else:
            notes.append(f"PASS: most recent report date {report_date} is not later than analysis date {analysis_date}.")

        required_sections = {
            "fundamental_report": bool((state.get("analyst_reports", {}) or {}).get("fundamental")),
            "technical_report": bool((state.get("analyst_reports", {}) or {}).get("technical")),
            "bull_case": bool((state.get("debate_transcript", {}) or {}).get("bull")),
            "bear_case": bool((state.get("debate_transcript", {}) or {}).get("bear")),
            "research_manager_ruling": bool((state.get("debate_transcript", {}) or {}).get("research_manager_ruling")),
            "risk_ruling": bool(risk_assessment.get("risk_ruling") or risk_assessment.get("approval")),
            "portfolio_manager_reasoning": bool(state.get("pm_reasoning")),
        }
        for section, present in required_sections.items():
            notes.append(f"{'PASS' if present else 'FAIL'}: {section} {'present' if present else 'missing'}.")

        total_score = float(scorecard.get("total_score", 0) or 0)
        decision = str(state.get("final_decision", "UNKNOWN"))
        expected_action = execution_plan.get("action", "UNKNOWN")
        if decision == expected_action:
            notes.append(f"PASS: final decision {decision} matches execution plan action {expected_action}.")
        else:
            notes.append(f"WARN: final decision {decision} differs from execution plan action {expected_action}.")

        approval = str(risk_assessment.get("approval", "N/A"))
        if approval == "VETO" and decision == "BUY":
            notes.append("FAIL: final decision is BUY even though risk manager vetoed the trade.")
        elif approval == "HOLD" and decision == "BUY":
            notes.append("WARN: final decision is BUY while risk approval is HOLD.")
        else:
            notes.append(f"PASS: final decision {decision} is not in direct conflict with risk approval {approval}.")

        if decision == "BUY" and total_score < 26:
            notes.append(f"FAIL: BUY decision is inconsistent with low total score {total_score:.2f}.")
        elif decision in {"HOLD", "AVOID", "SELL"} and total_score >= 32:
            notes.append(f"WARN: conservative final decision {decision} despite strong total score {total_score:.2f}.")
        else:
            notes.append(f"PASS: final decision {decision} is directionally consistent with total score {total_score:.2f}.")

        return notes

    def _derive_audit_status(self, notes: list[str]) -> str:
        if any(note.startswith("FAIL:") for note in notes):
            return "FAIL"
        if any(note.startswith("WARN:") for note in notes):
            return "WARN"
        return "PASS"

    def _deterministic_audit_summary(self, notes: list[str], status: str, state: AgentState) -> str:
        decision = str(state.get("final_decision", "UNKNOWN"))
        analysis_date = str(state.get("analysis_date", "N/A"))
        note_lines = "\n".join(f"- {note}" for note in notes)
        return (
            "## What\n"
            f"The audit manager verified date integrity, required evidence coverage, and final decision consistency for analysis date {analysis_date}.\n\n"
            "## Why\n"
            "This final check helps the user confirm that the workflow stayed point-in-time clean and that the final recommendation is traceable.\n\n"
            "## How\n"
            "The audit compares source dates against the selected analysis date, checks that all required sections were generated, and validates the final decision against score and risk outputs.\n\n"
            "## Audit Result\n"
            f"- Audit status: {status}\n"
            f"- Final decision under review: {decision}\n"
            f"- Verification log:\n{note_lines}"
        )

    def _deterministic_portfolio_reasoning(self, state: AgentState) -> str:
        scorecard = self._scorecard(state)
        execution_plan = self._execution_plan(state)
        risk_plan = self._risk_plan(state)
        return (
            f"FINAL_DECISION: {execution_plan.get('action', 'HOLD')}, "
            f"CONFIDENCE: {scorecard.get('confidence', 0.5):.2f}, "
            f"PROS: Positive trend structure; positive cash generation, "
            f"CONS: Momentum can reverse; macro conditions can change, "
            f"CONCERNS: {risk_plan.get('primary_risk', 'Execution and trend risk')}, "
            f"REASON: The rating is anchored in the point-in-time screen, the explicit 40-point scorecard, "
            f"and the proposed risk allocation of {risk_plan.get('capital_allocation', 'N/A')}."
        )

    def _node_fundamental_analyst(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Fundamental Analyst...")

        if not self.tier1_llm:
            return {"analyst_reports": {"fundamental": self._deterministic_fundamental_report(state)}}

        system_msg = SystemMessage(content="""You are a strict, quantitative Hedge Fund Fundamental Analyst.
        You must not hallucinate, simulate, or guess.
        Analyze the exact metrics provided. State the figures, explain what they imply, and stay grounded in the data.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Metrics: {state['market_metrics']}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        fallback = self._deterministic_fundamental_report(state)
        content = self._invoke_llm(self.tier1_llm, [system_msg, user_msg], fallback)
        return {"analyst_reports": {"fundamental": content}}

    def _node_technical_analyst(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Technical Analyst...")

        if not self.tier1_llm:
            return {"analyst_reports": {"technical": self._deterministic_technical_report(state)}}

        system_msg = SystemMessage(content="""You are a strict Hedge Fund Technical Analyst.
        Analyze the exact metrics provided (price, moving averages, returns, volume, volatility).
        You must not hallucinate. Use concise markdown bullet points.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Metrics: {state.get('market_metrics', {})}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        fallback = self._deterministic_technical_report(state)
        content = self._invoke_llm(self.tier1_llm, [system_msg, user_msg], fallback)
        return {"analyst_reports": {"technical": content}}

    def _node_bull_researcher(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Bull Researcher...")

        if not self.tier2_llm:
            return {"debate_transcript": {"bull": self._deterministic_bull_case(state)}}

        system_msg = SystemMessage(content="""You are the Bull Researcher.
        Build the strongest long thesis using only the provided analyst reports, scorecard, macro context, and exact market metrics.
        You are independent from the Bear Researcher. If prior bear arguments are supplied, rebut them only with the provided evidence.
        Format the answer with markdown sections: ## What, ## Why, ## How, ## Evidence.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Metrics: {state.get('market_metrics', {})}\n"
            f"Reports: {state.get('analyst_reports', {})}\n"
            f"Macro context: {state.get('macro_context', {})}\n"
            f"Scorecard: {state.get('research_scorecard', {})}\n"
            f"Prior debate transcript: {state.get('debate_transcript', {})}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        fallback = self._deterministic_bull_case(state)
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], fallback)
        return {"debate_transcript": {"bull": content}}

    def _node_bear_researcher(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Bear Researcher...")

        if not self.tier2_llm:
            return {"debate_transcript": {"bear": self._deterministic_bear_case(state)}}

        system_msg = SystemMessage(content="""You are the Bear Researcher.
        Build the strongest risk or downside thesis using only the provided analyst reports, scorecard, macro context, and exact market metrics.
        You are independent from the Bull Researcher. If prior bull arguments are supplied, rebut them only with the provided evidence.
        Format the answer with markdown sections: ## What, ## Why, ## How, ## Evidence.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Metrics: {state.get('market_metrics', {})}\n"
            f"Reports: {state.get('analyst_reports', {})}\n"
            f"Macro context: {state.get('macro_context', {})}\n"
            f"Scorecard: {state.get('research_scorecard', {})}\n"
            f"Prior debate transcript: {state.get('debate_transcript', {})}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        fallback = self._deterministic_bear_case(state)
        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], fallback)
        return {"debate_transcript": {"bear": content}}

    def _node_debate_sync(self, state: AgentState) -> Dict[str, Any]:
        current_round = state.get("debate_round", 0) + 1
        logger.info(f"[{state['ticker']}] Synchronizing Debate Round {current_round}")
        return {"debate_round": current_round}

    def _node_research_manager(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Research Manager synthesizing debate...")

        fallback = self._deterministic_research_ruling(state)
        if not self.tier2_llm:
            return {"debate_transcript": {"research_manager_ruling": fallback}}

        system_msg = SystemMessage(content="""You are the Research Manager.
        Review the bull case, bear case, and scorecard. Use only the provided evidence.
        Format the answer with markdown sections: ## What, ## Why, ## How, ## Final Decision.
        In Final Decision, state the score, verdict, and the key criteria that drove the call.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Debate: {state.get('debate_transcript', {})}\n"
            f"Scorecard: {state.get('research_scorecard', {})}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], fallback)
        return {"debate_transcript": {"research_manager_ruling": content}}

    def _node_trader_agent(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Trader Agent...")
        fallback_plan = self._execution_plan(state) or {
            "action": "HOLD",
            "entry_zone": "Await clearer setup",
            "stop_loss": "N/A",
            "position_size": "0%",
        }

        if not self.tier2_llm:
            return {"trader_plan": {"plan_details": fallback_plan}}

        system_msg = SystemMessage(content="You are the Trader Agent. Provide action, entry logic, stop loss, and position size. Respect the supplied analysis date and do not use any future information.")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Debate: {state.get('debate_transcript', {})}\n"
            f"Execution plan draft: {fallback_plan}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], str(fallback_plan))
        return {"trader_plan": {"plan_details": content}}

    def _node_risk_manager(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Running Risk Manager...")
        fallback_risk = self._risk_plan(state) or {"approval": "REVIEW", "capital_allocation": "0%", "primary_risk": "No risk plan available"}

        if not self.tier2_llm:
            risk_assessment = dict(fallback_risk)
            risk_assessment["risk_ruling"] = self._deterministic_risk_ruling(state)
            return {"risk_assessment": risk_assessment}

        system_msg = SystemMessage(content="""You are the Risk Manager.
        Review the trade plan and respond with APPROVE, REDUCE SIZE, or VETO, with reasons.
        Format the answer with markdown sections: ## What, ## Why, ## How, ## Allocation Decision.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Trader Plan: {state.get('trader_plan', {})}\n"
            f"Debate: {state.get('debate_transcript', {})}\n"
            f"Scorecard: {state.get('research_scorecard', {})}\n"
            f"Risk draft: {fallback_risk}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], fallback_risk.get("approval", "REVIEW"))
        approval = "APPROVED"
        upper = content.upper()
        if "VETO" in upper:
            approval = "VETO"
        elif "REDUCE" in upper:
            approval = "REDUCE SIZE"

        risk_assessment = dict(fallback_risk)
        risk_assessment["approval"] = approval
        risk_assessment["risk_ruling"] = content
        return {"risk_assessment": risk_assessment}

    def _node_portfolio_manager(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Portfolio Manager finalizing decision...")

        fallback = self._deterministic_portfolio_reasoning(state)
        if not self.tier2_llm:
            execution_plan = self._execution_plan(state)
            scorecard = self._scorecard(state)
            return {
                "final_decision": execution_plan.get("action", "HOLD"),
                "confidence": float(scorecard.get("confidence", 0.5)),
                "pm_reasoning": fallback,
                "audit_status": "PASS",
            }

        system_msg = SystemMessage(content="""You are the Portfolio Manager.
        Review the analyst reports, debate transcript, research scorecard, and risk plan.
        Reply with FINAL_DECISION, CONFIDENCE, PROS, CONS, CONCERNS, and REASON.
        Respect the supplied analysis date and do not use any future information.""")
        user_msg = HumanMessage(content=(
            f"Ticker: {state['ticker']}\n"
            f"Analysis date: {state['analysis_date']}\n"
            f"Reports: {state.get('analyst_reports', {})}\n"
            f"Debate: {state.get('debate_transcript', {})}\n"
            f"Scorecard: {state.get('research_scorecard', {})}\n"
            f"Risk Plan: {state.get('risk_plan', {})}\n"
            f"Rule: {self._historical_guardrail(state)}"
        ))

        content = self._invoke_llm(self.tier2_llm, [system_msg, user_msg], fallback)
        upper = content.upper()
        if "FINAL_DECISION: SELL" in upper:
            decision = "SELL"
        elif "FINAL_DECISION: HOLD" in upper:
            decision = "HOLD"
        else:
            decision = "BUY"

        confidence_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", content, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else float(self._scorecard(state).get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        return {
            "final_decision": decision,
            "confidence": confidence,
            "pm_reasoning": content,
        }

    def _node_audit_manager(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[{state['ticker']}] Audit Manager validating evidence chain...")
        notes = self._collect_audit_findings(state)
        status = self._derive_audit_status(notes)
        summary = self._deterministic_audit_summary(notes, status, state)
        return {
            "audit_status": status,
            "audit_notes": notes,
            "debate_transcript": {"audit_manager_ruling": summary},
        }

    def _should_continue_debate(self, state: AgentState):
        max_rounds = self.llm_factory.config.get("debate", {}).get("max_debate_rounds", 2)
        if state.get("debate_round", 0) < max_rounds:
            return "continue"
        return "end"

    def _build_graph(self):
        self.graph.add_node("fundamental_analyst", self._node_fundamental_analyst)
        self.graph.add_node("technical_analyst", self._node_technical_analyst)
        self.graph.add_node("bull_researcher", self._node_bull_researcher)
        self.graph.add_node("bear_researcher", self._node_bear_researcher)
        self.graph.add_node("debate_sync", self._node_debate_sync)
        self.graph.add_node("research_manager", self._node_research_manager)
        self.graph.add_node("trader_agent", self._node_trader_agent)
        self.graph.add_node("risk_manager", self._node_risk_manager)
        self.graph.add_node("portfolio_manager", self._node_portfolio_manager)
        self.graph.add_node("audit_manager", self._node_audit_manager)

        self.graph.set_entry_point("fundamental_analyst")
        self.graph.add_edge("fundamental_analyst", "technical_analyst")
        self.graph.add_edge("technical_analyst", "bull_researcher")
        self.graph.add_edge("technical_analyst", "bear_researcher")
        self.graph.add_edge("bull_researcher", "debate_sync")
        self.graph.add_edge("bear_researcher", "debate_sync")
        self.graph.add_conditional_edges(
            "debate_sync",
            self._should_continue_debate,
            {
                "continue": "technical_analyst",
                "end": "research_manager",
            },
        )
        self.graph.add_edge("research_manager", "trader_agent")
        self.graph.add_edge("trader_agent", "risk_manager")
        self.graph.add_edge("risk_manager", "portfolio_manager")
        self.graph.add_edge("portfolio_manager", "audit_manager")
        self.graph.add_edge("audit_manager", END)

        self.compiled_graph = self.graph.compile()

    def run_analysis(self, initial_state: dict) -> dict:
        logger.info(f"Launching agent swarm for {initial_state['ticker']}")
        final_state = self.compiled_graph.invoke(initial_state)
        final_state.setdefault("agent_models", self.describe_agent_setup())
        final_state.setdefault("analysis_protocol", self._analysis_protocol())
        final_state.setdefault("audit_notes", [])
        logger.info(
            f"Analysis complete for {initial_state['ticker']}: "
            f"{final_state.get('final_decision')} ({final_state.get('confidence', 0) * 100:.0f}%)"
        )
        return final_state
