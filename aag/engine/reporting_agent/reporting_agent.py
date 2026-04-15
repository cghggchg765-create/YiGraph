"""LangGraph-based Reporting Agent for direct report generation."""

import logging
from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from aag.reasoner.model_deployment import Reasoner
from .state import ReportingState
logger = logging.getLogger(__name__)
class ReportingAgent:
    """Reporting agent with a single generation node.

    Workflow:
        START -> generate_report -> END
    """

    def __init__(self, reasoner: Reasoner, max_retry: int = 2) -> None:
        if reasoner is None:
            raise ValueError("ReportingAgent requires a valid reasoner")
        if max_retry < 0:
            raise ValueError("max_retry must be >= 0")

        self.reasoner = reasoner
        self.max_retry = max_retry

    def generate_report(self, state: ReportingState) -> Dict[str, Any]:
        """Generate final report via the project-native Reasoner method.

        Args:
            state: Current reporting workflow state.

        Returns:
            Partial state update containing final_report.
        """
        question = str(state.get("question") or "").strip()
        tool_description = str(state.get("tool_description") or "").strip()
        tool_result = state.get("tool_result")

        if not question:
            return {
                "final_report": "Unable to generate report because question is empty.",
            }
        if not tool_description:
            tool_description = "Unknown graph algorithm"
        try:
            output = self.reasoner.generate_answer_from_algorithm_result(
                question=question,
                tool_description=tool_description,
                tool_result=tool_result,
            )
            final_report = str(output or "").strip()
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.error("generate_report failed: %s", exc, exc_info=True)
            final_report = (
                "Report generation failed due to temporary model service error. "
                "Please retry later."
            )
        if not final_report:
            final_report = "Model returned an empty report. Please retry with the same input."

        return {
            "final_report": final_report,
        }
    
    def compile(self) -> Any:
        """Compile the reporting StateGraph.

        Returns:
            Compiled LangGraph application.
        """
        workflow = StateGraph(ReportingState)
        workflow.add_node("generate_report", self.generate_report)
        workflow.add_edge(START, "generate_report")
        workflow.add_edge("generate_report", END)
        return workflow.compile()
__all__ = ["ReportingAgent"]
