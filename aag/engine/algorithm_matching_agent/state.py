"""State definitions for Algorithm Matching Agent."""

from typing import Any, Dict, List, Optional, TypedDict
from typing_extensions import Annotated


def merge_node_results(
    existing: Dict[str, Dict[str, Any]],
    new: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Reducer function: merge single-node results from parallel branches."""
    result = existing.copy() if existing else {}
    if new:
        result.update(new)
    return result


class NodeMatchingState(TypedDict):
    """Local workflow state for matching algorithms on one DAG node."""

    node_id: str  # Node ID (e.g., "q1")
    question: str  # Natural language question for the current node
    dataset_schema: Dict[str, Any]  # Graph database schema information

    intent_type: Optional[str]  # Intent classification: "graph_algorithm", "graph_query", "numeric_analysis"

    retrieved_task_types: List[Dict[str, Any]]  # Retrieved candidate task types
    retrieved_algorithms: List[Dict[str, Any]]  # Retrieved candidate algorithm details

    selected_task_type_id: Optional[str]  # Final Task Type ID selected by the LLM
    selected_algorithm_id: Optional[str]  # Final Algorithm ID selected by the LLM

    validation_error: str  # Error message when algorithm validation fails
    retry_count: int  # Retry count for the current node


class MatchingOrchestratorState(TypedDict):
    """Global orchestrator state for the algorithm matching agent."""

    dag_payload: Dict[str, Any]  # Original DAG payload containing all subqueries
    dataset_schema: Dict[str, Any]  # Global graph dataset schema information

    # Aggregated parallel results, e.g.:
    # {"q1": {"task_type": "...", "algorithm": "..."}, ...}
    matched_nodes: Annotated[Dict[str, Dict[str, Any]], merge_node_results]

    global_error: str  # Global-level error record
