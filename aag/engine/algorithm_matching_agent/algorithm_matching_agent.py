"""LangGraph-based Algorithm Matching Agent using a Map-Reduce workflow."""

import copy
import logging
from typing import Any, Dict, List, Union

try:
    from langgraph.graph import END, START, Send, StateGraph
except Exception:  # pragma: no cover - compatibility fallback
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

from .state import (
    MatchingOrchestratorState,
    NodeMatchingState,
)
from aag.expert_search_engine.search import ExpertSearchEngine
from aag.reasoner.model_deployment import Reasoner

logger = logging.getLogger(__name__)


class AlgorithmMatchingAgent:
    """Algorithm matching agent with parallel node workers and reduce aggregation."""

    def __init__(
        self,
        reasoner: Reasoner,
        expert_search_engine: ExpertSearchEngine,
        max_retry: int = 3,
    ) -> None:
        if reasoner is None:
            raise ValueError("AlgorithmMatchingAgent requires a valid reasoner")
        if expert_search_engine is None:
            raise ValueError("AlgorithmMatchingAgent requires a valid expert_search_engine")

        self.reasoner = reasoner
        self.expert_search_engine = expert_search_engine
        self.max_retry = max_retry

        self._node_matching_subgraph_app = self._compile_node_matching_subgraph()

    # Node Matching Sub-graph (Worker)
    def classify_intent(self, state: NodeMatchingState) -> Dict[str, Any]:
        """Classify question intent for one DAG node."""

        question = (state.get("question") or "").strip()
        if not question:
            return {
                "intent_type": "graph_algorithm",
                "validation_error": "question is empty",
            }

        try:
            result = self.reasoner.classify_question_type(question) or {}
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Intent classification failed for node %s", state.get("node_id"))
            return {
                "intent_type": "graph_algorithm",
                "validation_error": f"classify_intent failed: {exc}",
            }

        if isinstance(result, dict) and result.get("error"):
            return {
                "intent_type": "graph_algorithm",
                "validation_error": str(result.get("error")),
            }

        intent_type = str((result or {}).get("type", "graph_algorithm") or "graph_algorithm").strip()
        if not intent_type:
            intent_type = "graph_algorithm"

        return {"intent_type": intent_type, "validation_error": ""}

    def route_after_classify(self, state: NodeMatchingState) -> str:
        """Route by intent type after classification."""

        intent_type = (state.get("intent_type") or "").strip().lower()
        if intent_type in {"graph_query", "numeric_analysis"}:
            return "END"
        return "retrieve_knowledge"

    def retrieve_knowledge(self, state: NodeMatchingState) -> Dict[str, Any]:
        """Retrieve candidate task types and algorithms from ExpertSearchEngine."""

        question = (state.get("question") or "").strip()
        if not question:
            return {
                "retrieved_task_types": [],
                "retrieved_algorithms": [],
                "selected_task_type_id": None,
                "validation_error": "question is empty",
            }

        try:
            task_type_list = self.expert_search_engine.retrieve_task_type(question) or []
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Task type retrieval failed for node %s", state.get("node_id"))
            return {
                "retrieved_task_types": [],
                "retrieved_algorithms": [],
                "selected_task_type_id": None,
                "validation_error": f"retrieve_task_type failed: {exc}",
            }

        if not isinstance(task_type_list, list):
            task_type_list = []

        selected_task_type_id = None
        try:
            task_type_result = self.reasoner.select_task_type(question, task_type_list) or {}
            if isinstance(task_type_result, dict) and not task_type_result.get("error"):
                selected_task_type_id = task_type_result.get("id")
            elif isinstance(task_type_result, dict) and task_type_result.get("error"):
                return {
                    "retrieved_task_types": task_type_list,
                    "retrieved_algorithms": [],
                    "selected_task_type_id": None,
                    "validation_error": str(task_type_result.get("error")),
                }
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Task type selection failed for node %s", state.get("node_id"))
            return {
                "retrieved_task_types": task_type_list,
                "retrieved_algorithms": [],
                "selected_task_type_id": None,
                "validation_error": f"select_task_type failed: {exc}",
            }

        algorithm_list: List[Dict[str, Any]] = []
        if selected_task_type_id:
            try:
                algorithm_list = (
                    self.expert_search_engine.retrieve_algorithm(question, selected_task_type_id) or []
                )
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                logger.exception("Algorithm retrieval failed for node %s", state.get("node_id"))
                return {
                    "retrieved_task_types": task_type_list,
                    "retrieved_algorithms": [],
                    "selected_task_type_id": selected_task_type_id,
                    "validation_error": f"retrieve_algorithm failed: {exc}",
                }

        if not isinstance(algorithm_list, list):
            algorithm_list = []

        return {
            "retrieved_task_types": task_type_list,
            "retrieved_algorithms": algorithm_list,
            "selected_task_type_id": selected_task_type_id,
            "validation_error": "",
        }

    def select_algorithm(self, state: NodeMatchingState) -> Dict[str, Any]:
        """Use Reasoner to select the best algorithm from retrieved candidates."""

        question = (state.get("question") or "").strip()
        algorithm_list = state.get("retrieved_algorithms") or []
        dataset_schema = state.get("dataset_schema") or {}

        if not algorithm_list:
            return {
                "selected_algorithm_id": None,
                "validation_error": "No candidate algorithms were retrieved",
            }

        try:
            algorithm_result = self.reasoner.select_algorithm(
                question,
                algorithm_list,
                graph_schema=dataset_schema,
            ) or {}
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logger.exception("Algorithm selection failed for node %s", state.get("node_id"))
            return {
                "selected_algorithm_id": None,
                "validation_error": f"select_algorithm failed: {exc}",
            }

        if isinstance(algorithm_result, dict) and algorithm_result.get("error"):
            return {
                "selected_algorithm_id": None,
                "validation_error": str(algorithm_result.get("error")),
            }

        selected_algorithm_id = (algorithm_result or {}).get("id")
        if not selected_algorithm_id:
            return {
                "selected_algorithm_id": None,
                "validation_error": "Algorithm selection result missing 'id'",
            }

        return {
            "selected_algorithm_id": selected_algorithm_id,
            "validation_error": "",
        }

    def validate_algorithm(self, state: NodeMatchingState) -> Dict[str, Any]:
        """Validate selected algorithm and manage retry count."""

        selected_algorithm_id = state.get("selected_algorithm_id")
        retry_count = int(state.get("retry_count", 0) or 0)
        algo_index = getattr(self.expert_search_engine, "algo_index", {}) or {}

        if not selected_algorithm_id:
            return {
                "validation_error": "selected_algorithm_id is empty",
                "retry_count": retry_count + 1,
            }

        if selected_algorithm_id not in algo_index:
            return {
                "validation_error": f"selected algorithm not found in algo_index: {selected_algorithm_id}",
                "retry_count": retry_count + 1,
            }

        return {"validation_error": ""}

    def route_after_validate(self, state: NodeMatchingState) -> str:
        """Retry selection when validation fails and retry quota remains."""

        validation_error = (state.get("validation_error") or "").strip()
        retry_count = int(state.get("retry_count", 0) or 0)

        if validation_error and retry_count < self.max_retry:
            return "select_algorithm"
        return "END"

    def _compile_node_matching_subgraph(self) -> Any:
        """Compile worker sub-graph for matching one DAG node."""

        workflow = StateGraph(NodeMatchingState)

        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("retrieve_knowledge", self.retrieve_knowledge)
        workflow.add_node("select_algorithm", self.select_algorithm)
        workflow.add_node("validate_algorithm", self.validate_algorithm)

        workflow.add_edge(START, "classify_intent")
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_after_classify,
            {
                "retrieve_knowledge": "retrieve_knowledge",
                "END": END,
            },
        )

        workflow.add_edge("retrieve_knowledge", "select_algorithm")
        workflow.add_edge("select_algorithm", "validate_algorithm")
        workflow.add_conditional_edges(
            "validate_algorithm",
            self.route_after_validate,
            {
                "select_algorithm": "select_algorithm",
                "END": END,
            },
        )

        return workflow.compile()

    # Orchestrator (Map-Reduce)
    def _map_entry(self, state: MatchingOrchestratorState) -> Dict[str, Any]:
        """Entry node for the map phase; keeps state unchanged."""

        return {}

    def distribute_nodes(
        self,
        state: MatchingOrchestratorState,
    ) -> Union[List[Send], str]:
        """Map stage: fan out one worker per subquery via Send API."""

        dag_payload = state.get("dag_payload") or {}
        subquery_plan = dag_payload.get("subquery_plan") or {}
        subqueries = subquery_plan.get("subqueries") or []

        if not isinstance(subqueries, list) or not subqueries:
            return "aggregate_results"

        sends: List[Send] = []
        dataset_schema = state.get("dataset_schema") or {}

        for subquery in subqueries:
            if not isinstance(subquery, dict):
                continue

            node_id = str(subquery.get("id") or "").strip()
            question = str(subquery.get("query") or "").strip()
            if not node_id or not question:
                continue

            sends.append(
                Send(
                    "node_matching_subgraph",
                    {
                        "node_id": node_id,
                        "question": question,
                        "dataset_schema": dataset_schema,
                        "intent_type": None,
                        "retrieved_task_types": [],
                        "retrieved_algorithms": [],
                        "selected_task_type_id": None,
                        "selected_algorithm_id": None,
                        "validation_error": "",
                        "retry_count": 0,
                    },
                )
            )

        if not sends:
            return "aggregate_results"

        return sends

    def node_matching_subgraph(self, state: NodeMatchingState) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Run worker sub-graph for one node and return reducer-ready payload."""

        final_state = self._node_matching_subgraph_app.invoke(state)

        node_id = str(final_state.get("node_id") or state.get("node_id") or "").strip()
        intent_type = str(final_state.get("intent_type") or "").strip().lower()

        if intent_type in {"graph_query", "numeric_analysis"}:
            task_type_value = intent_type
            algorithm_value = None
        else:
            task_type_value = final_state.get("selected_task_type_id")
            algorithm_value = final_state.get("selected_algorithm_id")

        if not node_id:
            return {"matched_nodes": {}}

        return {
            "matched_nodes": {
                node_id: {
                    "task_type": task_type_value,
                    "algorithm": algorithm_value,
                    "intent_type": final_state.get("intent_type"),
                    "validation_error": final_state.get("validation_error", ""),
                }
            }
        }

    def aggregate_results(self, state: MatchingOrchestratorState) -> Dict[str, Any]:
        """Reduce stage: write matched task_type/algorithm back to dag_payload."""

        dag_payload = copy.deepcopy(state.get("dag_payload") or {})
        matched_nodes = state.get("matched_nodes") or {}

        # Update subquery entries by node id (q1, q2, ...).
        subquery_plan = dag_payload.get("subquery_plan") or {}
        subqueries = subquery_plan.get("subqueries")
        if isinstance(subqueries, list):
            for subquery in subqueries:
                if not isinstance(subquery, dict):
                    continue
                node_id = str(subquery.get("id") or "").strip()
                if not node_id:
                    continue

                matched = matched_nodes.get(node_id) or {}
                if not isinstance(matched, dict) or not matched:
                    continue

                subquery["task_type"] = matched.get("task_type")
                subquery["algorithm"] = matched.get("algorithm")

        # Keep steps section in sync when possible (keys often look like "1", "2", ...).
        steps = dag_payload.get("steps")
        if isinstance(steps, dict):
            for step_key, step_value in steps.items():
                if not isinstance(step_value, dict):
                    continue
                step_key_str = str(step_key)
                candidate_node_id = f"q{step_key_str}" if step_key_str.isdigit() else step_key_str
                matched = matched_nodes.get(candidate_node_id) or {}
                if not isinstance(matched, dict) or not matched:
                    continue

                step_value["task_type"] = matched.get("task_type")
                step_value["algorithm"] = matched.get("algorithm")

        return {"dag_payload": dag_payload}

    def compile(self) -> Any:
        """Compile the orchestrator app."""

        workflow = StateGraph(MatchingOrchestratorState)

        workflow.add_node("distribute_nodes", self._map_entry)
        workflow.add_node("node_matching_subgraph", self.node_matching_subgraph)
        workflow.add_node("aggregate_results", self.aggregate_results)

        workflow.add_edge(START, "distribute_nodes")
        workflow.add_conditional_edges(
            "distribute_nodes",
            self.distribute_nodes,
            {
                "aggregate_results": "aggregate_results",
            },
        )

        # All parallel workers converge into one reduce node.
        workflow.add_edge("node_matching_subgraph", "aggregate_results")
        workflow.add_edge("aggregate_results", END)

        return workflow.compile()


__all__ = ["AlgorithmMatchingAgent"]
