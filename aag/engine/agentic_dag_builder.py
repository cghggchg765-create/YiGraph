"""LangGraph DAG builder sub-graph for AAG.

This module converts a user query into a validated GraphWorkflowDAG by using
existing AAG prompt templates and GraphWorkflowDAG native build logic.
"""

import importlib
import json
from typing import Any, Dict, Optional, TypedDict, Union

from aag.models.graph_workflow_dag import GraphWorkflowDAG

try:
    from aag.reasoner.prompt_template.llm_prompt_en import (
        expert_subqueries_with_algorithms_prompt_en,
        rewrite_query_prompt_en,
    )
except Exception:
    from aag.reasoner.prompt_template.llm_prompt_zh import (
        expert_subqueries_with_algorithms_prompt_zh as expert_subqueries_with_algorithms_prompt_en,
    )
    from aag.reasoner.prompt_template.llm_prompt_zh import (
        rewrite_query_prompt_zh as rewrite_query_prompt_en,
    )


_BASE_LLM: Optional[Any] = None


class DAGBuilderState(TypedDict):
    """Global state for DAG builder StateGraph."""

    original_query: str
    algorithm_library_info: str
    dataset_info: str
    rewritten_query: str
    subquery_plan_json: Dict[str, Any]
    dag_payload: Dict[str, Any]
    validation_errors: str
    retry_count: int


def set_dag_builder_llm(llm: Any) -> None:
    """Set shared LLM instance for DAG builder nodes.

    Args:
        llm: A LangChain-compatible chat model or callable wrapper.
    """

    global _BASE_LLM
    _BASE_LLM = llm


def build_langchain_llm_from_reasoner_config(reasoner_config: Any) -> Any:
    """Build a LangChain chat model from existing Reasoner config.

    Args:
        reasoner_config: Config object containing llm provider settings.

    Returns:
        A LangChain-compatible chat model instance.

    Raises:
        RuntimeError: If required dependencies are missing.
        ValueError: If provider is unsupported or required config is missing.
    """

    llm_cfg = getattr(reasoner_config, "llm", None)
    if llm_cfg is None:
        raise ValueError("Invalid reasoner config: missing llm section")

    provider = str(getattr(llm_cfg, "provider", "ollama") or "ollama").lower()

    if provider == "ollama":
        ollama_cfg = getattr(llm_cfg, "ollama", {}) or {}
        model_name = ollama_cfg.get("model_name")
        if not model_name:
            raise ValueError("Ollama provider requires reasoner.llm.ollama.model_name")

        try:
            langchain_ollama = importlib.import_module("langchain_ollama")
            chat_ollama_cls = getattr(langchain_ollama, "ChatOllama")
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency langchain-ollama. Install it in your active environment."
            ) from exc

        base_url = f"http://localhost:{ollama_cfg.get('port', 11434)}"
        return chat_ollama_cls(
            model=model_name,
            base_url=base_url,
            temperature=0,
        )

    if provider == "openai":
        openai_cfg = getattr(llm_cfg, "openai", {}) or {}
        api_key = openai_cfg.get("api_key")
        if not api_key:
            raise ValueError("OpenAI provider requires reasoner.llm.openai.api_key")

        try:
            langchain_openai = importlib.import_module("langchain_openai")
            chat_openai_cls = getattr(langchain_openai, "ChatOpenAI")
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency langchain-openai. Install it in your active environment."
            ) from exc

        return chat_openai_cls(
            model=openai_cfg.get("model", "gpt-4o"),
            api_key=api_key,
            base_url=openai_cfg.get("base_url") or None,
            temperature=float(openai_cfg.get("temperature", 0.0)),
        )

    raise ValueError(f"Unsupported llm provider for DAG builder: {provider}")


def _to_text_response(raw: Any) -> str:
    """Normalize an LLM response object into plain text."""

    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    if hasattr(raw, "content"):
        content = getattr(raw, "content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    return str(raw).strip()


def _invoke_text_llm(prompt_text: str) -> str:
    """Invoke configured LLM and return text output."""

    if _BASE_LLM is None:
        return ""

    try:
        if hasattr(_BASE_LLM, "invoke"):
            return _to_text_response(_BASE_LLM.invoke(prompt_text))
        if callable(_BASE_LLM):
            return _to_text_response(_BASE_LLM(prompt_text))
    except Exception:
        return ""

    return ""


def _extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Extract and parse a JSON object from model output.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed JSON object. Returns empty dict on failure.
    """

    if not raw_text:
        return {}

    candidate = raw_text.strip()
    fence = chr(96) * 3
    if fence in candidate:
        candidate = candidate.replace(f"{fence}json", "").replace(fence, "").strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def init_dag_builder_state(
    query: str,
    algorithm_library_info: str,
    dataset_info: str,
) -> DAGBuilderState:
    """Initialize DAG builder state.

    Args:
        query: Original user query.
        algorithm_library_info: Serialized algorithm library information.
        dataset_info: Serialized dataset/schema information.

    Returns:
        Initialized DAGBuilderState.
    """

    return {
        "original_query": query,
        "algorithm_library_info": algorithm_library_info,
        "dataset_info": dataset_info,
        "rewritten_query": "",
        "subquery_plan_json": {},
        "dag_payload": {},
        "validation_errors": "",
        "retry_count": 0,
    }


def rewrite_node(state: DAGBuilderState) -> Dict[str, str]:
    """Rewrite original query using existing rewrite prompt template.

    Args:
        state: Current DAG builder state.

    Returns:
        State update with rewritten_query.
    """

    original_query = state.get("original_query", "")
    if not original_query:
        return {"rewritten_query": ""}

    prompt_text = rewrite_query_prompt_en.format(
        original_query=original_query,
        algorithm_library_info=state.get("algorithm_library_info", ""),
        dataset_info=state.get("dataset_info", ""),
    )
    raw_output = _invoke_text_llm(prompt_text)
    payload = _extract_json_object(raw_output)

    rewritten_query = str(payload.get("rewritten_query", "")).strip()
    if not rewritten_query:
        rewritten_query = original_query

    return {"rewritten_query": rewritten_query}


def plan_node(state: DAGBuilderState) -> Dict[str, Dict[str, Any]]:
    """Generate subquery plan JSON using existing expert planning prompt.

    Args:
        state: Current DAG builder state.

    Returns:
        State update with subquery_plan_json.
    """

    rewritten_query = state.get("rewritten_query", "")
    prompt_text = expert_subqueries_with_algorithms_prompt_en.format(
        expert_instruction=rewritten_query,
        algorithm_library_info=state.get("algorithm_library_info", ""),
        dataset_info=state.get("dataset_info", ""),
    )

    validation_errors = state.get("validation_errors", "")
    if validation_errors:
        prompt_text += (
            "\n\nNote: The subquery plan you generated last time caused the following error, "
            "please fix it:\n" + validation_errors
        )

    raw_output = _invoke_text_llm(prompt_text)
    subquery_plan_json = _extract_json_object(raw_output)

    return {"subquery_plan_json": subquery_plan_json}


def validate_and_build_node(state: DAGBuilderState) -> Dict[str, Any]:
    """Build and validate DAG by using GraphWorkflowDAG native method.

    Args:
        state: Current DAG builder state.

    Returns:
        Success: dag_payload and empty validation_errors.
        Failure: validation_errors and incremented retry_count.
    """

    dag = GraphWorkflowDAG()

    try:
        dag.build_from_subquery_plan(state["subquery_plan_json"])
        dag_payload = dag.get_dag_info()
        if not isinstance(dag_payload, dict):
            raise ValueError("dag.get_dag_info() must return a dict")
        return {"dag_payload": dag_payload, "validation_errors": ""}
    except Exception as exc:
        return {
            "validation_errors": str(exc),
            "retry_count": state.get("retry_count", 0) + 1,
        }


def route_after_validation(state: DAGBuilderState) -> str:
    """Route control flow after validation/build node.

    Args:
        state: Current DAG builder state.

    Returns:
        END when no error, or plan_node when retry is still allowed.

    Raises:
        ValueError: When retry limit is reached.
    """

    validation_errors = state.get("validation_errors", "")
    retry_count = state.get("retry_count", 0)

    if not validation_errors:
        return "END"

    if retry_count < 3:
        return "plan_node"

    raise ValueError(
        f"DAG builder reached max retries ({retry_count}). Last error: {validation_errors}"
    )


def _compile_dag_builder_app() -> Any:
    """Compile DAG builder StateGraph app."""

    try:
        langgraph_graph = importlib.import_module("langgraph.graph")
        END = getattr(langgraph_graph, "END")
        START = getattr(langgraph_graph, "START")
        StateGraph = getattr(langgraph_graph, "StateGraph")
    except Exception as exc:
        raise RuntimeError("LangGraph is not available in current environment") from exc

    workflow = StateGraph(DAGBuilderState)

    workflow.add_node("rewrite_node", rewrite_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("validate_and_build_node", validate_and_build_node)

    workflow.add_edge(START, "rewrite_node")
    workflow.add_edge("rewrite_node", "plan_node")
    workflow.add_edge("plan_node", "validate_and_build_node")

    workflow.add_conditional_edges(
        "validate_and_build_node",
        route_after_validation,
        {
            "plan_node": "plan_node",
            "END": END,
        },
    )

    return workflow.compile()


dag_builder_app: Optional[Any]
try:
    dag_builder_app = _compile_dag_builder_app()
except Exception:
    dag_builder_app = None


def run_agentic_dag_builder(
    question: str,
    available_tools: Union[list, str],
    dataset_info: str = "",
) -> GraphWorkflowDAG:
    """Run DAG builder pipeline and return generated GraphWorkflowDAG.

    The internal LangGraph state remains JSON-serializable and only passes
    `dag_payload` between nodes. The final DAG object is reconstructed from
    payload for backward compatibility with existing scheduler code.

    Args:
        question: User input question.
        available_tools: Tool library metadata list or pre-serialized string.
        dataset_info: Optional dataset/schema information string.

    Returns:
        Generated GraphWorkflowDAG instance.

    Raises:
        ValueError: If DAG generation fails.
    """

    global dag_builder_app

    if dag_builder_app is None:
        dag_builder_app = _compile_dag_builder_app()

    if isinstance(available_tools, str):
        algorithm_library_info = available_tools
    else:
        algorithm_library_info = json.dumps(available_tools, ensure_ascii=False, indent=2)

    initial_state = init_dag_builder_state(
        query=question,
        algorithm_library_info=algorithm_library_info,
        dataset_info=dataset_info,
    )
    final_state = dag_builder_app.invoke(initial_state)

    dag_payload = final_state.get("dag_payload")
    if not isinstance(dag_payload, dict) or not dag_payload:
        raise ValueError(
            "DAG generation failed: dag_payload is empty; "
            f"validation_errors={final_state.get('validation_errors', '')}"
        )

    subquery_plan = dag_payload.get("subquery_plan")
    if not isinstance(subquery_plan, dict):
        raise ValueError("DAG generation failed: dag_payload.subquery_plan is invalid")

    generated_dag = GraphWorkflowDAG()
    generated_dag.build_from_subquery_plan(subquery_plan)

    return generated_dag
