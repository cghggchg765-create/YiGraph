"""Global workflow state definition for multi-agent orchestration."""

from typing import Any, Dict, Optional, TypedDict


class WorkflowGlobalState(TypedDict):
   
    original_query: str
    dataset_name: Optional[str]
    dataset_schema: Dict[str, Any]
    algorithm_library: str

    dag_payload: Dict[str, Any]

    final_answer: Optional[str]
    global_error: str
