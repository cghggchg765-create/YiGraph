from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class RetryPolicy:
    max_attempts: int = 3          # Total attempts (including the first)
    max_error_history: int = 2     # Max entries retained in error_history
    keep_last_k_in_prompt: int = 2 # Number of errors injected into the prompt
    extra_constraints: Optional[str] = None


DEFAULT_POLICY = RetryPolicy()

OPERATION_POLICIES: Dict[str, RetryPolicy] = {
    # "parameter_extraction": RetryPolicy(extra_constraints="Return JSON only."),
    # "dependency_resolution": RetryPolicy(extra_constraints="Return JSON only."),
    # "code_generation": RetryPolicy(extra_constraints="Return python code only."),
}


def get_policy(operation_type: str) -> RetryPolicy:
    return OPERATION_POLICIES.get(operation_type, DEFAULT_POLICY)