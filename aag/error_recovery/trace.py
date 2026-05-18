from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time


@dataclass
class PromptTraceItem:
    fn: str
    base_prompt: str
    meta: Dict[str, Any]
    ts: float


class PromptTraceBuffer:
    """
    Records the rendered base_prompt at runtime for:
    - debug / observability
    - on retry, looking up the latest base_prompt by fn_name and building an enhanced_prompt with error context
    """
    def __init__(self, maxlen: int = 200):
        self.maxlen = maxlen
        self.items: List[PromptTraceItem] = []

    def record(self, fn: str, base_prompt: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if not base_prompt:
            return
        self.items.append(
            PromptTraceItem(fn=fn, base_prompt=base_prompt, meta=meta or {}, ts=time.time())
        )
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen:]

    def last_prompt(self, fn: str) -> Optional[str]:
        for it in reversed(self.items):
            if it.fn == fn:
                return it.base_prompt
        return None

    def last_item(self, fn: str) -> Optional[PromptTraceItem]:
        for it in reversed(self.items):
            if it.fn == fn:
                return it
        return None

    def clear(self) -> None:
        self.items.clear()