"""Base agent contract for LangGraph nodes.

All graph nodes inherit from ``BaseAgent`` and use ``_with_intent()`` to
tag their responses with the intent route that handled the request.
"""

from __future__ import annotations

from abc import ABC
from typing import Any

from groqflow.state import AgentState


class BaseAgent(ABC):
    """Abstract base for every GroqFlow graph node.

    Subclasses must set ``intent_route`` to a descriptive string (e.g.,
    ``"greeter"``, ``"rag"``, ``"booking"``) and implement ``__call__``.
    """

    intent_route: str = "unknown"

    def _with_intent(self, state: AgentState, **updates: Any) -> AgentState:
        """Merge updates into the state and tag with the node's intent route.

        Args:
            state: The current agent state.
            **updates: Key-value pairs to merge into the state.

        Returns:
            A new state dict with updates applied and ``intent_route`` set.
        """
        merged = dict(state)
        merged.update(updates)
        merged["intent_route"] = updates.get("intent_route", self.intent_route)
        return merged
