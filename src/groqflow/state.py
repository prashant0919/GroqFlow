"""LangGraph conversation state schema.

Defines the typed state dictionary shared across all graph nodes. The ``messages``
field uses LangGraph's ``add_messages`` reducer to accumulate messages across nodes
within a single graph invocation.
"""

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Attributes:
        messages: Conversation message history with automatic accumulation.
        context: Retrieved document context from the FAISS knowledge base.
        choice: Classifier output indicating the selected intent category.
        user_details: Extracted booking details (partial or complete).
        booking_confirmed: Whether all required booking fields have been collected.
        intent_route: The name of the node that handled the current request.
        source_filter: List of source file paths to restrict retrieval scope.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    choice: str
    user_details: dict[str, Any] | None
    booking_confirmed: bool
    intent_route: str
    source_filter: list[str] | None
