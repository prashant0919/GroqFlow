"""FAISS retrieval node with source filtering.

Queries the tenant's knowledge base for document chunks relevant to the
user's message. When ``source_filter`` is set in the state, retrieval is
restricted to chunks from the selected files only.
"""

from groqflow.agents.base import BaseAgent
from groqflow.knowledge_base import KnowledgeBase
from groqflow.state import AgentState


class RetrieverNode(BaseAgent):
    """Retrieves relevant document chunks and stores them in ``context``."""

    intent_route = "retriever"

    def __init__(self, knowledge_base: KnowledgeBase, retrieval_k: int) -> None:
        self.knowledge_base = knowledge_base
        self.retrieval_k = retrieval_k

    def __call__(self, state: AgentState) -> AgentState:
        """Query the FAISS index and populate the ``context`` field.

        Args:
            state: Must contain at least one message. Optionally contains
                ``source_filter`` to restrict retrieval scope.

        Returns:
            Updated state with ``context`` set to concatenated chunk texts.

        Raises:
            ValueError: If ``messages`` is empty.
        """
        if not state.get("messages"):
            raise ValueError("State must contain at least one message.")

        query = state["messages"][-1].content
        context = self.knowledge_base.retrieve(
            query,
            k=self.retrieval_k,
            sources=state.get("source_filter"),
        )
        return self._with_intent(
            state,
            context="\n\n".join(context) if context else "",
            intent_route=self.intent_route,
        )
