"""Context-grounded RAG response synthesizer.

Takes the retrieved document context from the ``RetrieverNode`` and generates
a grounded answer using the LLM. If no context is available, responds with a
helpful fallback suggesting an appointment.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from groqflow.agents.base import BaseAgent
from groqflow.prompts.rag_prompt import rag_booking_prompt
from groqflow.state import AgentState


class RagNode(BaseAgent):
    """Synthesizes a response grounded in retrieved document context."""

    intent_route = "rag"

    def __init__(self, llm: BaseChatModel) -> None:
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Expected an instance of BaseChatModel for `llm`.")
        self.chain = rag_booking_prompt | llm

    def __call__(self, state: AgentState) -> AgentState:
        """Generate a context-grounded response.

        If ``context`` is empty or whitespace, returns a fallback message
        instead of invoking the LLM with no grounding material.

        Args:
            state: Must contain ``messages`` and ``context``.

        Returns:
            Updated state with the synthesized response appended to ``messages``.

        Raises:
            ValueError: If ``messages`` is empty.
        """
        if not state.get("messages"):
            raise ValueError("State must contain at least one message.")

        context = state.get("context", "")
        if not context.strip():
            return self._with_intent(
                state,
                messages=[
                    AIMessage(
                        content=(
                            "I don't have enough information in my knowledge base to "
                            "answer that question. Would you like to schedule an "
                            "appointment so our team can help you directly?"
                        )
                    )
                ],
                intent_route=self.intent_route,
            )

        response = self.chain.invoke(
            {
                "user_query": state["messages"][-1].content,
                "context": context,
            }
        )
        return self._with_intent(
            state,
            messages=[AIMessage(content=response.content)],
            intent_route=self.intent_route,
        )
