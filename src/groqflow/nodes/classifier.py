"""Intent classification node for the LangGraph agent.

Routes user messages to the appropriate handler:
- Category 1: Greeting
- Category 2: Knowledge retrieval (RAG)
- Category 3: Appointment booking
- Category 0: Off-topic / unsupported

When knowledge files are selected (``source_filter`` is set), a fast-path
regex classifier is used to avoid an unnecessary LLM call.
"""

import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from groqflow.agents.base import BaseAgent
from groqflow.config_loader import TenantConfig
from groqflow.prompts.classification_prompt import classification_prompt_for
from groqflow.state import AgentState


class ClassifierNode(BaseAgent):
    """Classifies user intent and routes to the appropriate graph branch.

    When ``source_filter`` is present in the state, uses deterministic regex
    matching for greetings and booking keywords, defaulting to knowledge
    retrieval. This reduces latency by ~200ms per query.
    """

    intent_route = "classifier"

    def __init__(self, llm: BaseChatModel, tenant: TenantConfig) -> None:
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Expected an instance of BaseChatModel for `llm`.")
        self.chain = classification_prompt_for(tenant) | llm

    def __call__(self, state: AgentState) -> AgentState:
        """Classify the latest message and return the state with a ``choice``.

        Args:
            state: Must contain at least one message in ``messages``.

        Returns:
            Updated state with ``choice`` set to an ``AIMessage`` containing
            the category number.

        Raises:
            ValueError: If ``messages`` is empty.
        """
        if not state.get("messages"):
            raise ValueError("State must contain at least one message.")

        last_message = state["messages"][-1].content
        normalized_message = last_message.strip().lower()
        has_selected_sources = state.get("source_filter") is not None
        if has_selected_sources:
            if re.fullmatch(
                r"(?:hi|hello|hey|good morning|good afternoon|good evening)[!.?]*",
                normalized_message,
            ):
                category = "1"
            elif re.search(
                r"\b(?:book|schedule|reschedule|cancel|reserve)\b.{0,40}"
                r"\b(?:appointment|visit|consultation|meeting|reservation|slot)\b|"
                r"\b(?:appointment|visit|consultation|meeting|reservation|slot)\b"
                r".{0,40}\b(?:book|schedule|reschedule|cancel|reserve)\b",
                normalized_message,
            ):
                category = "3"
            else:
                category = "2"
            choice = AIMessage(content=category)
        else:
            choice = self.chain.invoke({"user_query": last_message})
        return self._with_intent(state, choice=choice, intent_route=self.intent_route)
