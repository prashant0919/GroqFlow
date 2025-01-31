"""Off-topic and unsupported request handler.

Provides polite responses for queries that fall outside the chatbot's
domain (off-topic) or capabilities (unsupported). Selected randomly
to avoid repetitive responses.
"""

import random

from langchain_core.messages import AIMessage

from groqflow.agents.base import BaseAgent
from groqflow.state import AgentState


class ApologyResponder(BaseAgent):
    """Handles off-topic (category 0) and unclassified requests."""

    intent_route = "apology"

    OFF_TOPIC_MESSAGES = [
        "I'm here to help with questions about our business, products, and appointments. How can I assist you today?",
        "That's a bit outside what I can help with. Feel free to ask about our services or book an appointment!",
        "I specialize in answering questions about our company and scheduling appointments. What would you like to know?",
    ]

    UNSUPPORTED_MESSAGES = [
        "I'm sorry, but I can't assist with that request right now. Is there anything else I can help you with?",
        "I apologize, but that topic is outside my current capabilities. Feel free to ask about our services or book an appointment.",
        "Unfortunately, I can't help with that particular inquiry. I'm here for company questions and scheduling.",
    ]

    def __call__(self, state: AgentState) -> AgentState:
        """Select and return an appropriate fallback response.

        Uses ``OFF_TOPIC_MESSAGES`` for category ``"0"`` (explicitly off-topic)
        and ``UNSUPPORTED_MESSAGES`` for all other unhandled categories.
        """
        choice = state.get("choice")
        category = ""
        if choice is not None and hasattr(choice, "content"):
            category = choice.content.strip()

        messages = (
            self.OFF_TOPIC_MESSAGES if category == "0" else self.UNSUPPORTED_MESSAGES
        )
        return self._with_intent(
            state,
            messages=[AIMessage(content=random.choice(messages))],
            intent_route=self.intent_route,
        )
