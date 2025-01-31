"""Greeting response node.

Generates a context-appropriate greeting using the tenant's name and
description as prompt context.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from groqflow.agents.base import BaseAgent
from groqflow.config_loader import TenantConfig
from groqflow.prompts.greeting_prompt import greeting_prompt_for
from groqflow.state import AgentState


class GreeterNode(BaseAgent):
    """Produces a tenant-branded greeting response via the LLM."""

    intent_route = "greeter"

    def __init__(self, llm: BaseChatModel, tenant: TenantConfig) -> None:
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Expected an instance of BaseChatModel for `llm`.")
        self.llm = llm
        self.tenant = tenant

    def __call__(self, state: AgentState) -> AgentState:
        """Generate a greeting and return the updated state."""
        response = self.llm.invoke(greeting_prompt_for(self.tenant))
        return self._with_intent(
            state,
            messages=[AIMessage(content=response.content)],
            intent_route=self.intent_route,
        )
