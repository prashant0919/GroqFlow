"""Slot-filling booking node with multi-turn detail extraction.

Implements a finite state machine for appointment booking:
1. Initial turn: Prompts for all required fields
2. Subsequent turns: Extracts provided fields, merges with existing state,
   prompts for remaining missing fields
3. Completion: Generates confirmation message and sets ``booking_confirmed``

The required fields are defined per tenant in the YAML configuration and
compiled into a Pydantic model at load time.
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from groqflow.agents.base import BaseAgent
from groqflow.config_loader import TenantConfig, get_user_details_model
from groqflow.prompts.detail_collection_prompt import booking_prompts
from groqflow.state import AgentState


class BookingNode(BaseAgent):
    """Multi-turn appointment booking with dynamic slot filling.

    Uses the tenant's Pydantic-generated ``UserDetails`` model to determine
    which fields are required and which have been provided. Extraction is
    performed by the LLM via a JSON output parser.
    """

    intent_route = "booking"

    def __init__(self, llm: BaseChatModel, tenant: TenantConfig) -> None:
        if not isinstance(llm, BaseChatModel):
            raise TypeError("Expected an instance of BaseChatModel for `llm`.")
        self.llm = llm
        self.user_details_model = get_user_details_model(tenant)
        (
            self.detail_inquiry_prompt,
            self.extraction_prompt,
            self.inform_missing_prompt,
            self.json_parser,
        ) = booking_prompts(self.user_details_model)

    def _get_missing_fields(self, details: dict[str, Any]) -> list[str]:
        """Return human-readable names of required fields that are still empty."""
        missing = []
        for field in self.user_details_model.get_required_field_names():
            value = details.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field.replace("_", " "))
        return missing

    def _merge_details(
        self, existing: dict[str, Any], extracted: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge newly extracted details into existing state, preserving non-empty values."""
        merged = dict(existing)
        for key, value in extracted.items():
            if value is not None and (not isinstance(value, str) or value.strip()):
                merged[key] = value
        return merged

    def _extract_details(self, user_input: str) -> dict[str, Any]:
        """Extract booking details from user input using the LLM and JSON parser."""
        chain = self.extraction_prompt | self.llm | self.json_parser
        return chain.invoke(
            {
                "required_fields": self.user_details_model.get_required_fields(),
                "user_input": user_input,
            }
        )

    def __call__(self, state: AgentState) -> AgentState:
        """Process a booking turn: extract details, check completeness, respond.

        State machine transitions:
        - Already confirmed → return "already confirmed" message
        - No existing details → prompt for all required fields
        - Partial details → extract from input, prompt for remaining
        - All fields present → confirm and set ``booking_confirmed = True``

        Args:
            state: Must contain at least one message in ``messages``.

        Returns:
            Updated state with booking progress.

        Raises:
            ValueError: If ``messages`` is empty.
        """
        if not state.get("messages"):
            raise ValueError("State must contain at least one message.")

        if state.get("booking_confirmed"):
            return self._with_intent(
                state,
                messages=[
                    AIMessage(
                        content=(
                            "Your appointment is already confirmed. "
                            "If you need to make changes, please contact our team directly."
                        )
                    )
                ],
                user_details=state.get("user_details"),
                booking_confirmed=True,
                intent_route=self.intent_route,
            )

        user_input = state["messages"][-1].content
        existing_details = dict(state.get("user_details") or {})

        try:
            extracted = self._extract_details(user_input)
            details = self._merge_details(existing_details, extracted)
        except Exception:
            details = dict(existing_details)

        # Initialize missing fields with empty strings to signal booking is in progress.
        for field in self.user_details_model.get_required_field_names():
            if field not in details:
                details[field] = ""

        missing = self._get_missing_fields(details)

        if not missing:
            details_str = "\\n".join(
                f"- **{k.replace('_', ' ').title()}**: {v}"
                for k, v in details.items()
                if v
            )
            confirmation = (
                f"Your appointment request has been recorded with the following details:\n"
                f"{details_str}\\n\\n"
                "This is a request, not a scheduled appointment. The team must confirm a date and time."
            )
            return self._with_intent(
                state,
                messages=[AIMessage(content=confirmation)],
                user_details=details,
                booking_confirmed=True,
                intent_route=self.intent_route,
            )

        if not existing_details:
            requested_fields = ", ".join(
                self.user_details_model.model_fields[field].description
                or field.replace("_", " ")
                for field in self.user_details_model.get_required_field_names()
            )
            return self._with_intent(
                state,
                messages=[
                    AIMessage(
                        content=(
                            "To request an appointment, please provide: "
                            f"{requested_fields}."
                        )
                    )
                ],
                user_details=details,
                intent_route=self.intent_route,
            )

        response = (self.inform_missing_prompt | self.llm).invoke(
            {"missing_fields": ", ".join(missing)}
        )
        return self._with_intent(
            state,
            messages=[AIMessage(content=response.content)],
            user_details=details,
            intent_route=self.intent_route,
        )
