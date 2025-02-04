from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from groqflow.nodes.booking import BookingNode


def test_initial_booking_request_skips_llm_for_field_prompt() -> None:
    class UserDetails:
        model_fields = {
            "patient_name": SimpleNamespace(description="Patient's full name"),
            "dob": SimpleNamespace(description="Date of birth (YYYY-MM-DD)"),
        }

        @classmethod
        def get_required_field_names(cls) -> list[str]:
            return list(cls.model_fields)

    class UnexpectedLlmCall:
        def invoke(self, prompt: str):
            raise AssertionError("A model call is unnecessary to list required fields")

    node = object.__new__(BookingNode)
    node.user_details_model = UserDetails
    node.llm = UnexpectedLlmCall()
    node._extract_details = lambda user_input: {}

    result = node(
        {
            "messages": [HumanMessage(content="Let's book an appointment")],
            "user_details": {},
        }
    )

    assert result["messages"][-1].content == (
        "To request an appointment, please provide: "
        "Patient's full name, Date of birth (YYYY-MM-DD)."
    )
