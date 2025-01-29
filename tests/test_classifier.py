from langchain_core.messages import AIMessage, HumanMessage

from groqflow.nodes.classifier import ClassifierNode


class UnexpectedModelCall:
    def invoke(self, values):
        raise AssertionError("Selected-file intent routing should not call the model")


def make_classifier() -> ClassifierNode:
    classifier = object.__new__(ClassifierNode)
    classifier.chain = UnexpectedModelCall()
    return classifier


def test_selected_document_overrides_chatbot_domain_for_questions() -> None:
    result = make_classifier()(
        {
            "messages": [HumanMessage(content="How does school enrollment work?")],
            "source_filter": ["/docs/school.pdf"],
        }
    )

    assert result["choice"] == AIMessage(content="2")


def test_selected_document_preserves_greeting_and_booking_routes() -> None:
    classifier = make_classifier()
    sources = ["/docs/clinic.pdf"]

    greeting = classifier(
        {"messages": [HumanMessage(content="Hello!")], "source_filter": sources}
    )
    booking = classifier(
        {
            "messages": [HumanMessage(content="Please book an appointment")],
            "source_filter": sources,
        }
    )

    assert greeting["choice"] == AIMessage(content="1")
    assert booking["choice"] == AIMessage(content="3")
