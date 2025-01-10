from pathlib import Path

from langchain_core.messages import AIMessage

from groqflow import chat_store
from groqflow.chat_service import generate_reply
from groqflow.config_loader import load_tenant_config


def test_conversation_history_is_scoped_and_persistent(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(chat_store, "DATABASE_PATH", tmp_path / "history.sqlite3")
    conversation_id = chat_store.create_conversation("user-a", "hospital_001")

    chat_store.append_turn(
        conversation_id,
        "user-a",
        "hospital_001",
        "Where are you located?",
        "We are downtown.",
        {"context": "location", "booking_confirmed": False},
    )

    loaded = chat_store.load_conversation(conversation_id, "user-a", "hospital_001")
    assert loaded is not None
    assert loaded["title"] == "Where are you located?"
    assert loaded["messages"] == [
        {"role": "user", "content": "Where are you located?"},
        {"role": "assistant", "content": "We are downtown."},
    ]
    assert loaded["agent_state"]["context"] == "location"
    assert (
        chat_store.load_conversation(conversation_id, "user-b", "hospital_001") is None
    )
    assert chat_store.load_conversation(conversation_id, "user-a", "school_001") is None


def test_append_turn_does_not_persist_graph_messages(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(chat_store, "DATABASE_PATH", tmp_path / "history.sqlite3")
    conversation_id = chat_store.create_conversation("user-a", "hospital_001")

    chat_store.append_turn(
        conversation_id,
        "user-a",
        "hospital_001",
        "Hello",
        "Hello there",
        {
            "context": "",
            "messages": [AIMessage(content="Hello there")],
            "choice": AIMessage(content="1"),
        },
    )

    loaded = chat_store.load_conversation(conversation_id, "user-a", "hospital_001")
    assert loaded is not None
    assert loaded["agent_state"] == {"context": ""}
    assert loaded["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello there"},
    ]


def test_generate_reply_preserves_intent_route_in_agent_state(monkeypatch) -> None:
    tenant = load_tenant_config("configs/demo_school.yaml")
    graph_input = {}

    class FakeGraph:
        def invoke(self, state):
            graph_input.update(state)
            return {
                "messages": [AIMessage(content="Hello there")],
                "context": "school information",
                "user_details": {"student_name": "Ava"},
                "booking_confirmed": False,
                "intent_route": "greeter",
            }

    monkeypatch.setattr("groqflow.chat_service._graph_for", lambda tenant: FakeGraph())

    response, next_state = generate_reply(
        tenant,
        "Hello",
        [{"role": "user", "content": "Hi"}],
        {"context": "", "user_details": {}, "booking_confirmed": False},
        source_filter=["/docs/policy.pdf"],
    )

    assert response == "Hello there"
    assert next_state["intent_route"] == "greeter"
    assert next_state["context"] == "school information"
    assert graph_input["source_filter"] == ["/docs/policy.pdf"]


def test_conversation_list_is_tenant_scoped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(chat_store, "DATABASE_PATH", tmp_path / "history.sqlite3")
    chat_store.create_conversation("user-a", "hospital_001")
    chat_store.create_conversation("user-a", "school_001")

    hospital_chats = chat_store.list_conversations("user-a", "hospital_001")
    school_chats = chat_store.list_conversations("user-a", "school_001")

    assert len(hospital_chats) == 1
    assert len(school_chats) == 1
    assert hospital_chats[0]["id"] != school_chats[0]["id"]
