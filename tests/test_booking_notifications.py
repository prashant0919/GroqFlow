import json

from langchain_core.messages import AIMessage

from groqflow import booking_notifications, chat_service
from groqflow.config_loader import load_tenant_config


def test_send_booking_request_posts_idempotent_authenticated_event(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "N8N_BOOKING_WEBHOOK_URL", "https://n8n.example/webhook/bookings"
    )
    monkeypatch.setenv("N8N_WEBHOOK_TOKEN", "test-token")
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(booking_notifications, "urlopen", fake_urlopen)
    payload = {"event": "booking.requested", "event_id": "event-123"}

    status = booking_notifications.send_booking_request(payload)

    assert status == "sent"
    assert captured["url"] == "https://n8n.example/webhook/bookings"
    assert captured["headers"]["X-idempotency-key"] == "event-123"
    assert captured["headers"]["Authorization"] == "Bearer test-token"
    assert captured["payload"] == payload
    assert captured["timeout"] == 8


def test_send_booking_request_is_disabled_without_webhook(monkeypatch) -> None:
    monkeypatch.delenv("N8N_BOOKING_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(
        booking_notifications,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected request")
        ),
    )

    assert booking_notifications.send_booking_request({"event_id": "event-123"}) == (
        "not_configured"
    )


def test_booking_notification_retries_with_same_event_id(monkeypatch) -> None:
    tenant = load_tenant_config("configs/demo_school.yaml")

    class FakeGraph:
        def invoke(self, state):
            return {
                "messages": [AIMessage(content="Request recorded")],
                "user_details": {"student_name": "Ava"},
                "booking_confirmed": True,
                "intent_route": "booking",
            }

    attempts = []

    def send(payload):
        attempts.append(payload)
        if len(attempts) == 1:
            raise TimeoutError("n8n did not respond in time")
        return "sent"

    monkeypatch.setattr(chat_service, "_graph_for", lambda tenant: FakeGraph())
    monkeypatch.setattr(chat_service, "log_conversation", lambda **kwargs: None)
    monkeypatch.setattr(chat_service, "send_booking_request", send)

    _, pending_state = chat_service.generate_reply(
        tenant,
        "Ava",
        [],
        {"booking_confirmed": False},
        conversation_id="conversation-123",
    )
    _, sent_state = chat_service.generate_reply(
        tenant,
        "Please check my request",
        [],
        pending_state,
        conversation_id="conversation-123",
    )

    assert pending_state["booking_notification_status"] == "pending"
    assert sent_state["booking_notification_status"] == "sent"
    assert attempts[0]["event_id"] == attempts[1]["event_id"]
    assert attempts[1]["conversation_id"] == "conversation-123"
    assert attempts[1]["booking_details"] == {"student_name": "Ava"}
