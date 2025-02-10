"""Chat service orchestration layer.

Bridges the Streamlit UI with the LangGraph agent. Handles:
- Conversation history reconstruction from message dicts to LangChain messages
- Graph invocation with source filtering
- MLflow telemetry logging
- Idempotent n8n webhook dispatch with turn-based retry
"""

import logging
import os
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from groqflow.booking_notifications import send_booking_request
from groqflow.config import GROQ_MODEL
from groqflow.config_loader import TenantConfig
from groqflow.graphs.intranet import build_graph
from groqflow.logger import log_conversation
from groqflow.resources import get_knowledge_base

_graphs: dict[str, Any] = {}


def _graph_for(tenant: TenantConfig) -> object:
    """Get or create a cached compiled graph for a tenant.

    The cache key includes the tenant ID and model name to support
    runtime model switching.
    """
    model = os.getenv("GROQ_MODEL", GROQ_MODEL)
    cache_key = f"{tenant.id}:{model}"
    if cache_key not in _graphs:
        _graphs[cache_key] = build_graph(
            tenant,
            get_knowledge_base(tenant),
            model=model,
        )
    return _graphs[cache_key]


def generate_reply(
    tenant: TenantConfig,
    message: str,
    history: list[dict[str, str]],
    agent_state: dict[str, Any],
    source_filter: list[str] | None = None,
    conversation_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate an assistant response for a user message.

    Reconstructs conversation history, invokes the LangGraph agent,
    logs the interaction to MLflow, and dispatches booking webhooks
    with idempotency guarantees.

    Args:
        tenant: The active tenant configuration.
        message: The user's current message text.
        history: Prior conversation messages as ``{"role": ..., "content": ...}`` dicts.
        agent_state: The persisted agent state from the previous turn.
        source_filter: Optional list of active knowledge file paths.
        conversation_id: The current conversation's UUID (included in webhook payloads).

    Returns:
        A tuple of ``(response_text, next_agent_state)``.
    """
    messages: list[HumanMessage | AIMessage] = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    messages.append(HumanMessage(content=message))

    started_at = time.perf_counter()
    result = _graph_for(tenant).invoke(
        {
            "messages": messages,
            "context": agent_state.get("context", ""),
            "user_details": agent_state.get("user_details"),
            "booking_confirmed": agent_state.get("booking_confirmed", False),
            "intent_route": agent_state.get("intent_route", "unknown"),
            "source_filter": source_filter,
        }
    )
    response = str(result["messages"][-1].content)
    booking_confirmed = bool(result.get("booking_confirmed", False))
    intent_route = (
        result.get("intent_route") or agent_state.get("intent_route") or "unknown"
    )

    try:
        log_conversation(
            tenant_name=tenant.name,
            user_message=message,
            bot_response=response,
            latency_seconds=time.perf_counter() - started_at,
            intent_route=intent_route,
            booking_confirmed=booking_confirmed,
        )
    except Exception:
        logging.exception("Conversation logging failed")

    next_agent_state: dict[str, Any] = {
        "context": result.get("context", agent_state.get("context", "")),
        "user_details": result.get("user_details", agent_state.get("user_details")),
        "booking_confirmed": booking_confirmed,
        "intent_route": intent_route,
    }

    # Webhook dispatch with idempotent retry logic.
    notification_status = agent_state.get("booking_notification_status")
    should_notify = booking_confirmed and (
        not agent_state.get("booking_confirmed")
        or notification_status in {"pending", "not_configured"}
    )
    if should_notify:
        event_id = agent_state.get("booking_notification_id") or str(uuid4())
        next_agent_state["booking_notification_id"] = event_id
        payload = {
            "event": "booking.requested",
            "event_id": event_id,
            "occurred_at": datetime.now(UTC).isoformat(),
            "status": "request_received_not_scheduled",
            "chatbot": {"id": tenant.id, "name": tenant.name},
            "conversation_id": conversation_id,
            "booking_details": next_agent_state["user_details"] or {},
        }
        try:
            next_agent_state["booking_notification_status"] = send_booking_request(
                payload
            )
        except Exception:
            logging.exception("n8n booking notification failed")
            next_agent_state["booking_notification_status"] = "pending"
    elif notification_status:
        next_agent_state["booking_notification_status"] = notification_status
        next_agent_state["booking_notification_id"] = agent_state.get(
            "booking_notification_id"
        )

    if "choice" in result or "choice" in agent_state:
        next_agent_state["choice"] = result.get("choice", agent_state.get("choice"))
    return response, next_agent_state
