"""MLflow conversation tracker.

Creates one MLflow run per conversation turn, recording tenant context,
response latency, intent routing, and booking status as structured
parameters, metrics, and searchable tags.
"""

import mlflow


def log_conversation(
    tenant_name: str,
    user_message: str,
    bot_response: str,
    latency_seconds: float,
    intent_route: str | None = None,
    booking_confirmed: bool = False,
) -> None:
    """Log a single conversation turn to MLflow.

    Each tenant is tracked under its own experiment
    (``groqflow_{tenant_name}``), keeping runs organized per workspace.

    Args:
        tenant_name: The name of the active tenant (e.g., ``City Hospital``).
        user_message: The raw message sent by the user.
        bot_response: The response returned by the agent.
        latency_seconds: End-to-end graph invocation duration in seconds.
        intent_route: Which LangGraph node handled the request.
        booking_confirmed: Whether an appointment was confirmed in this turn.
    """
    experiment_name = f"groqflow_{tenant_name.lower().replace(' ', '_')}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("tenant", tenant_name)
        # Truncate to stay within MLflow's 500-char parameter limit.
        mlflow.log_param("user_message", user_message[:500])
        mlflow.log_param("bot_response", bot_response[:500])

        mlflow.log_metric("latency_seconds", round(latency_seconds, 4))

        mlflow.set_tag("intent_route", intent_route or "unknown")
        mlflow.set_tag("booking_confirmed", str(booking_confirmed))
