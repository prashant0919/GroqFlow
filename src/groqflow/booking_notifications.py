"""Outbound n8n webhook dispatcher for booking events.

Dispatches ``booking.requested`` events to an n8n workflow via HTTP POST.
Each request includes an ``X-Idempotency-Key`` header derived from the event ID
to prevent duplicate processing on retries. Optionally attaches a Bearer token
for webhook authentication.
"""

import json
import os
from urllib.request import Request, urlopen


def send_booking_request(payload: dict) -> str:
    """POST a booking event payload to the configured n8n webhook.

    Args:
        payload: JSON-serializable event payload. Must contain an ``event_id``
            key used as the idempotency key.

    Returns:
        ``"sent"`` on successful delivery, ``"not_configured"`` if no webhook
        URL is set.

    Raises:
        RuntimeError: If the webhook returns a non-2xx HTTP status.
        urllib.error.URLError: On network-level failures (timeout, DNS, etc.).
    """
    webhook_url = os.getenv("N8N_BOOKING_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return "not_configured"

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Idempotency-Key": str(payload["event_id"]),
    }
    token = os.getenv("N8N_WEBHOOK_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        webhook_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=8) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"n8n webhook returned HTTP {response.status}")
    return "sent"
