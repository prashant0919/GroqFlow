# ADR-002: Idempotent n8n Webhooks with X-Idempotency-Key

**Status:** Accepted  
**Date:** Day 21 — X-Idempotency-Key Deduplication & Bearer Token Security  
**Author:** Prashant Bastola  
**Deciders:** Prashant Bastola

## Context

When GroqFlow completes the booking slot-filling flow (all required fields collected), it dispatches a `booking.requested` event to an n8n webhook. This webhook can trigger downstream actions: sending confirmation emails, Telegram notifications, or creating calendar entries.

## Problem

Without idempotency protection, network timeouts or retries lead to duplicate appointment notifications and bookings.

## Decision

Implement **at-least-once delivery with client-side idempotency keys**:
1. Generate `event_id` once on first booking confirmation and persist in `agent_state["booking_notification_id"]`.
2. Send `X-Idempotency-Key: {event_id}` in the HTTP header.
3. Track status (`"sent"`, `"pending"`, `"not_configured"`).
4. Retry failed deliveries on the next user turn using the **same** `event_id`.
