# GroqFlow Engineering & Error Resolution Log

> **Author:** Prashant Bastola  
> **Period:** Day 01 – Day 30  
> **Repository:** GroqFlow — Multi-Tenant AI Knowledge Assistant & Autonomous Appointment Booking Engine

This document chronicles the engineering decisions, technical challenges, and their resolutions encountered during the 30-day build of GroqFlow. Each issue was resolved locally before committing clean, passing code.

---

## Phase 1: Foundation & Multi-Tenant Configuration (Days 01–05)

### Day 01 — System Architecture Specification & Project Scaffolding

Established the project structure with `pyproject.toml` (setuptools backend), `src/groqflow/` package layout, and initial `.gitignore`. Chose a `src/` layout to prevent accidental imports from the project root during development.

**Decision:** Selected LangGraph over a simple LangChain sequential chain because the appointment booking flow requires stateful cycles — the agent must loop back to ask for missing fields, which is not possible with a linear chain.

### Day 02 — Multi-Tenant YAML Config Schema & Dynamic Workspace Loader

Implemented `config_loader.py` with Pydantic `BaseModel` validation for tenant configs. Each tenant YAML file defines booking fields that are dynamically compiled into a Pydantic model via `create_model()`.

**Design Choice:** Used `pydantic.create_model()` to generate per-tenant `UserDetails` models at runtime, so adding a new booking field only requires editing the YAML — no code changes.

### Day 03 — Groq LLM Client Setup & LangChain Configuration

Configured `ChatGroq` via `langchain-groq` with environment-driven model selection (`GROQ_MODEL`), temperature, and token limits. Wrapped configuration in `config.py` to centralize environment variable access.

### Day 04 — SQLite Persistence Layer for Tenant-Isolated Chat History

Built `chat_store.py` with a context-managed SQLite connection, WAL journaling for concurrent reads, and foreign key enforcement. Schema uses `conversations` and `messages` tables with composite indexes on `(owner_id, tenant_id)`.

**Design Choice:** Used WAL mode (`PRAGMA journal_mode=WAL`) to allow Streamlit's rerun-based concurrency model to read while the write lock is held during `append_turn`.

### Day 05 — Unit Test Harness & Base Exception Hierarchy

Set up `pytest` with `tmp_path` fixtures for SQLite tests. Created `BaseAgent` abstract base with an `intent_route` attribute and `_with_intent()` helper to ensure every node tags its responses with the route taken.

---

## Phase 2: Hybrid PDF Ingestion & Tenant-Aware FAISS RAG (Days 06–12)

### Day 06 — PDF Text Extraction Pipeline with PyMuPDF and pypdf

Implemented `_load_document()` in `KnowledgeBase` using `PyPDFLoader` (pypdf) as the primary text extractor. Digital PDFs with selectable text are handled efficiently without OCR overhead.

### Day 07 — Scanned Page Detection & RapidOCR Fallback Integration

#### Issue #01: Handling Scanned vs. Digital PDFs Without Blowing Up Memory/Latency

**Problem:** Naïvely running OCR on every page of every uploaded PDF caused unacceptable latency (30+ seconds per document) and excessive memory consumption from high-resolution pixmap rendering. Most uploaded PDFs are digital (selectable text), making full-document OCR wasteful.

**Investigation:** Profiled ingestion with a mixed corpus of digital and scanned PDFs. Found that digital pages already have text extracted by pypdf, and only scanned pages (certificates, handwritten forms) produce empty or near-empty text content.

**Resolution:** Implemented a **page-level character density heuristic**: after pypdf extracts text, each page is checked for a minimum character threshold (40 characters). Only pages falling below this threshold trigger the RapidOCR pipeline. This reduces OCR invocations by ~85% on typical mixed-content uploads.

```python
# knowledge_base.py — conditional OCR trigger
for document in documents:
    if len(document.page_content.strip()) >= 40:
        continue  # Digital page — skip OCR
    # Render page to pixmap and run RapidOCR...
```

**Validation:** Tested with a 50-page hospital policy PDF (all digital) and a 3-page scanned patient intake form. Ingestion time dropped from 34s to 2.1s for the digital PDF, and the scanned form was correctly OCR'd.

### Day 08 — Sentence-Transformers Embedding & Tenant FAISS Index Store

Integrated `HuggingFaceEmbeddings` (default `all-MiniLM-L6-v2`) for document chunk embedding. Each tenant gets an isolated FAISS index directory under `vector_db/{tenant_id}/`.

**Design Choice:** Used `IndexFlatL2` for simplicity and correctness at the current document scale. For production deployments exceeding 100K chunks, this should be replaced with `IndexIVFFlat` or `IndexHNSW`.

### Day 09 — Selective File Filtering & Hot-Reindexing on Duplicate Upload

#### Issue #02: Stale Vectors in FAISS When a User Re-uploads an Updated PDF

**Problem:** When a user uploaded a new version of a PDF with the same filename, the old document's chunk vectors remained in the FAISS index. This caused retrieval to return outdated content alongside the new version, producing contradictory or stale answers.

**Investigation:** FAISS does not natively support document-level operations — it operates on individual vectors. The `docstore._dict` maps vector IDs to `Document` objects with `metadata["source"]` tracking the origin filename.

**Resolution:** Built deterministic document tracking using the `source` metadata field. Before adding new chunks, `_get_deleted_chunk_ids()` scans the docstore for all existing chunks matching the filename and collects their IDs. These are purged from the FAISS index before the new chunks are added. The entire operation (delete + add + save) is batched in `update_many()` to minimize disk I/O.

```python
# knowledge_base.py — hot-reindex flow
deleted_chunk_ids.update(self._get_deleted_chunk_ids(indexed_source))
for page in pages:
    page.metadata["source"] = indexed_source
# ... batch delete, then batch add
```

**Validation:** Uploaded a 5-page PDF, verified 5 chunks in the index. Re-uploaded with 7 pages. Confirmed old 5 chunks were removed and 7 new chunks were added — no duplicates.

### Day 10 — Vector Store Deletion & Index Garbage Collection

Implemented `delete()` and `delete_many()` methods to remove all chunks associated with a list of source filenames. Uses set-based ID collection to ensure a single `vector_store.delete()` call per batch, followed by one `_save_vector_store()`.

### Day 11 — Context-Aware Retriever with Strict Source Grounding

Built `RetrieverNode` with source-filtered MMR (Maximal Marginal Relevance) retrieval. When the user selects specific knowledge files in the sidebar, only chunks from those files are considered. The filter uses a lambda predicate on `metadata["source"]` with `fetch_k` set to the total index size to ensure the filter doesn't accidentally exclude relevant results.

### Day 12 — RAG Retrieval Evaluation & Smoke Tests

Added unit tests for:
- Source filtering (only selected PDFs participate in retrieval)
- Empty source list returns no results (prevents full-index search on deselection)
- Batch update correctly replaces old chunks
- Hidden/temporary files are excluded from ingestion

---

## Phase 3: LangGraph Orchestration & Slot-Filling Booking Agent (Days 13–19)

### Day 13 — LangGraph Conversation State Schema & Graph Topology

Defined `AgentState` as a `TypedDict` with `messages` (annotated with `add_messages` reducer), `context`, `choice`, `user_details`, `booking_confirmed`, `intent_route`, and `source_filter`. The `add_messages` annotation enables LangGraph's built-in message accumulation across nodes.

### Day 14 — Autonomous Intent Router Node (Knowledge Q&A vs Booking)

Implemented `ClassifierNode` with a dual-path routing strategy:

1. **With selected sources**: Uses regex-based fast-path classification for greetings and booking keywords, defaulting to knowledge retrieval. This avoids an unnecessary LLM call when the intent is deterministic.
2. **Without selected sources**: Falls through to the LLM-based classifier prompt for full natural language understanding.

**Design Choice:** The regex fast-path exists because when a user has selected specific knowledge files, we can safely assume non-greeting, non-booking queries are knowledge questions. This reduces latency by ~200ms per query.

### Day 15 — Dynamic Slot-Filling Node for Missing Appointment Details

Built `BookingNode` with multi-turn extraction:
1. First turn: Prompts for all required fields
2. Subsequent turns: Extracts provided fields via LLM JSON output parser, merges with existing state, and prompts only for remaining missing fields
3. Completion: Generates confirmation message and sets `booking_confirmed = True`

### Day 16 — Booking Confirmation State Machine & Validation Logic

The booking flow acts as a finite state machine:
- `no_details` → `partial_details` → `all_details_confirmed`
- Once confirmed, subsequent booking intents return an "already confirmed" message
- State transitions are persisted in `agent_state` via SQLite

### Day 17 — Multi-Turn Conversation Memory Binding in LangGraph

Conversation history is reconstructed from SQLite on each Streamlit rerun and passed as the `messages` list to the graph. The `add_messages` reducer in `AgentState` accumulates the new response. Agent state (booking details, confirmation flag, intent route) is serialized as JSON in the `conversations` table.

**Design Choice:** Excluded `messages` and `choice` (AIMessage objects) from the serialized `agent_state` to avoid JSON serialization issues with LangChain message types. These are reconstructed from the `messages` table instead.

### Day 18 — End-to-End Agent Loop Integration Tests

Added integration tests that mock the LangGraph graph and verify:
- Intent route is preserved across turns
- Source filter is passed through to the graph
- Agent state round-trips through `generate_reply()`

### Day 19 — Prompt Guardrails for Cross-Domain Knowledge Files

The classifier allows knowledge questions about any uploaded file, regardless of the tenant's domain. A hospital chatbot can answer questions from a school policy PDF if that file is selected. This is by design — the knowledge base is the source of truth, not the tenant description.

---

## Phase 4: n8n Event Webhooks, Idempotency & MLflow Observability (Days 20–25)

### Day 20 — Outbound n8n Webhook Client for booking.requested Events

Implemented `send_booking_request()` in `booking_notifications.py` using Python's `urllib.request` (no external HTTP library dependency). The function POSTs a JSON payload with `event`, `event_id`, `occurred_at`, `status`, chatbot metadata, `conversation_id`, and `booking_details`.

### Day 21 — X-Idempotency-Key Deduplication & Bearer Token Security

#### Issue #03: Duplicate Appointment Notifications if n8n Times Out

**Problem:** If the n8n webhook timed out or returned an error, the booking was still marked as confirmed in the conversation state. On the next user message, `generate_reply()` would detect `booking_confirmed=True` and attempt to send the notification again — but with a **new** `event_id`, bypassing any downstream deduplication.

**Investigation:** Traced the notification flow in `chat_service.py`. The `event_id` was generated fresh on each attempt because it wasn't persisted in `agent_state`.

**Resolution:** 
1. Generate the `event_id` once on the first notification attempt and persist it in `agent_state["booking_notification_id"]`
2. On retry (next user message where `booking_notification_status` is `"pending"`), reuse the same `event_id`
3. Attach the `event_id` as the `X-Idempotency-Key` header so n8n can deduplicate at the workflow level
4. Persist `booking_notification_status` as `"sent"`, `"pending"`, or `"not_configured"`

```python
# chat_service.py — idempotent retry logic
event_id = agent_state.get("booking_notification_id") or str(uuid4())
next_agent_state["booking_notification_id"] = event_id
# ... POST with X-Idempotency-Key: event_id
```

**Validation:** Test `test_booking_notification_retries_with_same_event_id` verifies that two consecutive `generate_reply()` calls (first fails, second succeeds) use the same `event_id` in the payload.

### Day 22 — Persistent Webhook Retry Queue on Subsequent Chat Turns

Extended the retry logic to also retry `"not_configured"` status (webhook URL was empty on first attempt but may have been configured since). The retry is triggered on each `generate_reply()` call where `booking_confirmed=True` and `booking_notification_status` is not `"sent"`.

### Day 23 — MLflow Experiment Tracking for Latency and RAG Telemetry

Built `logger.py` to create one MLflow run per conversation turn, logging:
- **Parameters:** tenant name, user message (truncated to 500 chars), bot response
- **Metrics:** `latency_seconds` (end-to-end graph invocation time)
- **Tags:** `intent_route`, `booking_confirmed`

Each tenant gets its own MLflow experiment (`groqflow_{tenant_name}`).

### Day 24 — Structured Error Logging & Failure Recovery Audit

Added `try/except` guards around non-critical paths (MLflow logging, webhook delivery) so conversation flow is never interrupted by telemetry failures. All exceptions are logged with `logging.exception()` for full stack traces.

### Day 25 — Webhook Mock Server & Idempotency Unit Tests

Wrote comprehensive tests for `booking_notifications.py`:
- `test_send_booking_request_posts_idempotent_authenticated_event`: Verifies URL, headers (including `X-Idempotency-Key` and `Authorization: Bearer`), payload, and timeout
- `test_send_booking_request_is_disabled_without_webhook`: Verifies no HTTP request is made when `N8N_BOOKING_WEBHOOK_URL` is empty
- `test_booking_notification_retries_with_same_event_id`: End-to-end retry test through `generate_reply()`

---

## Phase 5: Streamlit Workspace UI, Docker & CI/CD Hardening (Days 26–30)

### Day 26 — Streamlit Multi-Tenant Sidebar & Dynamic Chatbot Creator UI

Built the Streamlit sidebar with:
- Tenant selector (loads all YAML configs from `configs/`)
- Knowledge file multiselect (shows uploaded PDFs per tenant)
- "Create chatbot" form that generates a new YAML config and reloads the app

**Design Choice:** New chatbot creation generates a slug-based `id` and writes a YAML file to `configs/`. This keeps tenant registration file-based and git-trackable for static tenants, while dynamic tenants are created at runtime.

### Day 27 — Knowledge Base File Manager UI & Responsive Light/Dark Workspace

Implemented the workspace view with upload/index/delete workflows:
- Upload: Writes to a temporary dotfile (`.{uuid}.upload.pdf`) first, then renames on successful indexing to prevent partial uploads from corrupting the document store
- Delete: Removes chunks from FAISS and deletes the source file
- Atomic file operations prevent race conditions on concurrent Streamlit reruns

### Day 28 — Production Dockerfile with Linux System Libs for RapidOCR

#### Issue #04: Missing Linux Shared Libraries Crashing RapidOCR/OpenCV in Headless Docker & Streamlit Cloud

**Problem:** After containerizing the application, `RapidOCR` and `OpenCV` crashed on import with `ImportError: libGL.so.1: cannot open shared object file`. The same error occurred on Streamlit Community Cloud deployments.

**Investigation:** The `python:3.11-slim` base image strips non-essential system libraries. OpenCV (`opencv-python`) requires `libGL` for rendering operations (even in headless mode), and `libglib2.0-0` for GLib. `libgomp1` is required by FAISS for OpenMP parallelism.

**Resolution:**
1. **Docker:** Added an `apt-get install` layer for `libgl1`, `libglib2.0-0`, and `libgomp1` in the Dockerfile
2. **Streamlit Community Cloud:** Created `packages.txt` in the repository root listing the same three packages. Community Cloud reads this file during build and installs the listed apt packages.

```dockerfile
# Dockerfile — system library layer
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
```

```
# packages.txt — Streamlit Community Cloud
libgl1
libglib2.0-0
libgomp1
```

**Validation:** Built and ran the Docker image with a scanned PDF upload. RapidOCR processed the pages without import errors. Deployed to Streamlit Community Cloud and confirmed successful build with OCR functionality.

### Day 29 — GitHub Actions CI Pipeline with Strict Ruff Linting & Smoke Tests

Configured `.github/workflows/ci.yml` with:
1. Ruff lint check (`ruff check . --output-format=github`)
2. Pytest run (`pytest -q`)
3. YAML config validation (ensures all tenant configs have `tenant.name`)
4. Config loader smoke test (imports and validates `demo_hospital.yaml`)

**Design Choice:** CI uses `pip install -e ".[dev]"` to install the package in editable mode with dev dependencies (Ruff, pytest). This ensures the test environment matches the development environment exactly.

### Day 30 — Architecture Diagrams, Engineering Log & Production Documentation

Finalized:
- This engineering log documenting all issues and resolutions
- Architecture Decision Records (ADRs) for key design choices
- Mermaid.js architecture diagram in README.md
- Production readiness guide with deployment strategies
- Pre-commit hooks for local quality gates

---

## Issue Resolution Summary

| Issue | Day | Root Cause | Resolution | Test Coverage |
|---|---|---|---|---|
| #01 — OCR memory/latency on digital PDFs | Day 07 | Unnecessary OCR on pages with extractable text | Character density heuristic (40-char threshold) | `test_load_document_ocr_fills_scanned_page_text` |
| #02 — Stale vectors after PDF re-upload | Day 09 | No chunk cleanup before re-indexing | Deterministic source tracking + batch purge | `test_update_many_batches_add_and_save` |
| #03 — Duplicate webhook notifications | Day 21 | Fresh `event_id` generated per retry attempt | Persistent `event_id` + `X-Idempotency-Key` | `test_booking_notification_retries_with_same_event_id` |
| #04 — Missing system libs in Docker/Cloud | Day 28 | `python:3.11-slim` strips OpenCV/FAISS dependencies | `packages.txt` + Dockerfile apt layer | Manual integration test |
