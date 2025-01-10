"""SQLite conversation persistence layer.

Provides a repository pattern for tenant-isolated chat histories and booking
state. Uses WAL journaling for concurrent read access and foreign key
enforcement for referential integrity between conversations and messages.
"""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_PATH = Path(
    os.getenv("CHAT_DATABASE_PATH", PROJECT_ROOT / "data" / "conversations.sqlite3")
)


def _connect() -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode and create tables if needed.

    Returns:
        A configured ``sqlite3.Connection`` with row factory enabled.
    """
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            title TEXT NOT NULL,
            agent_state TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_owner_tenant
            ON conversations(owner_id, tenant_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id, id);
        """
    )
    return connection


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Context manager that yields a SQLite connection and ensures cleanup.

    The connection is used within a transaction context (``with connection:``),
    which auto-commits on success or rolls back on exception.
    """
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def list_conversations(owner_id: str, tenant_id: str) -> list[dict[str, str]]:
    """List all conversations for a user within a tenant, newest first.

    Args:
        owner_id: The user's stable identifier.
        tenant_id: The tenant to scope the query to.

    Returns:
        A list of dicts with ``id``, ``title``, and ``updated_at`` keys.
    """
    with _connection() as connection:
        rows = connection.execute(
            """SELECT id, title, updated_at FROM conversations
               WHERE owner_id = ? AND tenant_id = ?
               ORDER BY updated_at DESC""",
            (owner_id, tenant_id),
        ).fetchall()
    return [dict(row) for row in rows]


def create_conversation(owner_id: str, tenant_id: str) -> str:
    """Create a new conversation and return its ID.

    Args:
        owner_id: The user's stable identifier.
        tenant_id: The tenant this conversation belongs to.

    Returns:
        The UUID of the newly created conversation.
    """
    conversation_id = str(uuid4())
    now = datetime.now(UTC).isoformat()
    with _connection() as connection:
        connection.execute(
            """INSERT INTO conversations
               (id, owner_id, tenant_id, title, agent_state, created_at, updated_at)
               VALUES (?, ?, ?, ?, '{}', ?, ?)""",
            (conversation_id, owner_id, tenant_id, "New conversation", now, now),
        )
    return conversation_id


def load_conversation(
    conversation_id: str, owner_id: str, tenant_id: str
) -> dict | None:
    """Load a conversation with its messages and agent state.

    The query is scoped to ``owner_id`` and ``tenant_id`` to enforce
    tenant isolation — a user cannot load another user's or another
    tenant's conversation.

    Args:
        conversation_id: The UUID of the conversation.
        owner_id: The user's stable identifier.
        tenant_id: The tenant to scope the query to.

    Returns:
        A dict with ``id``, ``title``, ``agent_state``, and ``messages`` keys,
        or ``None`` if no matching conversation exists.
    """
    with _connection() as connection:
        conversation = connection.execute(
            """SELECT id, title, agent_state FROM conversations
               WHERE id = ? AND owner_id = ? AND tenant_id = ?""",
            (conversation_id, owner_id, tenant_id),
        ).fetchone()
        if conversation is None:
            return None
        messages = connection.execute(
            """SELECT role, content FROM messages
               WHERE conversation_id = ? ORDER BY id""",
            (conversation_id,),
        ).fetchall()
    return {
        "id": conversation["id"],
        "title": conversation["title"],
        "agent_state": json.loads(conversation["agent_state"]),
        "messages": [dict(message) for message in messages],
    }


def append_turn(
    conversation_id: str,
    owner_id: str,
    tenant_id: str,
    user_message: str,
    assistant_message: str,
    agent_state: dict,
) -> None:
    """Append a user/assistant message pair and update the conversation state.

    The conversation title is set to the first user message (truncated to 64
    characters) if it is still the default ``"New conversation"`` placeholder.

    LangChain message objects (``messages``, ``choice``) are excluded from
    the serialized agent state to avoid JSON encoding issues.

    Args:
        conversation_id: The UUID of the conversation.
        owner_id: The user's stable identifier.
        tenant_id: The tenant to scope the query to.
        user_message: The user's message text.
        assistant_message: The assistant's response text.
        agent_state: The current agent state dict to persist.

    Raises:
        ValueError: If the conversation does not exist for this user/tenant.
    """
    now = datetime.now(UTC).isoformat()
    with _connection() as connection:
        conversation = connection.execute(
            """SELECT title FROM conversations
               WHERE id = ? AND owner_id = ? AND tenant_id = ?""",
            (conversation_id, owner_id, tenant_id),
        ).fetchone()
        if conversation is None:
            raise ValueError("Conversation does not exist for this user and tenant.")
        title = conversation["title"]
        if title == "New conversation":
            title = " ".join(user_message.split())[:64] or title
        connection.executemany(
            """INSERT INTO messages (conversation_id, role, content, created_at)
               VALUES (?, ?, ?, ?)""",
            [
                (conversation_id, "user", user_message, now),
                (conversation_id, "assistant", assistant_message, now),
            ],
        )
        connection.execute(
            """UPDATE conversations SET title = ?, agent_state = ?, updated_at = ?
               WHERE id = ?""",
            (
                title,
                json.dumps(
                    {
                        key: value
                        for key, value in agent_state.items()
                        if key not in {"messages", "choice"}
                    }
                ),
                now,
                conversation_id,
            ),
        )
