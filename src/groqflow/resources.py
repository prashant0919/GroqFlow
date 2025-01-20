"""Singleton knowledge base factory with per-tenant caching.

Maintains a dictionary of ``KnowledgeBase`` instances keyed by tenant ID
and vector store path. Ensures each tenant's FAISS index is loaded only once
per process lifetime.
"""

from pathlib import Path

from groqflow.config import VECTOR_DB_PATH, tenant_paths
from groqflow.config_loader import TenantConfig
from groqflow.knowledge_base import KnowledgeBase

_knowledge_bases: dict[str, KnowledgeBase] = {}


def get_knowledge_base(
    tenant: TenantConfig | None = None,
    documents_path: Path | None = None,
    vector_db_path: Path | None = None,
) -> KnowledgeBase:
    """Get or create a cached ``KnowledgeBase`` for a tenant.

    Args:
        tenant: Tenant configuration. When provided, paths are derived
            from ``tenant_paths()``. Overrides can be passed via
            ``documents_path`` and ``vector_db_path``.
        documents_path: Optional explicit document directory.
        vector_db_path: Optional explicit FAISS index directory.

    Returns:
        A ``KnowledgeBase`` instance (cached on first creation).
    """
    if tenant is not None:
        default_documents, default_index = tenant_paths(tenant)
        documents_path = documents_path or default_documents
        vector_db_path = vector_db_path or default_index
        key = tenant.id
    else:
        documents_path = documents_path or Path(VECTOR_DB_PATH).parent
        vector_db_path = vector_db_path or Path(VECTOR_DB_PATH)
        key = str(vector_db_path.resolve())

    documents_path.mkdir(parents=True, exist_ok=True)
    key = f"{key}:{vector_db_path.resolve()}"
    if key not in _knowledge_bases:
        _knowledge_bases[key] = KnowledgeBase(
            folder_path=str(documents_path),
            vector_db_path=str(vector_db_path),
        )
    return _knowledge_bases[key]


def has_knowledge_base(vector_db_path: Path) -> bool:
    """Check whether a persisted FAISS index exists at the given path."""
    return KnowledgeBase.has_index(vector_db_path)
