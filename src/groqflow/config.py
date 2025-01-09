"""Centralized environment configuration and tenant path resolution.

Loads ``.env`` variables via ``python-dotenv``, reads the default tenant config,
and exposes path constants for vector indexes, uploaded documents, and LLM settings.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from groqflow.config_loader import (
    TenantConfig,
    get_user_details_model,
    load_tenant_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CONFIG_PATH = os.getenv(
    "TENANT_CONFIG_PATH", str(PROJECT_ROOT / "configs" / "demo_hospital.yaml")
)
TENANT = load_tenant_config(DEFAULT_CONFIG_PATH)
UserDetails = get_user_details_model(TENANT)

VECTOR_DB_ROOT = Path(os.getenv("VECTOR_DB_PATH", PROJECT_ROOT / "vector_db"))
DOCUMENTS_ROOT = Path(
    os.getenv("UPLOADED_DOCUMENTS_PATH", PROJECT_ROOT / "uploaded_documents")
)
if not VECTOR_DB_ROOT.is_absolute():
    VECTOR_DB_ROOT = PROJECT_ROOT / VECTOR_DB_ROOT
if not DOCUMENTS_ROOT.is_absolute():
    DOCUMENTS_ROOT = PROJECT_ROOT / DOCUMENTS_ROOT
VECTOR_DB_PATH = str(VECTOR_DB_ROOT / TENANT.id)
UPLOADED_DOCUMENTS_PATH = str(DOCUMENTS_ROOT / TENANT.id)

# LLM inference settings.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))


def tenant_paths(tenant: TenantConfig) -> tuple[Path, Path]:
    """Return the (documents_directory, vector_index_directory) for a tenant.

    Args:
        tenant: The tenant configuration.

    Returns:
        A tuple of ``(documents_path, vector_db_path)`` rooted under the
        configured ``DOCUMENTS_ROOT`` and ``VECTOR_DB_ROOT``.
    """
    return DOCUMENTS_ROOT / tenant.id, VECTOR_DB_ROOT / tenant.id
