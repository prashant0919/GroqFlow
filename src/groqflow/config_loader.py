"""Pydantic-validated YAML tenant configuration loader.

Each tenant is defined by a YAML file in ``configs/`` with a ``tenant`` root key
containing identity, UI theming, and booking field definitions. The booking fields
are compiled into a dynamic Pydantic model at load time via ``create_model()``, so
adding a new field requires only a YAML edit — no code changes.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, create_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UIConfig(BaseModel):
    """Streamlit theme configuration for a tenant workspace."""

    primary_hue: str = "indigo"
    secondary_hue: str = "blue"
    avatar: str | None = None


class BookingField(BaseModel):
    """Schema for a single booking detail field (e.g., patient name, DOB)."""

    name: str
    description: str
    required: bool = True
    type: str = "string"


class BookingConfig(BaseModel):
    """Defines the set of fields required for appointment booking in a tenant."""

    fields: list[BookingField]


class TenantConfig(BaseModel):
    """Top-level tenant configuration loaded from a YAML file.

    Attributes:
        id: Unique tenant identifier (e.g., ``hospital_001``).
        name: Human-readable tenant name displayed in the UI.
        description: Tenant description used in LLM prompt context.
        ui: Streamlit theme settings.
        booking: Booking field definitions for slot-filling extraction.
    """

    id: str
    name: str
    description: str
    ui: UIConfig
    booking: BookingConfig


def load_tenant_config(filepath: str) -> TenantConfig:
    """Load and validate a tenant configuration from a YAML file.

    Args:
        filepath: Absolute or project-relative path to the YAML config.

    Returns:
        A validated ``TenantConfig`` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the YAML is missing the ``tenant`` root key.
        pydantic.ValidationError: If the config fails schema validation.
    """
    config_path = Path(filepath)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TenantConfig(**data["tenant"])


def save_tenant_config(config: TenantConfig, filepath: str | Path) -> Path:
    """Serialize a tenant configuration to a YAML file.

    Args:
        config: The tenant configuration to save.
        filepath: Destination path for the YAML file.

    Returns:
        The resolved ``Path`` where the file was written.
    """
    config_path = Path(filepath)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {"tenant": config.model_dump(mode="json")},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return config_path


def get_user_details_model(config: TenantConfig) -> type[BaseModel]:
    """Dynamically build a Pydantic model from a tenant's booking field definitions.

    The generated model has one field per booking entry in the tenant config.
    Required fields use ``Field(...)`` (mandatory), optional fields default to ``None``.

    Args:
        config: The tenant configuration containing booking field definitions.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass named ``UserDetails``
        with helper class methods for field introspection.
    """
    fields: dict[str, Any] = {}

    for field in config.booking.fields:
        annotation: type = str
        if field.type == "boolean":
            annotation = bool

        if field.required:
            pydantic_field = Field(..., description=field.description)
        else:
            annotation = annotation | None  # type: ignore[assignment]
            pydantic_field = Field(None, description=field.description)

        fields[field.name] = (annotation, pydantic_field)

    model = create_model("UserDetails", **fields)  # type: ignore[call-overload]

    @classmethod  # type: ignore[misc]
    def get_required_field_names(cls) -> list[str]:
        """Return the names of all required booking fields."""
        return [name for name, f in cls.model_fields.items() if f.is_required()]

    @classmethod  # type: ignore[misc]
    def get_required_fields(cls) -> str:
        """Return a comma-separated string of required field names."""
        return ",".join(cls.get_required_field_names())  # type: ignore[attr-defined]

    model.get_required_field_names = get_required_field_names
    model.get_required_fields = get_required_fields

    return model
