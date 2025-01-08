from pathlib import Path

from groqflow.config_loader import (
    BookingConfig,
    BookingField,
    TenantConfig,
    UIConfig,
    load_tenant_config,
    save_tenant_config,
)


def test_save_tenant_config_can_be_loaded_again(tmp_path: Path) -> None:
    chatbot = TenantConfig(
        id="custom_clinic",
        name="Community Clinic",
        description="A local primary care clinic.",
        ui=UIConfig(),
        booking=BookingConfig(
            fields=[
                BookingField(
                    name="patient_name",
                    description="Patient name",
                    required=True,
                )
            ]
        ),
    )

    path = save_tenant_config(chatbot, tmp_path / "custom_clinic.yaml")
    restored = load_tenant_config(str(path))

    assert restored == chatbot
