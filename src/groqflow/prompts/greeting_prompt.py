from groqflow.config_loader import TenantConfig


def greeting_prompt_for(tenant: TenantConfig) -> str:
    return f"""
Create a concise, professional greeting for the following business. The greeting should be:
- Approximately 10 words or less
- Friendly and context-appropriate
- Warm and engaging
- Reflective of the business's industry and values

Business Information:

Name: {tenant.name}
Description: {tenant.description}

Provide only the greeting, without any additional text or explanations.
"""
