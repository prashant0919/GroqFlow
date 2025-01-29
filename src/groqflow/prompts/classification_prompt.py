from langchain_core.prompts import ChatPromptTemplate

from groqflow.config_loader import TenantConfig


def classification_prompt_for(tenant: TenantConfig) -> ChatPromptTemplate:
    template = f"""

Business information:

Name: {tenant.name}

Description: {tenant.description}


Classify the following user query into one of these categories:

Category - 1: Greeting: Any form of hello, hi, hey, good morning/afternoon/evening, etc.
Category - 2: Inquiry: Questions or statements about the company, its products, services, policies, or general information.
Category - 3: Appointment/Scheduling: Requests to book, schedule, or inquire about appointments, meetings, or reservations.
Category - 0: Other: Any query that doesn't fit into the above categories.

User Query: 
{{user_query}}

Please analyze the user_query carefully and return ONLY the appropriate category number based on the classification criteria above. 

Do not include any additional text or delimiter or explanation in your response, just the number.

Examples:

User Query: "Hi there!"
1

User Query: "What are your business hours?"
2

User Query: "I need to reschedule my appointment for next week."
3

User Query: "The weather is nice today."
0
"""
    return ChatPromptTemplate.from_template(template=template)
