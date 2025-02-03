from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

suffix = """
You are a friendly AI assistant tasked with collecting user details for an appointment. Your goal is to gather information efficiently and professionally.

Please ask the user to provide their details based on the following required fields: {required_fields}.

Present the question in a clear, concise manner, requesting all required information at once. For example:

"Could you please provide your [list of required fields separated by commas]? Please enter the information in the following format:"

Then, provide a sample entry that matches the order of the required fields. For instance:
Sample: [appropriate sample based on required fields]

Ensure your response is polite, human-like, and easy to understand. If there are any specific instructions or formats for certain fields (e.g., phone number format), please mention them clearly.

After presenting the question and sample, add a brief encouraging statement, such as:
"Thank you for your cooperation. This information will help us process your appointment efficiently."
"""

example_prompt = PromptTemplate.from_template("{output}")

examples = [
    {
        "output": "Awesome! To book a session, I need some basic info from you. Can you please provide the following details: First name, Last name, Email address, and Phone number? For example: John, Doe, john@doe.com, 98998998. Let me know!",
    },
    {
        "output": "To book your session, I need some quick details from you. Can you please provide the following:\n\nFirst Name, Last Name, Email Address, and Phone Number?\n\n(Note: Use the format: first_name, last_name, email, phone_number)",
    },
    {
        "output": "To book a session, I need some basic information from you. Can you please provide the following details in the following format: 'first name, last name, email, phone number'?",
    },
]

extraction_prompt_template = """
Please extract the following personal details from the provided text. Adhere to these guidelines:

- Extract these configured fields: {required_fields}
- For fields not mentioned, return null.
- Do not infer information that the user did not provide.

User input:
{user_input}

Output format:
{format_instructions}

Note: Be precise and avoid extracting unrelated information. If you're unsure about any detail, use "null" rather than guessing.
"""


inform_missing_prompt_template = """
You are an AI assistant designed to help users complete their personal information. Your task is to politely request the missing details in a clear, concise, and user-friendly manner.

Missing fields: {missing_fields}

Guidelines:

Example structure:
"It looks like we're missing a few details. Could you please provide your [list of missing fields]? 
For example: [example based on missing fields]"

Sample responses:
  "It looks like we're missing your last name and email. Could you please provide these? 
   For example: Doe, john@example.com"

  "We need your age and address to complete your profile. Could you please provide these? 
   For example: 30, 123 Main Street"

Please formulate an appropriate response based on the actual missing fields.
"""


missing_filler_prompt_template = """
You are an AI assistant tasked with updating a user's personal information. Your goal is to extract specific details from the user's response and update the existing information accurately.

Current information: {json_response}

Missing fields to update: {missing_fields}

Instructions:

User's response:
{user_query}

Output format:
{format_instructions}
"""


def booking_prompts(user_details_model: type) -> tuple:
    parser = JsonOutputParser(pydantic_object=user_details_model)
    inquiry = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=suffix,
        input_variables=["required_fields"],
    )
    extraction = PromptTemplate(
        template=extraction_prompt_template,
        input_variables=["required_fields", "user_input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    missing = PromptTemplate.from_template(template=inform_missing_prompt_template)
    return inquiry, extraction, missing, parser
