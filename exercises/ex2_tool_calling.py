from pprint import pprint
from textwrap import dedent
from utils import get_attendance_id

from data.projects import project_info

from langchain_aws import ChatBedrockConverse
from langchain.tools import tool, ToolException
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage


model = ChatBedrockConverse(
    model='eu.amazon.nova-pro-v1:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

# Exercise 2 - tool calling
#   In this exercise you will implement a tool with an LLM call.
#   The tool will return information about a project, given its code.
#   The user will ask the LLM to draft an email about a project.
#
#  The tool is already defined and implemented in the `get_project_info` function and `tool_config`.
#
#  * Define an agent using `create_agent` with the model and tool.
#  * Call the agent using the following prompt:
#       "Send an email from Charlie asking the contact about progress on Project P1."
#
#  * Print the response messages and make sure the tool call is successful.
#
#  * Add the `send_email` tool to the agent and call it with the same prompt.
#       * Does that change the response?
@tool
def get_project_info(project_code: str) -> dict[str, object]:
    """
    Extract project info by project code.
    """
    if project_code not in project_info:
        raise ToolException(f'Project code {project_code} not found.')

    return project_info[project_code]


@tool
def send_email(email: str, email_subject: str, email_body: str) -> str:
    """Send an email to the given email address with the given text."""
    # For the purpose of this exercise, we will just return a success message.
    email_body = dedent(f"""
        ---
        to: {email}
        subject: {email_subject}
        ---
        {email_body}
    """)

    print('Send email called, generated email is:')
    print(email_body)

    return f'Email sent to {email} with subject "{email_subject}".'


# TODO: Uncomment to print all response messages once the agent is implemented.
# for msg in response["messages"]:
#     msg.pretty_print()
