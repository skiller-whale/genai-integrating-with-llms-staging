from pprint import pprint
from utils import get_attendance_id

from data.projects import project_info

from langchain_aws import ChatBedrock
from langchain.tools import tool, ToolException
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage


model = ChatBedrock(
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
#       "Draft an email asking the contact about progress on Project P1."
#
#  * Print the response messages and make sure the tool call is successful.
@tool
def get_project_info(project_code: str) -> dict[str, object]:
    """
    Extract project info by project code.
    """
    if project_code not in project_info:
        raise ToolException(f'Project code {project_code} not found.')

    return project_info[project_code]

# TODO: Uncomment to print all response messages once the agent is implemented.
# for msg in response["messages"]:
#     msg.pretty_print()
