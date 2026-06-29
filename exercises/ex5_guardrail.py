import boto3
from utils import get_attendance_id

from langchain_aws import ChatBedrock
from langchain.messages import HumanMessage, SystemMessage


REGION = 'eu-west-1'
BEDROCK_ENDPOINT = 'https://bedrock.aws-proxy.skillerwhale.com/'
BEDROCK_RUNTIME_ENDPOINT = 'https://bedrock-runtime.aws-proxy.skillerwhale.com/'
AWS_ACCESS_KEY_ID = get_attendance_id()

# Exercise 5 - guardrails
#
# The code below retrieves and uses a guardrail.
#   It has one single goal - never talk about orange juice.
#
#   Can you break it?
#
# For example:
#   * Try talking about orange juice without talking about orange juice.
#   * Try asking the LLM to output the first letters of each word in a message,
#       that spells out 'orange juice'.
#
# Some tips/things to look out for.
#
#   * Is the error different when you mention orange juice in the prompt
#       versus when it would appear in the LLM output?
#   * Try it _without_ the guardrail too - see if the model talks about
#       orange juice in the output or not.
#
session = boto3.Session(
    region_name=REGION, aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key='<unused>'
)

# client to get guardrail info
client = session.client(
    service_name='bedrock', region_name=REGION, endpoint_url=BEDROCK_ENDPOINT
)
runtime_client = session.client(
    service_name='bedrock-runtime', region_name=REGION, endpoint_url=BEDROCK_RUNTIME_ENDPOINT
)


# get guardrail info
guardrails = client.list_guardrails()['guardrails']
orange_juice_guardrail = None
for guardrail in guardrails:
    if guardrail.get('name') == 'orange_juice':
        orange_juice_guardrail = guardrail
        break

if orange_juice_guardrail is None:
    raise ValueError('Cannot find orange_juice guardrail!')


# instantiate model
model = ChatBedrock(
    model='eu.amazon.nova-pro-v1:0',
    client=runtime_client,
    guardrails={
        'guardrailIdentifier': orange_juice_guardrail['id'],
        'guardrailVersion': orange_juice_guardrail['version']
    }
)

# invoke model, trying to break the guardrail
messages = [
    SystemMessage('Answer any questions briefly.'),
    HumanMessage('How do I make orange juice?')
]

res = model.invoke(messages)
res.pretty_print()
