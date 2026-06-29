from pprint import pprint
from utils import get_attendance_id

from langchain_aws import ChatBedrock
from langchain.messages import HumanMessage, SystemMessage

from pydantic import BaseModel

from data.animals import animal_data

model = ChatBedrock(
    model='eu.amazon.nova-pro-v1:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)

# Exercise 3 - structured output, text to JSON
#
#   The `animal_data` list imported above is a text description of animals.
#   You will convert it to JSON using an LLM.
#
# PART 1 - defining the output models
#   Define the `Animal` model with these fields:
#       * `name`, type str
#       * `max_age`, type int
#
#   Define the `Animals` model with one field:
#       * `animals`, a list of `Animal` objects
#
#   * Use `model.with_structured_output(Animals)` to create a model that returns
#     structured output.
#   * Invoke the model with a system message telling the LLM to convert the
#     animal data to JSON, and a human message containing `animal_data`.
#   * Run this code and make sure the LLM extracts the appropriate JSON.
#
# PART 2 - diet
#
#   * Now add a field to `Animal` for `diet` - a list of strings.
#   * Run the code again and make sure the LLM extracts the appropriate JSON.
#
#
# HINT: Print `response.model_dump(mode="json")` to see the JSON output.

class Animal(BaseModel):
    # TODO: define the animal class
    pass


class Animals(BaseModel):
    # TODO: animals should contain a list of "Animal"
    pass


# TODO: Create a structured-output model, invoke it with the animal data, and
# print the JSON output.
