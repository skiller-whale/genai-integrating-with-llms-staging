import time
from pprint import pprint
from utils import get_attendance_id

from langchain_aws import ChatBedrockConverse
from langchain.messages import AIMessage, HumanMessage, SystemMessage

# Exercise 4 - Streaming/Blocking
#
# PART 1
#   * Run this script and make note of how long it takes to get a response from the LLM.
#
# PART 2
#   * Uncomment the rest of `user_messages` and run the for-loop.
#       * Is there a noticeable difference in time-to-first token for each message?
#

model = ChatBedrockConverse(
    model='eu.amazon.nova-pro-v1:0',
    endpoint_url='https://bedrock-runtime.aws-proxy.skillerwhale.com/',
    region_name='eu-west-1',
    aws_access_key_id=get_attendance_id(),
    aws_secret_access_key='<unused>',
)


# PART 2 - Uncomment the rest of `user_messages`.
user_messages = [
    HumanMessage('Explain the major limitations of LLMs.'),
    # HumanMessage('Write Clojure code to find the shortest route between two points in a network. Be concise.'),
    # HumanMessage('What is the smallest species of whale? Explain thoroughly.')
]


for user_message in user_messages:
    times_per_chunk = []

    # Send the message to the model, using a basic inference configuration.
    start_time = time.perf_counter()
    if len(user_messages) == 1:
        print('LLM Response, streamed:')
        print('-' * 50)

    for chunk in model.stream([user_message]):
        if len(user_messages) == 1:
            print(chunk.text, end="|", flush=True)

        times_per_chunk.append(time.perf_counter() - start_time)
        start_time = time.perf_counter()

    if len(user_messages) == 1:
        print()
        print('-' * 50)


    # BLOCKING
    if len(user_messages) == 1:
        print('LLM Response, blocking:')
        print('-' * 50)

    start_time_block = time.perf_counter()
    blocking_response = model.invoke([user_message])
    total_time_block = time.perf_counter() - start_time_block

    if len(user_messages) == 1:
        blocking_response.pretty_print()
        print('-' * 50)

    print()
    print('Message: ', user_message.content)
    print(f'Time to first chunk (streamed): {times_per_chunk[0]:.4f}')
    print(f'Time to full output (blocking): {total_time_block:.4f}')
    print('-' * 50)
    print()
