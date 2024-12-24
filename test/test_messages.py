from unittest.mock import MagicMock
from autogen.agentchat.conversable_agent import ConversableAgent
import autogen.messages
from autogen.formatting_utils import colored

import pytest

# def test_context():
#     agent = ConversableAgent("a0", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
#     agent1 = ConversableAgent("a1", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")
#     m1 = {
#             "content": "hello {name}",
#             "context": {
#                 "name": "there",
#             },
#         }

#     actual = autogen.messages.create_message_model(m1, agent)

#     expected = autogen.messages.BaseMessage(content="hello there")

#     # expect hello {name} to be printed
#     agent1.send(
#         {
#             "content": lambda context: f"hello {context['name']}",
#             "context": {
#                 "name": "there",
#             },
#         },
#         agent,
#     )
#     # expect hello there to be printed
#     agent.llm_config = {"allow_format_str_template": True}
#     agent1.send(
#         {
#             "content": "hello {name}",
#             "context": {
#                 "name": "there",
#             },
#         },
#         agent,
#     )
#     # expect hello there to be printed

@pytest.fixture
def sender() -> ConversableAgent:
    return ConversableAgent("sender", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

@pytest.fixture
def receiver() -> ConversableAgent:
    return ConversableAgent("receiver", max_consecutive_auto_reply=0, llm_config=False, human_input_mode="NEVER")

def test_tool_responses(sender: ConversableAgent, receiver: ConversableAgent):
    message = {
        "role": "tool",
        "tool_responses": [
            {"tool_call_id": "call_rJfVpHU3MXuPRR2OAdssVqUV", "role": "tool", "content": "Timer is done!"},
            {"tool_call_id": "call_zFZVYovdsklFYgqxttcOHwlr", "role": "tool", "content": "Stopwatch is done!"},
        ],
        "content": "Timer is done!\\n\\nStopwatch is done!",
    }
    actual = autogen.messages.create_message_model(message, sender=sender, receiver=receiver)

    assert isinstance(actual, autogen.messages.ToolResponseMessage)
    assert actual.role == "tool"
    assert actual.sender_name == "sender"
    assert actual.receiver_name == "receiver"
    assert actual.content == "Timer is done!\\n\\nStopwatch is done!"
    assert len(actual.tool_responses) == 2

    assert isinstance(actual.tool_responses[0], autogen.messages.ToolResponse)
    assert actual.tool_responses[0].tool_call_id == "call_rJfVpHU3MXuPRR2OAdssVqUV"
    assert actual.tool_responses[0].role == "tool"
    assert actual.tool_responses[0].content == "Timer is done!"

    assert isinstance(actual.tool_responses[1], autogen.messages.ToolResponse)
    assert actual.tool_responses[1].tool_call_id == "call_zFZVYovdsklFYgqxttcOHwlr"
    assert actual.tool_responses[1].role == "tool"
    assert actual.tool_responses[1].content == "Stopwatch is done!"

    expected_print = f"""{colored("user_proxy", "yellow")} (to chatbot):

{colored("***** Response from calling tool (call_rJfVpHU3MXuPRR2OAdssVqUV) *****", "green")}
Timer is done!
{colored("**********************************************************************", "green")}

--------------------------------------------------------------------------------
{colored("***** Response from calling tool (call_zFZVYovdsklFYgqxttcOHwlr) *****", "green")}
Stopwatch is done!
{colored("**********************************************************************", "green")}

--------------------------------------------------------------------------------"""
    
    mock = MagicMock()
    actual.print(f=mock)

    expected_call_args_list = [
        (('\x1b[33msender\x1b[0m (to receiver):\n',), {'flush': True}),
        (('\x1b[32m***** Response from calling tool (call_rJfVpHU3MXuPRR2OAdssVqUV) *****\x1b[0m',), {'flush': True}),
        (('Timer is done!',), {'flush': True}),
        (('\x1b[32m**********************************************************************\x1b[0m',), {'flush': True}),
        (('\n', '--------------------------------------------------------------------------------'), {'flush': True, 'sep': ''}),
        (('\x1b[32m***** Response from calling tool (call_zFZVYovdsklFYgqxttcOHwlr) *****\x1b[0m',), {'flush': True}),
        (('Stopwatch is done!',), {'flush': True}),
        (('\x1b[32m**********************************************************************\x1b[0m',), {'flush': True}),
        (('\n', '--------------------------------------------------------------------------------'), {'flush': True, 'sep': ''})
    ]

    assert mock.call_args_list == expected_call_args_list
