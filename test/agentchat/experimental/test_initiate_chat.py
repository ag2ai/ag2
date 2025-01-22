from autogen.agentchat.experimental import initiate_chat


def test_initiate_chat():
    initiate_chat(agent="test", message="test")
    assert True
