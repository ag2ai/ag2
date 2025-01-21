from autogen.messages.input_messages import PasswordInputRequestMessage, TextInputRequestMessage


class TestTextInputRequestMessage:
    def test_init(self) -> None:
        message = TextInputRequestMessage(prompt="Write something:")
        assert message.content.uuid is not None


class TestPasswordInputRequestMessage:
    def test_init(self) -> None:
        message = PasswordInputRequestMessage(prompt="Password:")
        assert message.content.uuid is not None
