# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import json
from pprint import pprint
from tempfile import TemporaryDirectory
from typing import Any, Callable, Optional
from unittest.mock import MagicMock
from uuid import UUID

import autogen
from autogen.cache.cache import Cache
from autogen.events.base_event import BaseEvent, wrap_event
from autogen.import_utils import optional_import_block, run_for_optional_imports
from autogen.io import IOWebsockets

from ..conftest import Credentials

# Check if the websockets module is available
with optional_import_block() as result:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect as ws_connect


@wrap_event
class TestTextEvent(BaseEvent):
    text: str

    def __init__(self, *, uuid: Optional[UUID] = None, text: str):
        super().__init__(uuid=uuid, text=text)

    def print(self, f: Optional[Callable[..., Any]] = None) -> None:
        f = f or print

        f(self.text)


@run_for_optional_imports(["websockets"], "websockets")
class TestConsoleIOWithWebsockets:
    def test_input_print(self) -> None:
        print()
        print("Testing input/print", flush=True)

        def on_connect(iostream: IOWebsockets) -> None:
            print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)

            print(" - on_connect(): Receiving message from client.", flush=True)

            msg = iostream.input()

            print(f" - on_connect(): Received message '{msg}' from client.", flush=True)

            assert msg == "Hello world!"

            for msg in ["Hello, World!", "Over and out!"]:
                print(f" - on_connect(): Sending message '{msg}' to client.", flush=True)

                text_message = TestTextEvent(text=msg)
                text_message.print(iostream.print)

            print(" - on_connect(): Receiving message from client.", flush=True)

            msg = iostream.input("May I?")

            print(f" - on_connect(): Received message '{msg}' from client.", flush=True)
            assert msg == "Yes"

            return

        with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8765) as uri:
            print(f" - test_setup() with websocket server running on {uri}.", flush=True)

            with ws_connect(uri) as websocket:
                print(f" - Connected to server on {uri}", flush=True)

                print(" - Sending message to server.", flush=True)
                websocket.send("Hello world!")

                for expected in ["Hello, World!", "Over and out!", "May I?"]:
                    print(" - Receiving message from server.", flush=True)
                    message = websocket.recv()
                    message = message.decode("utf-8") if isinstance(message, bytes) else message
                    # drop the newline character
                    if message.endswith("\n"):
                        message = message[:-1]

                    print(
                        f"   - Asserting received message '{message}' is the same as the expected message '{expected}'",
                        flush=True,
                    )
                    try:
                        message_dict = json.loads(message)
                        actual = message_dict["content"]["objects"][0]
                    except json.JSONDecodeError:
                        actual = message
                    assert actual == expected

                print(" - Sending message 'Yes' to server.", flush=True)
                websocket.send("Yes")

        print("Test passed.", flush=True)

    @run_for_optional_imports("openai", "openai")
    def test_chat(self, credentials_gpt_4o_mini: Credentials) -> None:
        print("Testing setup", flush=True)

        mock = MagicMock()

        def on_connect(iostream: IOWebsockets) -> None:
            print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)

            print(" - on_connect(): Receiving message from client.", flush=True)

            initial_msg = iostream.input()

            llm_config = {
                "config_list": credentials_gpt_4o_mini.config_list,
                "stream": True,
            }

            agent = autogen.ConversableAgent(
                name="chatbot",
                system_message="Complete a task given to you and reply TERMINATE when the task is done.",
                llm_config=llm_config,
            )

            # create a UserProxyAgent instance named "user_proxy"
            user_proxy = autogen.UserProxyAgent(
                name="user_proxy",
                system_message="A proxy for the user.",
                is_termination_msg=lambda x: x.get("content", "")
                and x.get("content", "").rstrip().endswith("TERMINATE"),
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
            )

            # we will use a temporary directory as the cache path root to ensure fresh completion each time
            with TemporaryDirectory() as cache_path_root:  # noqa: SIM117
                with Cache.disk(cache_path_root=cache_path_root) as cache:
                    print(
                        f" - on_connect(): Initiating chat with agent {agent} using message '{initial_msg}'",
                        flush=True,
                    )
                    try:
                        user_proxy.initiate_chat(
                            agent,
                            message=initial_msg,
                            cache=cache,
                        )
                    except Exception as e:
                        print(f" - on_connect(): Exception {e} raised during chat.", flush=True)
                        import traceback

                        print(traceback.format_exc())
                        raise e

            print(" - on_connect(): Chat completed with success.", flush=True)
            mock("Success")

            return

        with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8765) as uri:
            print(f" - test_setup() with websocket server running on {uri}.", flush=True)

            with ws_connect(uri) as websocket:
                print(f" - Connected to server on {uri}", flush=True)

                print(" - Sending message to server.", flush=True)
                # websocket.send("2+2=?")
                websocket.send("Please write a poem about spring in a city of your choice.")

                while True:
                    try:
                        message = websocket.recv()
                        message = message.decode("utf-8") if isinstance(message, bytes) else message
                        message_dict = json.loads(message)
                        # drop the newline character
                        # if message.endswith("\n"):
                        #     message = message[:-1]

                        print("*" * 80)
                        print("Received message:")
                        pprint(message_dict)
                        print()

                        # if "TERMINATE" in message:
                        #     print()
                        #     print(" - Received TERMINATE message.", flush=True)
                    except ConnectionClosed as e:
                        print("Connection closed:", e, flush=True)
                        break

        mock.assert_called_once_with("Success")
        print("Test passed.", flush=True)
