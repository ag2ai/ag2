# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from textual import on
from textual.app import App, ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Header, Input, Markdown

from autogen.beta import Agent, Context, Conversation, MemoryStream
from autogen.beta.events import ModelMessageChunk, ModelReasoning


class TUIAgent(App):
    def __init__(self, agent: Agent):
        super().__init__()

        self.stream = MemoryStream()
        self.agent = agent
        self.conversation: Conversation | None = None

    def on_mount(self) -> None:
        self.title = f"AG2 TUI of `{self.agent.name}` agent"
        self.query_one("#input", Input).focus()

    def compose(self) -> ComposeResult:
        yield Header(name=self.agent.name)
        yield ScrollableContainer(id="chat_history")
        yield Input(
            placeholder="Type message to your agent",
            id="input",
        )

    @on(Input.Submitted)
    async def input_submitter(self, event: Input.Submitted) -> None:
        if text := event.value:
            inp = event.input
            inp.disabled = True
            inp.clear()

            chat_container = self.query_one("#chat_history", ScrollableContainer)

            # Add user message to history
            user_message = Markdown(f"**You:** {text}", classes="user-message")
            await chat_container.mount(user_message)
            user_message.scroll_visible(immediate=True)

            # Create a buffer block for assistant streaming
            thinking_block = Markdown(f"**{self.agent.name}:** *Thinking...* ", classes="assistant-message")
            await chat_container.mount(thinking_block)
            thinking_block.scroll_visible(immediate=True)

            async def put_reasoning_chunk(event: ModelReasoning, ctx: Context) -> None:
                await thinking_block.append(event.content)

            # Create agent message block
            assistant_message = Markdown(f"**{self.agent.name}:** ", classes="assistant-message")
            text_mounted = False

            async def put_message_chunk(event: ModelMessageChunk, ctx: Context) -> None:
                nonlocal text_mounted
                if not text_mounted:
                    await thinking_block.remove()
                    await chat_container.mount(assistant_message)
                    text_mounted = True

                await assistant_message.append(event.content)
                assistant_message.scroll_visible(immediate=True)

            try:
                with (
                    self.stream.where(ModelMessageChunk).sub_scope(put_message_chunk),
                    self.stream.where(ModelReasoning).sub_scope(put_reasoning_chunk),
                ):
                    c = self.conversation = await (
                        self.conversation.ask(text) if self.conversation else self.agent.ask(text, stream=self.stream)
                    )
                    if not text_mounted:
                        await thinking_block.remove()
                        await chat_container.mount(assistant_message)

                    final_content = c.message.message.content
                    await assistant_message.update(f"**{self.agent.name}:** {final_content}")
                    assistant_message.scroll_visible(immediate=True)

            finally:
                inp.disabled = False
                inp.focus()
