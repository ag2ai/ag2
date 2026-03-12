import asyncio
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from pydantic import Field

from autogen.beta import Agent, MemoryStream, Variable
from autogen.beta.config import OpenAIConfig
from autogen.beta.events import ModelRequest, ModelResponse
from autogen.beta.middleware import LoggingMiddleware

MEMORY_DIR = Path("./memory")

logging.basicConfig(level=logging.INFO)

agent = Agent(
    "test-agent",
    config=OpenAIConfig(
        "gpt-5-nano",
        reasoning_effort="low",
        streaming=True,
    ),
    middleware=[LoggingMiddleware(logger=logging.getLogger("ambient"))],
)


@agent.tool
def read_conversation_history(
    date: Annotated[
        date,
        Field(
            default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            description="The date to read the conversation history for",
        ),
    ],
    user_id: Annotated[int, Variable()],
) -> str:
    """Read current user's conversation history for the given date (summaries from previous sessions).
    Use this when the user asks what was discussed earlier on the given date or to recall prior context.
    """
    path = MEMORY_DIR / str(user_id) / date.strftime("%Y-%m-%d.md")
    if not path.exists():
        return "No conversation history saved for today yet."
    return path.read_text(encoding="utf-8")


dp = Dispatcher()
chat_state: dict[int, MemoryStream] = {}


def _get_or_create_chat(chat_id: int) -> MemoryStream:
    if chat_id not in chat_state:
        chat_state[chat_id] = MemoryStream()
    return chat_state[chat_id]


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    _get_or_create_chat(message.chat.id)
    await message.answer("Hi, there! Send me any message and I'll reply using the agent.")


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    stream = chat_state.pop(message.chat.id, None)
    if stream is not None:
        await _write_conversation_to_memory(message.chat.id, stream)
    await message.answer("Context dropped. Next message will start a new conversation.")


@dp.message(F.text)
async def on_text(message: Message) -> None:
    if not message.text:
        return
    chat_stream = _get_or_create_chat(message.chat.id)

    status = await message.answer("Thinking…")
    try:
        reply = await agent.ask(
            message.text,
            stream=chat_stream,
            variables={"user_id": message.chat.id},
        )
        text = reply.content or "(no response)"

        await status.edit_text(text)
    except Exception as e:
        await status.edit_text(f"Error: {e}")


async def _write_conversation_to_memory(user_id: int, stream: MemoryStream) -> None:
    """Write current conversation transcript to folder-based memory."""
    # summarise the conversation
    events = await stream.history.get_events()
    lines: list[str] = []
    for ev in events:
        if isinstance(ev, ModelRequest):
            lines.append(f"**user:** {ev.content}")
        elif isinstance(ev, ModelResponse) and ev.content:
            lines.append(f"**assistant:** {ev.content}")
    if not lines:
        return

    # use `./memory/user_id/YYYY-MM-DD.md` as the today's conversation history file
    now = datetime.now(timezone.utc)
    user_dir = MEMORY_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / now.strftime("%Y-%m-%d.md")
    content = "\n\n".join(lines)

    def _write() -> None:
        # append current history to the file
        with path.open("a", encoding="utf-8") as f:
            if f.tell() != 0:
                f.write("\n\n---\n\n")
            f.write(f"## {now.strftime('%H:%M:%S')} UTC\n\n")
            f.write(content)

    await asyncio.to_thread(_write)


if __name__ == "__main__":
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

    asyncio.run(dp.start_polling(bot))
