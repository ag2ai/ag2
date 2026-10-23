# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Render a conversation as a judge prompt and locate each turn's token.

The probed model reads the whole conversation as one chat-templated judge
prompt. A probe reads one activation per turn, taken at the token holding the
turn's last character, so the activation summarizes everything up to and
including that turn.

Turn tokens are located from the tokenizer's character offsets rather than by
decoding and string-matching token windows, so the result is exact for any
fast tokenizer and fails loudly instead of drifting.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .types import Conversation

__all__ = (
    "DEFAULT_POSTFIX",
    "DEFAULT_PREFIX",
    "DEFAULT_SYSTEM_PROMPT",
    "PromptEncoding",
    "PromptTemplate",
    "TurnPositionError",
    "build_prompt",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise judge that identifies the most critical mistake in multi-agent conversations "
    "that fail to derive the correct answer."
)
DEFAULT_PREFIX = (
    "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
    "Your task is to predict the step with the mistake that should be directly responsible for the wrong solution.\n"
)
DEFAULT_POSTFIX = "Focus on errors that clearly derail the process.\nRespond ONLY with a number\n"


class TurnPositionError(ValueError):
    """Raised when a turn cannot be located in the tokenized prompt."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Wording of the judge prompt.

    Attributes:
        system: System message.
        prefix: Text before the task, answer and conversation.
        postfix: Text after the conversation.
        with_step: Prefix each turn with its index (``"3 - name: ..."``).
    """

    system: str = DEFAULT_SYSTEM_PROMPT
    prefix: str = DEFAULT_PREFIX
    postfix: str = DEFAULT_POSTFIX
    with_step: bool = True

    def chunks(self, conversation: Conversation) -> list[str]:
        """Return the user-message pieces: header, one piece per turn, footer.

        The user message is these pieces joined by newlines.
        """
        header = (
            f"{self.prefix}"
            f"The problem is: {conversation.question}\n"
            f"The answer for the problem is: {conversation.ground_truth}\n"
            "Here's the conversation history:"
        )
        turns = [
            f"{index} - {turn.name.strip()}: {turn.content.strip()}"
            if self.with_step
            else f"{turn.name.strip()}: {turn.content.strip()}"
            for index, turn in enumerate(conversation.history)
        ]
        return [header, *turns, self.postfix]

    def messages(self, conversation: Conversation) -> list[dict[str, str]]:
        """Return the chat messages (system + user) for ``conversation``."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": "\n".join(self.chunks(conversation))},
        ]


@dataclass(frozen=True, slots=True)
class PromptEncoding:
    """A tokenized judge prompt.

    Attributes:
        input_ids: Token ids of the full chat-templated prompt.
        turn_positions: For each turn, the index into ``input_ids`` of the
            token holding the turn's last character.
    """

    input_ids: tuple[int, ...]
    turn_positions: tuple[int, ...]


def build_prompt(
    conversation: Conversation,
    tokenizer: Any,
    *,
    template: PromptTemplate | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> PromptEncoding:
    """Tokenize ``conversation`` as a judge prompt and locate every turn.

    Args:
        conversation: The conversation to render.
        tokenizer: A Hugging Face *fast* tokenizer with a chat template.
        template: Prompt wording; defaults to :class:`PromptTemplate`.
        chat_template_kwargs: Extra variables for the chat template. Pin
            date-dependent templates here (e.g. Llama 3's ``date_string``) to
            make activations reproducible.

    Returns:
        The token ids and one token position per turn.

    Raises:
        TurnPositionError: If the tokenizer has no character offsets, or the
            chat template rewrote the conversation so a turn cannot be found.
    """
    template = template or PromptTemplate()
    chunks = template.chunks(conversation)
    user_content = "\n".join(chunks)
    text = tokenizer.apply_chat_template(
        template.messages(conversation),
        tokenize=False,
        add_generation_prompt=True,
        **dict(chat_template_kwargs or {}),
    )
    content_start = _locate_content(text, user_content)

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    except NotImplementedError as e:
        raise TurnPositionError("locating turns needs a fast tokenizer with offset mapping") from e

    turn_ends = [content_start + end for end in _chunk_ends(chunks)[1:-1]]
    turn_starts = [end - len(chunk) for end, chunk in zip(turn_ends, chunks[1:-1], strict=True)]
    positions = _last_tokens(encoded["offset_mapping"], turn_starts, turn_ends)
    return PromptEncoding(input_ids=tuple(encoded["input_ids"]), turn_positions=positions)


def _locate_content(text: str, user_content: str) -> int:
    """Return where ``user_content`` starts inside the rendered chat ``text``.

    Chat templates commonly trim message content, so the stripped content is
    searched and the offset corrected for any stripped leading whitespace.
    """
    stripped = user_content.strip()
    found = text.find(stripped)
    if found < 0:
        raise TurnPositionError("the chat template altered the conversation text; turns cannot be located")
    return found - (len(user_content) - len(user_content.lstrip()))


def _chunk_ends(chunks: Sequence[str]) -> list[int]:
    ends: list[int] = []
    position = 0
    for chunk in chunks:
        position += len(chunk)
        ends.append(position)
        position += 1  # the "\n" joining chunks
    return ends


def _last_tokens(
    offsets: Sequence[tuple[int, int]], turn_starts: Sequence[int], turn_ends: Sequence[int]
) -> tuple[int, ...]:
    """For each turn span, return the last token that starts inside the text up to the span's end.

    Tokens with empty spans (special tokens on some tokenizers) are skipped.
    """
    positions: list[int] = []
    token = 0
    last = -1
    for turn_index, (start, end) in enumerate(zip(turn_starts, turn_ends, strict=True)):
        while token < len(offsets) and offsets[token][0] < end:
            if offsets[token][1] > offsets[token][0]:
                last = token
            token += 1
        if last < 0 or offsets[last][1] <= start:
            raise TurnPositionError(f"no token overlaps turn {turn_index}")
        positions.append(last)
    return tuple(positions)
