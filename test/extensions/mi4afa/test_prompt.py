# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from ag2.extensions.mi4afa import Conversation, PromptTemplate, Turn, TurnPositionError, build_prompt

from .conftest import TRIMMING_TEMPLATE, build_tokenizer, make_conversation


def _assert_positions_cover_turn_ends(conversation: Conversation, tokenizer: Any, template: PromptTemplate) -> None:
    encoding = build_prompt(conversation, tokenizer, template=template)
    text = tokenizer.apply_chat_template(template.messages(conversation), tokenize=False, add_generation_prompt=True)
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]

    search_from = 0
    for turn_index, chunk in enumerate(template.chunks(conversation)[1:-1]):
        start = text.index(chunk, search_from)
        last_char = start + len(chunk) - 1
        token_start, token_end = offsets[encoding.turn_positions[turn_index]]
        assert token_start <= last_char < token_end, f"turn {turn_index} token does not hold its last character"
        search_from = start + len(chunk)


def test_prompt_text_matches_the_reference_wording() -> None:
    conversation = Conversation(
        question="Q?",
        ground_truth="42",
        history=(Turn(" Planner ", " plan it \n"), Turn("Coder", "code")),
    )

    [system, user] = PromptTemplate().messages(conversation)

    assert system["role"] == "system"
    assert system["content"].startswith("You are a precise judge")
    assert user["content"] == (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem.\n"
        "Your task is to predict the step with the mistake that should be directly responsible for the wrong solution.\n"
        "The problem is: Q?\n"
        "The answer for the problem is: 42\n"
        "Here's the conversation history:\n"
        "0 - Planner: plan it\n"
        "1 - Coder: code\n"
        "Focus on errors that clearly derail the process.\n"
        "Respond ONLY with a number\n"
    )


def test_without_step_indices() -> None:
    chunks = PromptTemplate(with_step=False).chunks(make_conversation(2))
    assert chunks[1].startswith("Planner: ")


@pytest.mark.parametrize(
    "history",
    [
        pytest.param(make_conversation(5).history, id="plain"),
        pytest.param((Turn("A", "vérifié ✓ 日本語"), Turn("B", "naïve café")), id="unicode"),
        pytest.param((Turn("A", ""), Turn("B", "   "), Turn("C", "done.")), id="empty-content"),
        pytest.param((Turn("A", "same text"), Turn("A", "same text"), Turn("A", "same text")), id="repeated-turns"),
        pytest.param((Turn("A", "word " * 400), Turn("B", "short")), id="long-turn"),
        pytest.param((Turn("A", "ends with newline\n\n"), Turn("B", "x")), id="trailing-newlines"),
    ],
)
@pytest.mark.parametrize("chat_template", ["chatml", "trimming"])
def test_turn_positions_hold_each_turns_last_character(history: tuple[Turn, ...], chat_template: str) -> None:
    tokenizer = build_tokenizer() if chat_template == "chatml" else build_tokenizer(TRIMMING_TEMPLATE)
    conversation = Conversation(question="Q?", ground_truth="A", history=history)

    _assert_positions_cover_turn_ends(conversation, tokenizer, PromptTemplate())


def test_positions_are_strictly_increasing(tokenizer: Any) -> None:
    encoding = build_prompt(make_conversation(6), tokenizer)
    assert len(encoding.turn_positions) == 6
    assert list(encoding.turn_positions) == sorted(set(encoding.turn_positions))


def test_input_ids_match_the_tokenized_chat_template(tokenizer: Any) -> None:
    conversation = make_conversation(3)
    encoding = build_prompt(conversation, tokenizer)

    reference = tokenizer.apply_chat_template(
        PromptTemplate().messages(conversation), tokenize=True, add_generation_prompt=True
    )
    ids = reference["input_ids"] if hasattr(reference, "keys") else reference
    assert list(encoding.input_ids) == list(ids)


def test_chat_template_kwargs_reach_the_template() -> None:
    tokenizer = build_tokenizer("{{ date_string }}|" + TRIMMING_TEMPLATE)
    encoding = build_prompt(make_conversation(2), tokenizer, chat_template_kwargs={"date_string": "26 Jul 2024"})
    assert tokenizer.decode(list(encoding.input_ids)).startswith("26 Jul 2024|")


def test_template_that_rewrites_content_is_rejected() -> None:
    tokenizer = build_tokenizer("{% for message in messages %}{{ message['content'] | upper }}{% endfor %}")
    with pytest.raises(TurnPositionError, match="altered"):
        build_prompt(make_conversation(2), tokenizer)


class _SlowTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return "\n".join(message["content"] for message in messages)

    def __call__(self, text: str, **kwargs: Any) -> Any:
        raise NotImplementedError("return_offset_mapping is not available when using Python tokenizers")


def test_slow_tokenizer_is_rejected_with_a_clear_error() -> None:
    with pytest.raises(TurnPositionError, match="fast tokenizer"):
        build_prompt(make_conversation(2), _SlowTokenizer())


def test_conversation_validation() -> None:
    with pytest.raises(ValueError, match="at least one turn"):
        Conversation(question="q", ground_truth="a", history=())
    with pytest.raises(ValueError, match="outside"):
        Conversation(question="q", ground_truth="a", history=(Turn("A", "x"),), mistake_step=1)


def test_from_who_and_when_record() -> None:
    record = {
        "question": "Q",
        "ground_truth": 55,
        "history": [{"name": "Planner", "content": "c", "role": "assistant"}, {"content": "no name"}],
        "mistake_step": "1",
        "mistake_agent": "Planner",
        "mistake_reason": "ignored",
    }

    conversation = Conversation.from_who_and_when(record)

    assert conversation.ground_truth == "55"
    assert conversation.history == (Turn("Planner", "c"), Turn("Unknown Agent", "no name"))
    assert conversation.mistake_step == 1
    assert conversation.mistake_agent == "Planner"


def test_history_given_as_list_is_stored_as_tuple() -> None:
    conversation = Conversation(question="q", ground_truth="a", history=[Turn("A", "x")])  # type: ignore[arg-type]
    assert conversation.history == (Turn("A", "x"),)
