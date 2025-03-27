# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from autogen.agentchat.termination import Keyword, MaxTurns


def test_keyword_termination():
    keyword = Keyword("TERMINATE")
    assert keyword.is_termination_message({"content": "TERMINATE"})
    assert not keyword.is_termination_message({"content": "CONTINUE"})


def test_max_turns_termination():
    max_turns = MaxTurns(3)
    assert not max_turns.is_termination_message({})
    assert not max_turns.is_termination_message({})
    assert max_turns.is_termination_message({})


def test_or_condition():
    keyword = Keyword("TERMINATE")
    max_turns = MaxTurns(3)
    or_condition = keyword | max_turns
    assert not or_condition.is_termination_message({"content": "CONTINUE"})
    assert not or_condition.is_termination_message({"content": "CONTINUE"})
    assert or_condition.is_termination_message({"content": "CONTINUE"})
    assert or_condition.is_termination_message({"content": "TERMINATE"})


def test_and_condition():
    keyword = Keyword("TERMINATE")
    max_turns = MaxTurns(3)
    and_condition = keyword & max_turns
    assert not and_condition.is_termination_message({"content": "CONTINUE"})
    assert not and_condition.is_termination_message({"content": "TERMINATE"})
    assert not and_condition.is_termination_message({"content": "TERMINATE"})
    assert and_condition.is_termination_message({"content": "TERMINATE"})


def test_not_condition():
    keyword = Keyword("TERMINATE")
    not_condition = ~keyword
    assert not not_condition.is_termination_message({"content": "TERMINATE"})
    assert not_condition.is_termination_message({"content": "CONTINUE"})


if __name__ == "__main__":
    pytest.main()
