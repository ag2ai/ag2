# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys

import pytest

import ag2.extensions


def test_torch_free_api_imports_and_probing_api_hints_install(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [name for name in sys.modules if name.startswith("ag2.extensions.mi4afa")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(ag2.extensions, "mi4afa", None, raising=False)

    module = importlib.import_module("ag2.extensions.mi4afa")

    conversation = module.Conversation(question="q", ground_truth="a", history=(module.Turn("A", "x"),))
    assert module.PromptTemplate().chunks(conversation)[1] == "0 - A: x"
    for name in ("ProbeAttributor", "ActivationExtractor", "LogisticProbe", "probe_failure_attribution"):
        with pytest.raises(ImportError, match=r'pip install "torch>=2.4,<3" "transformers>=4.56,<6"'):
            getattr(module, name)()
