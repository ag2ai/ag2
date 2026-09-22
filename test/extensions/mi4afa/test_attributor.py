# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import torch

from ag2.extensions.mi4afa import (
    ActivationExtractor,
    ActivationSite,
    ConversationActivations,
    ProbeAttributor,
)
from ag2.extensions.mi4afa.attributor import _split

from .conftest import make_conversation

SITE_A = ActivationSite(0, "resid_pre")
SITE_B = ActivationSite(0, "mlp_in")
SITES = (SITE_A, SITE_B)


class _NoModelExtractor:
    """Stands in for an extractor when activations are supplied directly; running the model is an error."""

    model = None

    def extract(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("the model must not run when activations are given")


def _synthetic(
    count: int, informative: Iterable[ActivationSite] | dict[ActivationSite, float], *, seed: int, turns: int = 5
) -> list[Any]:
    """Random activations where the mistake turn stands out along one direction at the informative sites."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    strength = informative if isinstance(informative, dict) else dict.fromkeys(informative, 6.0)
    examples = []
    for index in range(count):
        conversation = make_conversation(turns, mistake_step=rng.randrange(turns), tag=f"{seed}-{index}")
        values = torch.randn(len(SITES), turns, 6, generator=generator)
        for site_index, site in enumerate(SITES):
            values[site_index, conversation.mistake_step, 0] += strength.get(site, 0.0)
        examples.append(ConversationActivations(conversation=conversation, sites=SITES, values=values))
    return examples


def _attributor(**options: Any) -> ProbeAttributor:
    return ProbeAttributor(_NoModelExtractor(), device="cpu", **options)  # type: ignore[arg-type]


def test_fit_selects_the_site_on_validation_not_test() -> None:
    # site A is strong on train but absent on test; site B is weak on train and strong on test
    train = _synthetic(60, informative={SITE_A: 6.0, SITE_B: 1.5}, seed=1)
    test = _synthetic(40, informative={SITE_B: 6.0}, seed=2)
    attributor = _attributor()

    report = attributor.fit(train, validation_fraction=0.25, seed=7)
    sweep = attributor.sweep_sites(train, test)

    assert report.site == SITE_A
    assert attributor.site == SITE_A
    assert (report.train_count, report.validation_count) == (60, 15)
    assert [score.site for score in report.validation] == list(SITES)
    assert report.validation[0].step_accuracy > report.validation[1].step_accuracy
    # sweep_sites scores on test, so it would have picked site B — fit must not
    by_site = {score.site: score for score in sweep}
    assert by_site[SITE_B].step_accuracy > by_site[SITE_A].step_accuracy
    assert attributor.site == SITE_A  # sweep leaves the fitted state untouched


def test_evaluate_scores_the_fitted_site() -> None:
    train = _synthetic(60, informative=SITES, seed=3)
    test = _synthetic(30, informative=SITES, seed=4)
    attributor = _attributor()
    attributor.fit(train)

    score = attributor.evaluate(test)

    assert score.site == attributor.site
    assert score.count == 30
    assert score.step_accuracy > 0.9
    assert score.agent_accuracy >= score.step_accuracy


def test_explicit_validation_set_selects_the_site_and_is_not_trained_on() -> None:
    train = _synthetic(40, informative=SITES, seed=5)
    validation = _synthetic(20, informative=[SITE_B], seed=6)
    attributor = _attributor()

    report = attributor.fit(train, validation=validation)

    assert report.site == SITE_B
    assert (report.train_count, report.validation_count) == (40, 20)


def test_ties_prefer_the_earlier_site() -> None:
    examples = _synthetic(20, informative=SITES, seed=8)
    duplicated = [
        ConversationActivations(example.conversation, SITES, example.values[:1].repeat(2, 1, 1)) for example in examples
    ]

    report = _attributor(epochs=20).fit(duplicated)

    assert report.validation[0].step_accuracy == report.validation[1].step_accuracy
    assert report.site == SITE_A


def test_attribute_names_the_decisive_step_and_its_agent() -> None:
    attributor = _attributor()
    attributor.fit(_synthetic(60, informative=SITES, seed=9))
    [example] = _synthetic(1, informative=SITES, seed=10)

    attribution = attributor.attribute(example)

    step = example.conversation.mistake_step
    assert attribution.failed is True
    assert attribution.error_mode == "other"
    assert attribution.decisive_step == step
    assert attribution.responsible_agent == example.conversation.history[step].name
    assert str(attributor.site) in attribution.reasoning
    assert attributor.score_turns(example).shape == (5,)


def test_split_keeps_conversations_whole_and_disjoint() -> None:
    examples = _synthetic(23, informative=SITES, seed=11)

    fit, held = _split(examples, 0.2, seed=3)

    assert len(held) == 5
    assert len(fit) + len(held) == 23
    assert not {id(example) for example in fit} & {id(example) for example in held}
    again_fit, again_held = _split(examples, 0.2, seed=3)
    assert again_held == held
    assert again_fit == fit


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
def test_split_rejects_degenerate_fractions(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        _split(_synthetic(5, informative=SITES, seed=0), fraction, seed=0)


def test_split_needs_two_conversations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _split(_synthetic(1, informative=SITES, seed=0), 0.5, seed=0)


def test_unlabelled_training_data_is_rejected() -> None:
    [example] = _synthetic(1, informative=SITES, seed=0)
    unlabelled = ConversationActivations(make_conversation(5, mistake_step=None), SITES, example.values)
    with pytest.raises(ValueError, match="mistake_step"):
        _attributor().fit([unlabelled, unlabelled])


def test_examples_must_share_sites() -> None:
    examples = _synthetic(4, informative=SITES, seed=0)
    examples.append(ConversationActivations(examples[0].conversation, (SITE_A,), examples[0].values[:1]))
    with pytest.raises(ValueError, match="same sites"):
        _attributor().sweep_sites(examples, examples)


def test_unfitted_attributor_refuses_to_attribute() -> None:
    [example] = _synthetic(1, informative=SITES, seed=0)
    attributor = _attributor()
    assert not attributor.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        attributor.attribute(example)


def _real_attributor(model: Any, tokenizer: Any) -> ProbeAttributor:
    extractor = ActivationExtractor(model, tokenizer, chat_template_kwargs={"date_string": "26 Jul 2024"})
    return ProbeAttributor(extractor, epochs=20)


def _real_conversations(count: int) -> list[Any]:
    return [make_conversation(3 + index % 3, mistake_step=index % 3, tag=str(index)) for index in range(count)]


def test_fit_and_attribute_with_a_real_model(model: Any, tokenizer: Any) -> None:
    attributor = _real_attributor(model, tokenizer)
    conversations = _real_conversations(8)

    report = attributor.fit(conversations)

    assert report.site in attributor.extractor.available_sites()
    assert len(report.validation) == len(attributor.extractor.available_sites())
    attribution = attributor.attribute(conversations[0])
    assert 0 <= attribution.decisive_step < len(conversations[0].history)
    assert attributor.evaluate(conversations[:2]).count == 2


def test_save_and_load_round_trip(model: Any, tokenizer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model.config, "_name_or_path", "org/tiny-llama")
    attributor = _real_attributor(model, tokenizer)
    conversations = _real_conversations(8)
    attributor.fit(conversations)

    attributor.save(tmp_path / "probe")
    loaded = ProbeAttributor.load(tmp_path / "probe", model, tokenizer)

    assert sorted(path.name for path in (tmp_path / "probe").iterdir()) == ["probe.json", "probe.safetensors"]
    assert loaded.site == attributor.site
    assert loaded.extractor.chat_template_kwargs == {"date_string": "26 Jul 2024"}
    assert loaded.extractor.template == attributor.extractor.template
    assert (loaded.epochs, loaded.lr, loaded.l2) == (20, attributor.lr, attributor.l2)
    torch.testing.assert_close(loaded.score_turns(conversations[1]), attributor.score_turns(conversations[1]))


def test_load_rejects_a_different_model(
    model: Any, tokenizer: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model.config, "_name_or_path", "org/tiny-llama")
    attributor = _real_attributor(model, tokenizer)
    attributor.fit(_real_conversations(6))
    attributor.save(tmp_path)

    monkeypatch.setattr(model.config, "_name_or_path", "org/other-model")
    with pytest.raises(ValueError, match="trained on 'org/tiny-llama'"):
        ProbeAttributor.load(tmp_path, model, tokenizer)


def test_load_rejects_foreign_files(model: Any, tokenizer: Any, tmp_path: Path) -> None:
    (tmp_path / "probe.json").write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a mi4afa probe"):
        ProbeAttributor.load(tmp_path, model, tokenizer)
