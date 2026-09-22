# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Train, select, evaluate, persist and apply a failure-attribution probe."""

import json
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag2.eval.scorers import Attribution

from .activations import ActivationExtractor, ConversationActivations
from .probe import LogisticProbe
from .prompt import PromptTemplate
from .types import ActivationSite, Conversation, FitReport, SiteScore

__all__ = (
    "Example",
    "ProbeAttributor",
)

Example: TypeAlias = Conversation | ConversationActivations
"""A conversation, or activations already extracted from one (to skip re-running the model)."""

_WEIGHTS_FILE = "probe.safetensors"
_CONFIG_FILE = "probe.json"
_FORMAT = "ag2.mi4afa.probe"
_FORMAT_VERSION = 1


class ProbeAttributor:
    """Attribute a failed multi-agent conversation to its decisive step by probing a model's activations.

    The attributor reads one activation per turn from an open-weight model
    (see :class:`ActivationExtractor`), and a :class:`LogisticProbe` trained
    on labelled failures scores each turn. The highest-scoring turn is the
    predicted decisive step, and its speaker the responsible agent. The
    probed model's own output is never used.

    Typical use::

        attributor = ProbeAttributor.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
        report = attributor.fit(train)  # picks the site on a validation split of `train`
        score = attributor.evaluate(test)
        attribution = attributor.attribute(conversation)

    Args:
        extractor: Reads activations from the probed model.
        device: Where probes are trained and applied; defaults to the model's device.
        epochs: Probe optimization steps.
        lr: Probe learning rate.
        l2: Probe squared-norm penalty.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        *,
        device: torch.device | str | None = None,
        epochs: int = 300,
        lr: float = 0.05,
        l2: float = 1e-3,
    ) -> None:
        self.extractor = extractor
        self._device = torch.device(device) if device is not None else None
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self._site: ActivationSite | None = None
        self._probe: LogisticProbe | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | os.PathLike[str],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        sites: Sequence[ActivationSite] | None = None,
        template: PromptTemplate | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        **probe_options: Any,
    ) -> "ProbeAttributor":
        """Load a Hugging Face causal LM and its tokenizer and wrap them.

        Args:
            model_id: Hub id or local path of the model.
            device: Device to load the model onto.
            dtype: Model weight dtype.
            sites: Sites to read (defaults to all).
            template: Judge-prompt wording.
            chat_template_kwargs: Extra chat-template variables.
            **probe_options: Forwarded to :class:`ProbeAttributor` (``epochs``, ``lr``, ``l2``).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model = model.to(device).eval()
        extractor = ActivationExtractor(
            model, tokenizer, sites=sites, template=template, chat_template_kwargs=chat_template_kwargs
        )
        return cls(extractor, **probe_options)

    @property
    def device(self) -> torch.device:
        """Where probes are trained and applied."""
        if self._device is not None:
            return self._device
        return next(self.extractor.model.parameters()).device  # type: ignore[no-any-return]

    @property
    def site(self) -> ActivationSite:
        """The site the fitted probe reads."""
        if self._site is None:
            raise RuntimeError("the attributor is not fitted; call fit() or load() first")
        return self._site

    @property
    def probe(self) -> LogisticProbe:
        """The fitted probe."""
        if self._probe is None:
            raise RuntimeError("the attributor is not fitted; call fit() or load() first")
        return self._probe

    @property
    def is_fitted(self) -> bool:
        """Whether a probe has been fitted or loaded."""
        return self._probe is not None

    def extract(self, conversations: Sequence[Conversation]) -> list[ConversationActivations]:
        """Extract activations at every configured site, for reuse across :meth:`fit`, :meth:`evaluate` and :meth:`sweep_sites`."""
        return self.extractor.extract_many(conversations)

    def fit(
        self,
        train: Sequence[Example],
        *,
        validation: Sequence[Example] | None = None,
        validation_fraction: float = 0.2,
        seed: int = 42,
    ) -> FitReport:
        """Select the best site on held-out data, then train the probe there.

        Without ``validation``, a ``validation_fraction`` of ``train``'s
        conversations is held out (conversations are never split, so no
        conversation's turns land on both sides); one probe per site is trained
        on the rest and the site with the best validation step accuracy wins
        (ties broken by agent accuracy, then by earlier site). The final probe
        is refit on all of ``train``. With an explicit ``validation`` set, sites
        are scored on it and the final probe is fit on ``train`` alone.

        Test data should never be passed here; score it with :meth:`evaluate`.

        Args:
            train: Labelled conversations (``mistake_step`` set).
            validation: Optional labelled conversations to select the site on.
            validation_fraction: Share of ``train`` held out when ``validation`` is omitted.
            seed: Seed of the held-out split.

        Returns:
            The selected site and every site's validation score.
        """
        examples = self._examples(train)
        if validation is None:
            fit_examples, held_out = _split(examples, validation_fraction, seed)
        else:
            fit_examples, held_out = examples, self._examples(validation)
        sites = _shared_sites([*fit_examples, *held_out])

        scores = tuple(_score(self._train(fit_examples, site), held_out, site, self.device) for site in sites)
        best_index, _ = max(enumerate(scores), key=_selection_rank)
        best = sites[best_index]

        self._probe = self._train(examples, best)
        self._site = best
        return FitReport(
            site=best,
            validation=scores,
            train_count=len(examples),
            validation_count=len(held_out),
        )

    def evaluate(self, test: Sequence[Example]) -> SiteScore:
        """Score the fitted probe on labelled ``test`` conversations."""
        examples = self._examples(test, sites=(self.site,))
        return _score(self.probe, examples, self.site, self.device)

    def sweep_sites(self, train: Sequence[Example], test: Sequence[Example]) -> tuple[SiteScore, ...]:
        """Train one probe per site on ``train`` and score each on ``test``.

        An analysis tool: it measures where the information lives, so picking
        the best of these scores reports test-selected (optimistic) accuracy.
        Use :meth:`fit` + :meth:`evaluate` for an unbiased estimate. Leaves the
        attributor's fitted state untouched.
        """
        train_examples = self._examples(train)
        test_examples = self._examples(test)
        sites = _shared_sites([*train_examples, *test_examples])
        return tuple(_score(self._train(train_examples, site), test_examples, site, self.device) for site in sites)

    def score_turns(self, conversation: Example) -> torch.Tensor:
        """Return the probe's logit for every turn of ``conversation`` (on CPU)."""
        [example] = self._examples([conversation], sites=(self.site,), labelled=False)
        return self.probe.score(_site_values(example, self.site).to(self.device)).cpu()

    def attribute(self, conversation: Example) -> Attribution:
        """Name the decisive step and responsible agent of a failed conversation.

        Returns:
            An :class:`~ag2.eval.scorers.Attribution` whose ``decisive_step`` is
            an index into ``conversation.history``. The probe localizes but does
            not classify failures, so ``error_mode`` is ``"other"``.
        """
        scores = self.score_turns(conversation)
        step = int(scores.argmax())
        history = _conversation(conversation).history
        agent = history[step].name
        return Attribution(
            failed=True,
            error_mode="other",
            decisive_step=step,
            responsible_agent=agent,
            reasoning=f"probe at {self.site} scores turn {step} ({agent}) highest, logit {float(scores[step]):.3f}",
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the fitted probe and its settings to directory ``path``.

        The probe tensors go to ``probe.safetensors`` and the settings needed
        to reproduce its inputs (model id, site, prompt, chat-template
        variables, probe hyperparameters) to ``probe.json``.
        """
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        save_file(self.probe.state_dict(), directory / _WEIGHTS_FILE)
        config = {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "model_id": self.extractor.model_id,
            "site": {"layer": self.site.layer, "component": self.site.component},
            "template": asdict(self.extractor.template),
            "chat_template_kwargs": self.extractor.chat_template_kwargs,
            "probe": {"epochs": self.epochs, "lr": self.lr, "l2": self.l2},
        }
        (directory / _CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        model: Any,
        tokenizer: Any,
        *,
        device: torch.device | str | None = None,
    ) -> "ProbeAttributor":
        """Load a probe written by :meth:`save` for ``model``.

        The prompt and chat-template variables are restored from the saved
        settings so the probe sees the same inputs it was trained on.

        Raises:
            ValueError: If ``path`` is not a saved probe, or ``model`` is not the
                model the probe was trained on.
        """
        directory = Path(path)
        config = json.loads((directory / _CONFIG_FILE).read_text(encoding="utf-8"))
        if config.get("format") != _FORMAT or config.get("version") != _FORMAT_VERSION:
            raise ValueError(f"{directory} is not a mi4afa probe (format version {_FORMAT_VERSION})")

        extractor = ActivationExtractor(
            model,
            tokenizer,
            template=PromptTemplate(**config["template"]),
            chat_template_kwargs=config["chat_template_kwargs"],
        )
        saved_model = config.get("model_id")
        if saved_model and extractor.model_id and saved_model != extractor.model_id:
            raise ValueError(f"probe was trained on {saved_model!r}, not {extractor.model_id!r}")

        attributor = cls(extractor, device=device, **config["probe"])
        site = ActivationSite(layer=int(config["site"]["layer"]), component=config["site"]["component"])
        extractor.resolve_sites((site,))
        attributor._site = site
        attributor._probe = LogisticProbe.from_state_dict(load_file(directory / _WEIGHTS_FILE)).to(attributor.device)
        return attributor

    def _train(self, examples: Sequence[ConversationActivations], site: ActivationSite) -> LogisticProbe:
        features, labels = _site_matrix(examples, site)
        return LogisticProbe.fit(
            features.to(self.device), labels.to(self.device), epochs=self.epochs, lr=self.lr, l2=self.l2
        )

    def _examples(
        self,
        items: Sequence[Example],
        *,
        sites: Sequence[ActivationSite] | None = None,
        labelled: bool = True,
    ) -> list[ConversationActivations]:
        examples = [
            item if isinstance(item, ConversationActivations) else self.extractor.extract(item, sites=sites)
            for item in items
        ]
        if not examples:
            raise ValueError("no conversations given")
        if labelled and any(example.conversation.mistake_step is None for example in examples):
            raise ValueError("every conversation needs a mistake_step label")
        return examples


def _conversation(item: Example) -> Conversation:
    return item.conversation if isinstance(item, ConversationActivations) else item


def _split(
    examples: Sequence[ConversationActivations], fraction: float, seed: int
) -> tuple[list[ConversationActivations], list[ConversationActivations]]:
    if not 0 < fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {fraction}")
    if len(examples) < 2:
        raise ValueError("need at least two conversations to hold out a validation split")
    held_count = min(len(examples) - 1, max(1, round(len(examples) * fraction)))
    held = set(random.Random(seed).sample(range(len(examples)), held_count))
    fit = [example for index, example in enumerate(examples) if index not in held]
    held_out = [example for index, example in enumerate(examples) if index in held]
    return fit, held_out


def _shared_sites(examples: Sequence[ConversationActivations]) -> tuple[ActivationSite, ...]:
    sites = examples[0].sites
    if any(example.sites != sites for example in examples):
        raise ValueError("all conversations must be extracted at the same sites")
    return sites


def _site_values(example: ConversationActivations, site: ActivationSite) -> torch.Tensor:
    try:
        return example.values[example.sites.index(site)]
    except ValueError:
        raise ValueError(f"activations were not extracted at {site}") from None


def _site_matrix(
    examples: Sequence[ConversationActivations], site: ActivationSite
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.cat([_site_values(example, site) for example in examples]).float()
    labels = torch.cat([_one_hot(example) for example in examples])
    return features, labels


def _one_hot(example: ConversationActivations) -> torch.Tensor:
    labels = torch.zeros(len(example.conversation.history))
    labels[example.conversation.mistake_step] = 1.0
    return labels


def _score(
    probe: LogisticProbe, examples: Sequence[ConversationActivations], site: ActivationSite, device: torch.device
) -> SiteScore:
    step_hits = 0
    agent_hits = 0
    for example in examples:
        conversation = example.conversation
        predicted = int(probe.score(_site_values(example, site).to(device)).argmax())
        step_hits += predicted == conversation.mistake_step
        agent_hits += _same_agent(conversation.history[predicted].name, conversation.mistake_agent)
    return SiteScore(
        site=site,
        agent_accuracy=agent_hits / len(examples),
        step_accuracy=step_hits / len(examples),
        count=len(examples),
    )


def _same_agent(predicted: str, labelled: str | None) -> bool:
    """Agent names in Who&When are sometimes suffixed (``Expert`` vs ``Expert_1``), so a prefix match counts."""
    if labelled is None:
        return False
    return labelled.startswith(predicted) or predicted.startswith(labelled)


def _selection_rank(item: tuple[int, SiteScore]) -> tuple[float, float, int]:
    index, score = item
    return score.step_accuracy, score.agent_accuracy, -index
