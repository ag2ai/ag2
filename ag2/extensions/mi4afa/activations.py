# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Cache one activation per turn at chosen sites of a Hugging Face causal LM."""

import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from .prompt import PromptEncoding, PromptTemplate, build_prompt
from .types import ActivationSite, Conversation

__all__ = (
    "ActivationExtractor",
    "ConversationActivations",
)

_FINGERPRINT_FIELDS = ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size")


@dataclass(frozen=True, slots=True)
class ConversationActivations:
    """Per-turn activations of one conversation.

    Attributes:
        conversation: The conversation the activations were read from.
        sites: The sites, in the order of ``values``' first axis.
        values: ``(len(sites), len(conversation.history), hidden_size)`` tensor
            on CPU, in the model's dtype.
    """

    conversation: Conversation
    sites: tuple[ActivationSite, ...]
    values: torch.Tensor


class ActivationExtractor:
    """Run a causal LM over judge prompts and read one activation per turn.

    Works with decoder-only Hugging Face models whose decoder (from
    ``model.get_decoder()``) exposes ``layers`` with an ``mlp`` submodule —
    Llama, Qwen2 and Gemma 3 among them. Only the decoder is run, so no
    vocabulary logits are computed.

    Calls are serialized with a lock: the capture hooks share per-instance
    state, so one extractor can safely be used from several threads.

    Args:
        model: The causal LM, already on its target device and in eval mode.
        tokenizer: The model's fast tokenizer.
        sites: Sites to read; defaults to :meth:`available_sites`.
        template: Judge-prompt wording.
        chat_template_kwargs: Extra chat-template variables (see :func:`build_prompt`).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        sites: Iterable[ActivationSite] | None = None,
        template: PromptTemplate | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate()
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._requested_sites = tuple(sites) if sites is not None else None
        self._lock = threading.Lock()
        self._positions: torch.Tensor | None = None
        self._captured: dict[int, torch.Tensor] = {}

    @property
    def model_id(self) -> str | None:
        """The model's Hugging Face name or path, when known."""
        return getattr(getattr(self.model, "config", None), "_name_or_path", None) or None

    @property
    def num_layers(self) -> int:
        """Number of decoder layers."""
        return len(self._layers())

    def available_sites(self) -> tuple[ActivationSite, ...]:
        """Every readable site, in forward order: per layer ``resid_pre`` then ``mlp_in``, then ``resid_final``."""
        sites: list[ActivationSite] = []
        for layer in range(self.num_layers):
            sites.append(ActivationSite(layer, "resid_pre"))
            sites.append(ActivationSite(layer, "mlp_in"))
        sites.append(ActivationSite(self.num_layers, "resid_final"))
        return tuple(sites)

    @property
    def sites(self) -> tuple[ActivationSite, ...]:
        """The sites this extractor reads."""
        if self._requested_sites is None:
            return self.available_sites()
        return self.resolve_sites(self._requested_sites)

    def encode(self, conversation: Conversation) -> PromptEncoding:
        """Tokenize ``conversation`` with this extractor's prompt settings."""
        return build_prompt(
            conversation, self.tokenizer, template=self.template, chat_template_kwargs=self.chat_template_kwargs
        )

    def extract(
        self, conversation: Conversation, *, sites: Sequence[ActivationSite] | None = None
    ) -> ConversationActivations:
        """Read activations at every turn of ``conversation``.

        Args:
            conversation: The conversation to read.
            sites: Sites to read instead of :attr:`sites`.
        """
        return self.extract_encoded(conversation, self.encode(conversation), sites=sites)

    def extract_many(
        self, conversations: Iterable[Conversation], *, sites: Sequence[ActivationSite] | None = None
    ) -> list[ConversationActivations]:
        """:meth:`extract` each conversation in turn."""
        return [self.extract(conversation, sites=sites) for conversation in conversations]

    def extract_encoded(
        self,
        conversation: Conversation,
        encoding: PromptEncoding,
        *,
        sites: Sequence[ActivationSite] | None = None,
    ) -> ConversationActivations:
        """Read activations for an already-tokenized prompt.

        Args:
            conversation: The conversation ``encoding`` was built from.
            encoding: Token ids and one position per turn.
            sites: Sites to read instead of :attr:`sites`.

        Returns:
            The per-turn activations.

        Raises:
            ValueError: If ``encoding`` does not have one position per turn, or the
                prompt is longer than the model's context (``max_position_embeddings``).
        """
        if len(encoding.turn_positions) != len(conversation.history):
            raise ValueError(
                f"encoding has {len(encoding.turn_positions)} turn positions "
                f"but the conversation has {len(conversation.history)} turns"
            )
        limit = _context_length(self.model)
        if limit is not None and len(encoding.input_ids) > limit:
            raise ValueError(
                f"the prompt has {len(encoding.input_ids)} tokens, more than the {limit} the model supports; "
                "shorten the conversation or use a model with a longer context"
            )
        sites = self.resolve_sites(sites) if sites is not None else self.sites
        with self._lock:
            values = self._run(encoding, sites)
        return ConversationActivations(conversation=conversation, sites=sites, values=values)

    def _run(self, encoding: PromptEncoding, sites: Sequence[ActivationSite]) -> torch.Tensor:
        decoder = self.model.get_decoder()
        layers = self._layers()
        device = next(self.model.parameters()).device
        input_ids = torch.tensor([encoding.input_ids], device=device)
        self._positions = torch.tensor(encoding.turn_positions, device=device)
        self._captured = {}
        hooks = [
            _mlp(layers[layer]).register_forward_pre_hook(partial(self._capture, layer=layer))
            for layer in sorted({site.layer for site in sites if site.component == "mlp_in"})
        ]
        # Hidden states hold every layer at every token, so request them only when a residual site reads them.
        needs_hidden_states = any(site.component != "mlp_in" for site in sites)
        try:
            with torch.inference_mode():
                output = decoder(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    output_hidden_states=needs_hidden_states,
                    use_cache=False,
                )
                hidden_states = output.hidden_states
                rows = [
                    self._captured[site.layer]
                    if site.component == "mlp_in"
                    else hidden_states[site.layer][0, self._positions].cpu()
                    for site in sites
                ]
        finally:
            for hook in hooks:
                hook.remove()
            self._positions = None
            self._captured = {}
        return torch.stack(rows)

    def _capture(self, module: torch.nn.Module, args: tuple[Any, ...], *, layer: int) -> None:
        self._captured[layer] = args[0][0, self._positions].cpu()

    def resolve_sites(self, sites: Sequence[ActivationSite]) -> tuple[ActivationSite, ...]:
        """Return ``sites`` as a tuple after checking the model has every one.

        Raises:
            ValueError: If ``sites`` is empty or names a site the model lacks.
        """
        available = self.available_sites()
        unknown = [str(site) for site in sites if site not in available]
        if unknown:
            raise ValueError(f"sites not present in this model: {', '.join(unknown)}")
        if not sites:
            raise ValueError("at least one site is required")
        return tuple(sites)

    def _layers(self) -> Sequence[torch.nn.Module]:
        layers = getattr(self.model.get_decoder(), "layers", None)
        if layers is None:
            raise TypeError(f"{type(self.model).__name__} has no decoder layers to probe")
        return layers  # type: ignore[no-any-return]


def _text_config(model: Any) -> Any:
    """The model's config, or its text part when the config nests one."""
    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if callable(get_text_config) else config


def _context_length(model: Any) -> int | None:
    """The longest sequence the model's config declares (``max_position_embeddings``), or None if it declares none."""
    length = getattr(_text_config(model), "max_position_embeddings", None)
    return length if isinstance(length, int) and length > 0 else None


def _model_fingerprint(model: Any) -> dict[str, Any]:
    """The architecture fields of the model's config that the activations a probe reads depend on.

    The same weights loaded from the Hub and from a local copy give the same fingerprint. Two models that share
    an architecture but not their weights do too: the fingerprint does not tell them apart.
    """
    config = _text_config(model)
    return {field: getattr(config, field, None) for field in _FINGERPRINT_FIELDS}


def _mlp(layer: torch.nn.Module) -> torch.nn.Module:
    mlp = getattr(layer, "mlp", None)
    if not isinstance(mlp, torch.nn.Module):
        raise TypeError(f"decoder layer {type(layer).__name__} has no mlp submodule to hook")
    return mlp
