# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch

from ag2.extensions.mi4afa import ActivationExtractor, ActivationSite, PromptEncoding

from .conftest import make_conversation


def _reference(model: Any, encoding: PromptEncoding) -> tuple[tuple[torch.Tensor, ...], dict[int, torch.Tensor]]:
    """Hidden states of the full model plus every MLP input, captured independently."""
    mlp_inputs: dict[int, torch.Tensor] = {}
    hooks = [
        layer.mlp.register_forward_hook(_record_input(mlp_inputs, index))
        for index, layer in enumerate(model.get_decoder().layers)
    ]
    try:
        with torch.no_grad():
            output = model(input_ids=torch.tensor([encoding.input_ids]), output_hidden_states=True)
    finally:
        for hook in hooks:
            hook.remove()
    return output.hidden_states, mlp_inputs


def _record_input(store: dict[int, torch.Tensor], index: int) -> Any:
    def hook(module: torch.nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> None:
        store[index] = args[0].detach()

    return hook


def _hook_count(model: Any) -> int:
    return sum(len(layer.mlp._forward_pre_hooks) for layer in model.get_decoder().layers)


def test_available_sites_cover_every_layer_and_the_final_state(model: Any, tokenizer: Any) -> None:
    extractor = ActivationExtractor(model, tokenizer)

    assert extractor.num_layers == 2
    assert [str(site) for site in extractor.available_sites()] == [
        "L0.resid_pre",
        "L0.mlp_in",
        "L1.resid_pre",
        "L1.mlp_in",
        "L2.resid_final",
    ]


def test_values_match_independently_captured_activations(model: Any, tokenizer: Any) -> None:
    conversation = make_conversation(4)
    extractor = ActivationExtractor(model, tokenizer)
    encoding = extractor.encode(conversation)

    activations = extractor.extract(conversation)

    hidden_states, mlp_inputs = _reference(model, encoding)
    positions = list(encoding.turn_positions)
    assert activations.values.shape == (5, 4, 16)
    assert activations.values.device.type == "cpu"
    for index, site in enumerate(activations.sites):
        if site.component == "mlp_in":
            expected = mlp_inputs[site.layer][0, positions]
        else:
            expected = hidden_states[site.layer][0, positions]
        torch.testing.assert_close(activations.values[index], expected, msg=f"mismatch at {site}")


def test_resid_final_is_the_normalized_last_hidden_state(model: Any, tokenizer: Any) -> None:
    conversation = make_conversation(3)
    extractor = ActivationExtractor(model, tokenizer, sites=[ActivationSite(2, "resid_final")])
    encoding = extractor.encode(conversation)

    [final] = extractor.extract(conversation).values

    with torch.no_grad():
        last = model.get_decoder()(input_ids=torch.tensor([encoding.input_ids])).last_hidden_state
    torch.testing.assert_close(final, last[0, list(encoding.turn_positions)])


def test_site_subset_and_override(model: Any, tokenizer: Any) -> None:
    only = ActivationSite(1, "mlp_in")
    extractor = ActivationExtractor(model, tokenizer, sites=[only])
    conversation = make_conversation(2)

    assert extractor.extract(conversation).sites == (only,)
    override = extractor.extract(conversation, sites=[ActivationSite(0, "resid_pre")])
    assert override.sites == (ActivationSite(0, "resid_pre"),)
    assert override.values.shape == (1, 2, 16)


def test_unknown_or_empty_sites_are_rejected(model: Any, tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="L5.mlp_in"):
        _ = ActivationExtractor(model, tokenizer, sites=[ActivationSite(5, "mlp_in")]).sites
    with pytest.raises(ValueError, match="at least one site"):
        ActivationExtractor(model, tokenizer).resolve_sites(())


def test_hooks_are_removed_after_extraction(model: Any, tokenizer: Any) -> None:
    ActivationExtractor(model, tokenizer).extract(make_conversation(2))
    assert _hook_count(model) == 0


def test_hooks_are_removed_when_the_forward_pass_fails(
    model: Any, tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = model.get_decoder()
    monkeypatch.setattr(decoder, "forward", _raise_runtime_error)

    with pytest.raises(RuntimeError, match="forward failed"):
        ActivationExtractor(model, tokenizer).extract(make_conversation(2))
    assert _hook_count(model) == 0


def _raise_runtime_error(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("forward failed")


def test_encoding_must_match_the_conversation(model: Any, tokenizer: Any) -> None:
    extractor = ActivationExtractor(model, tokenizer)
    encoding = extractor.encode(make_conversation(3))
    with pytest.raises(ValueError, match="3 turn positions"):
        extractor.extract_encoded(make_conversation(2), encoding)


def test_concurrent_extraction_matches_serial(model: Any, tokenizer: Any) -> None:
    extractor = ActivationExtractor(model, tokenizer)
    conversations = [make_conversation(turns, tag=str(turns)) for turns in (2, 3, 4, 5)]
    serial = [extractor.extract(conversation).values for conversation in conversations]

    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(extractor.extract, conversations))

    for expected, got in zip(serial, parallel, strict=True):
        torch.testing.assert_close(got.values, expected)


def test_model_id_comes_from_the_config(model: Any, tokenizer: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    extractor = ActivationExtractor(model, tokenizer)

    monkeypatch.setattr(model.config, "_name_or_path", "org/tiny-llama")
    assert extractor.model_id == "org/tiny-llama"
    monkeypatch.setattr(model.config, "_name_or_path", "")
    assert extractor.model_id is None
