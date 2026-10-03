# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Offline fixtures: a byte-level BPE tokenizer trained in memory and a tiny random Llama."""

from typing import Any

import pytest
import tokenizers
import torch
import transformers

from ag2.extensions.mi4afa import Conversation, Turn

CHATML_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
TRIMMING_TEMPLATE = (
    "<|bos|>{% for message in messages %}<|im_start|>{{ message['role'] }}\n\n{{ message['content'] | trim }}<|im_end|>"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n\n{% endif %}"
)
SPECIAL_TOKENS = ["<|bos|>", "<|im_start|>", "<|im_end|>"]

_CORPUS = [
    "You are an AI assistant tasked with analyzing a multi-agent conversation history.",
    "The problem is: how much did I save? The answer for the problem is: $55",
    "0 - Planner: collect the ticket prices. 1 - Coder: the season pass costs $120.",
    "Verification_Expert: I verified the costs. Respond ONLY with a number.",
    "Focus on errors that clearly derail the process.",
    "système: vérifié ✓ 日本語のテキスト",
]


def build_tokenizer(chat_template: str = CHATML_TEMPLATE) -> Any:
    """Train a small byte-level BPE tokenizer and wrap it as a fast HF tokenizer."""
    model = tokenizers.Tokenizer(tokenizers.models.BPE())
    model.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    model.decoder = tokenizers.decoders.ByteLevel()
    model.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
    )
    model.train_from_iterator(_CORPUS, trainer=trainer)
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=model, bos_token="<|bos|>", eos_token="<|im_end|>"
    )
    tokenizer.chat_template = chat_template
    return tokenizer


def build_model(vocab_size: int, *, layers: int = 2, seed: int = 0) -> Any:
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    return transformers.LlamaForCausalLM(config).eval()


def make_conversation(turns: int = 4, *, mistake_step: int | None = 1, tag: str = "") -> Conversation:
    agents = ["Planner", "Coder", "Verification_Expert"]
    return Conversation(
        question=f"How much did I save {tag}?",
        ground_truth="$55",
        history=tuple(
            Turn(name=agents[index % len(agents)], content=f"step {index} {tag}: the pass costs ${index * 10}.")
            for index in range(turns)
        ),
        mistake_step=mistake_step,
        mistake_agent=None if mistake_step is None else agents[mistake_step % len(agents)],
    )


@pytest.fixture(scope="session")
def tokenizer() -> Any:
    return build_tokenizer()


@pytest.fixture(scope="session")
def model(tokenizer: Any) -> Any:
    return build_model(len(tokenizer))
