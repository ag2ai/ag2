---
status: accepted
date: 2026-11-10
---

# 0036. Usage prompt tokens count the whole input

## Context

Providers count prompt-cached input in different ways. OpenAI, Gemini, Mistral,
xAI and Z.ai report a prompt count that already includes the cached tokens.
Anthropic's `input_tokens` and Bedrock Converse's `inputTokens` count only the
uncached input, and report cache reads and writes beside it. Each mapper copied
its provider's number into `Usage.prompt_tokens`, so the field meant different
things per provider. On a cached Anthropic call it could be 3 while the model
read 8,630 tokens.

Only the AG-UI exporter corrected for this, with a per-provider table. Every
other consumer read the raw number. Telemetry under-reported Anthropic and
Bedrock input. `TokenMonitor` counted cached input on OpenAI but not on
Anthropic. `UsageReport.total` summed incompatible numbers across a
mixed-provider delegation tree. A price-based cost estimate would under-bill the
cached part.

Two consistent meanings were possible: `prompt_tokens` as the whole input, or
as the uncached input only. The uncached form matches Anthropic's raw field. It
would make every consumer add the cache fields back to get the input size, and
it would change the meaning of the field for the five providers that already
report the inclusive count.

## Decision

`Usage` fields have one meaning on every provider:

- `prompt_tokens` is the whole input the model read, including tokens read from
  and written to the prompt cache.
- `cache_read_input_tokens` and `cache_creation_input_tokens` are parts of
  `prompt_tokens`, never additions to it. The uncached input is `prompt_tokens`
  minus both. `cache_creation_extended_input_tokens` is a part of
  `cache_creation_input_tokens`.
- `total_tokens` is at least `prompt_tokens + completion_tokens`.

The provider mapper establishes this meaning, in `normalize_usage`. Consumers
(AG-UI, telemetry, `TokenMonitor`, `UsageReport`, cost and budget code) read
the fields as they are and keep no per-provider correction. A mapper for a
provider that reports the cache beside the input adds both cache counts to
`prompt_tokens` and `total_tokens`. A provider-reported total below input plus
output is replaced by that sum.

## Consequences

For Anthropic and Bedrock, `prompt_tokens`, `total_tokens`, telemetry
`gen_ai.usage.input_tokens` and the input-token metric grow by the cached
tokens. `TokenMonitor` thresholds are reached sooner on prompt-cached
workloads, and so are eval budgets: `TokenUsage.input` and `total`,
`token_budget` and `BudgetThresholds.max_tokens_per_task` read the same
`prompt_tokens`. An eval baseline recorded before this change is not comparable
on cached Anthropic or Bedrock workloads. Other providers are unchanged. Release
notes must call this out.

Usage stored by earlier versions is not rewritten. Anthropic and Bedrock
records from before this change keep the uncached count in `prompt_tokens`, and
a report that mixes old and new records mixes both meanings.

Cost and budget code can price `Usage` alone: the uncached input,
cache reads and cache writes are each derivable without knowing the provider.
A new provider mapper must follow this rule, and its tests should cover a call
with a cache read and a cache write.
