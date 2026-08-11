---
status: accepted
date: 2026-07-29
---

# The Mistral client normalises a response shape no other provider uses

Surfaced while adding `MistralConfig`. Mistral's chat-completions endpoint runs
one tool — `image_generation` — on its own side and returns the entire exchange
in a response shape the rest of `ag2/config/*` never encounters. Normalising it,
plus a handful of SDK shapes that fail silently if taken at face value, forced
five decisions recorded here.

## Context

### SDK target

The client targets `mistralai>=2.8.0,<3`. Version 2 restructured the package into
namespace subpackages, so the entry point is:

```python
from mistralai.client import Mistral   # v2
from mistralai import Mistral          # v1 — what every published example shows
```

Mistral's own docs and examples still show the v1 top-level import. The v2 path is
correct here and should not be "fixed".

Mistral rejects most of its server-side tools on chat completions but not all of
them. Verified against the live API on `mistral-small-latest` and
`mistral-medium-latest`:

| Tool | chat/completions |
| :--- | :--- |
| `image_generation` | works |
| `web_search`, `web_search_premium`, `code_interpreter` | HTTP 400 `invalid_tools`, code 1800 |

When `image_generation` runs, the usual `choice.message` is **`None`** and the
whole exchange arrives in `choice.messages`:

1. `assistant` — tool call `generate_image(prompt=…)`
2. `tool` — result `{"url": "https://…blob.core.windows.net/…image.jpg?…"}`
3. `assistant` — `content=[TextChunk(…), ImageURLChunk(…)]`

A client reading only `choice.message` returns an empty reply and no error. The
streaming form carries the same three stages as separate deltas, with the result
identified by `delta.tool_call_id`.

Two further properties matter. The tool is named `generate_image` on the wire,
whereas AG2's tool is `image_generation`. And the image is a **signed URL** with
roughly an hour of validity, where OpenAI and Gemini both return inline bytes.

## Decision

### 1. A tool call is server-executed iff its id has a result in the same response

Both code paths collect tool calls and tool-result messages, then partition:

- **id has a matching result** → Mistral already ran it. Emit
  `BuiltinToolCallEvent` + `BuiltinToolResultEvent`; do **not** return it on
  `ModelResponse.tool_calls`.
- **no result** → ours to execute. Return it for the agent to dispatch.

The rule is content-based rather than a name allowlist, so it works unchanged for
streaming and non-streaming, and for a turn mixing a server-side tool with the
caller's own function tools.

### 2. Server tool names are mapped onto AG2's

`generate_image` is rewritten to `image_generation` before the events are built,
via a small alias table applied **only on the server-executed branch**.

`ToolExecutor` registers its not-found guard with
`context.stream.where(ToolCallEvent)`, and `BuiltinToolCallEvent` subclasses
`ToolCallEvent`. A provider name that matches no registered AG2 tool therefore
produces a spurious `ToolNotFoundEvent` on every generated image. xAI never hits
this because its provider names already match AG2's.

### 3. Structured output is built from SDK model classes, not dicts

`response_proto_to_format` constructs `ResponseFormat(json_schema=JSONSchema(...))`
rather than the plain dict the Z.AI and DashScope clients use.

`JSONSchema` spells the schema body `schema_definition` and aliases it to `schema`
on the wire. Passing `{"schema": {...}}` as a TypedDict is accepted and the schema
is **silently dropped** — structured output degrades to free text with no error.
The model class makes the binding explicit and fails loudly instead.

### 4. Reasoning traces are split out of the answer

Message content arrives either as a bare `str` or as a list of chunks. `ThinkChunk`
carries the model's reasoning and is emitted as `ModelReasoning`, so it never
appears in `reply.body`; `TextChunk` alone forms the answer.

Only some models produce a trace. `magistral-medium-latest` returns
`[ThinkChunk, TextChunk]`; `magistral-small-latest` returns a plain string and
rejects `prompt_mode="reasoning"` with "Reasoning prompt mode is not enabled for
this model". The list form is therefore handled defensively rather than gated on
a model allowlist.

### 5. Generated images are downloaded so they land on `reply.files`

`reply.files` is the documented cross-provider contract for generated images, and
`BinaryResult` holds `data: bytes`. Satisfying it from a URL requires a fetch, so
`MistralClient` issues one `GET` per generated image and attaches the bytes.

- The fetch is best-effort: any `httpx.HTTPError` yields no `files` entry and the
  turn continues, with the URL still on the `BuiltinToolResultEvent`.
- It reuses the configured `async_client` when one is supplied.
- The media type comes from the URL suffix when the `content-type` header is not
  an image type — blob storage serves these as `application/octet-stream`, which
  would otherwise produce `generated.octet-stream`.

## Consequences / things that look wrong but are deliberate

- **`ag2[mistral]` caps OpenTelemetry at 1.39.x.** Every `mistralai` 2.x release
  (checked 2.0.0 through 2.8.0) pins `opentelemetry-semantic-conventions<0.61`,
  which in turn pins `opentelemetry-api==1.39.1`. This matters more here than in
  most repos, since `ag2[tracing]` is a first-class extra.

  A clean install of `ag2[mistral,tracing]` resolves to a *consistent* 1.39.x
  stack and traces normally — the telemetry suite passes against it. The failure
  mode is adding `ag2[mistral]` to an environment that already has newer
  OpenTelemetry: `opentelemetry-api` is downgraded on its own, leaving it
  mismatched against a newer `opentelemetry-sdk`, and tracing dies with
  `AttributeError: type object 'TraceFlags' has no attribute 'RANDOM_TRACE_ID'`.

  The pin is over-tight — the SDK itself works fine against current
  OpenTelemetry — so holding the newer versions explicitly is a valid workaround.
  `mistralai` 1.x carries no such pin; if the ceiling ever becomes intolerable,
  dropping to v1 is the escape hatch, at the cost of the v2 import path and
  whatever v2-only surface is in use by then.

- **The client makes a network call the other providers do not.** Downloading
  inside a model client is unusual, but the alternative is `reply.files` being
  empty for Mistral alone, which silently breaks any code written against the
  documented contract. The URL expiring within the hour also makes fetching at
  response time the more useful moment.
- **`BuiltinToolResultEvent` carries the URL, not the bytes**, while `files`
  carries the bytes. The event records what Mistral actually returned; `files` is
  the normalised surface. The URL is also kept on the image's `metadata`.
- **The event name is not the name Mistral used.** Anyone correlating AG2 events
  against a raw API trace will see `image_generation` where the wire said
  `generate_image`. The alias is deliberately narrow — a caller's own function
  tool named `generate_image` goes down the client-side branch untouched.
- **`ImageGenerationTool`'s `size` / `quality` / `output_format` are accepted and
  ignored.** Mistral's tool takes no configuration. Rejecting the options would
  make the provider-neutral tool unusable here; dropping them keeps agent code
  portable.
- **A generated image costs far more tokens than a plain reply** (~1000 vs ~80 for
  the same prompt), because Mistral bills every internal turn of the exchange it
  runs. Its `usage` also carries a `request_count` extra, which AG2's `Usage` has
  no field for and drops.

## Related decisions in the same client

- **Only the `ToolResultsEvent` batch is mapped to messages.** History carries the
  batch *and* each constituent `ToolResultEvent`; mapping both sends every result
  twice, which Mistral rejects with `invalid_request_message_order`. OpenAI,
  Gemini and xAI key on the batch alone; Mistral now matches.
- **Uploads default to `purpose="ocr"`.** Every Mistral purpose validates the
  payload — `batch` and `fine-tune` require schema-conforming JSONL and reject
  anything else with HTTP 422. `ocr` is the only one that accepts arbitrary bytes,
  and the only one whose files can be referenced from a message by id.
- **Two SDK shapes that read as bugs.** `files.download_async` returns an
  *unconsumed* streaming `httpx.Response`, so `.content` raises before
  `aread()`; and fields the API omits arrive as a falsy `Unset()` sentinel rather
  than `None`, which a bare pass-through would leak to callers. Both are handled
  in `files.py` and pinned by tests.
- **Image `detail` is read from `metadata` for URL inputs and `vendor_metadata`
  for binary ones.** The asymmetry is inherited from the xAI client rather than
  invented here, so the two providers stay consistent for callers.
