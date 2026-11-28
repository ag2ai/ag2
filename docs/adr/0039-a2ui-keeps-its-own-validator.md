---
status: accepted
date: 2026-11-15
---

# 0039. `ag2.a2ui` keeps its own validator and vendored specs

## Context

The official A2UI Python packages (`a2ui-core`, `a2ui-agent-sdk`, `a2ui-a2a`) validate the same
messages `ag2.a2ui` does. An earlier attempt (`ag2.a2ui.beta` on `a2ui-core` 0.3.0, ag2ai/ag2#3403)
delegated validation to them and was closed unreviewed. A prototype on 2026-11-15 ran one payload
corpus through ag2 and `a2ui-core` 0.6.0:

| Corpus | Agree | Total |
|---|---|---|
| v0.9 | 45 | 55 |
| v0.9.1 | 46 | 55 |
| v1.0 | 55 | 69 |
| Custom catalogs | 30 | 36 |

The differences come from what core is, not from bugs:

- Core is a stateful renderer-side processor. It stops at the first error, needs surfaces from
  earlier turns seeded before it accepts updates to them, and picks a version adapter per message
  instead of pinning one version.
- Core coerces where an agent framework must be strict: it accepts a string `weight: "1"` where the
  schema asks for a number, in both its default and its `STRICT_VALIDATION` mode (ag2 reports
  `'1' is not of type 'number'`), it accepts another version's binding key in v1.0, it leaves custom
  components open, and it does not reject messages a model must not send (`agentFunctionResponse`,
  renderer-only calls).
- Core's default mode skips the component graph checks and the component property checks: a dangling
  child reference and a CheckBox `value: "yes"` both pass, and fail only under `STRICT_VALIDATION`.
  ag2 has one mode, and it always runs both.
- Its type-error texts are Pydantic union noise a model cannot act on.
- It has no prompt generator and no parser for the LLM's `<a2ui-json>` text format (at 0.6.0; `a2ui-agent-sdk`
  has since added both, and ag2 follows their documented behaviour where it is spec, see below).
- `a2ui-a2a` pins `a2a-sdk<0.4`; ag2 requires `>=1.3`.
- Every package is 0.x and ships breaking changes every one to three weeks.

When this was decided (core 0.6.0, upstream before a2ui-project/a2ui#4044), ag2's vendored
agent→renderer schemas for v0.9.1 and v1.0 were identical to core's published ones, and for v0.9 only
descriptions differed. Later re-vendoring may run ahead of core's releases;
`test/a2ui/test_schema_drift.py` lists each known difference. The `beta` attempt needed a workaround layer (replaying operations
one at a time, synthesising `createSurface` messages, folding `updateComponents`) to hide core's
nature.

## Decision

`ag2.a2ui` keeps its own validator and vendored specs and takes no runtime dependency on `a2ui-core`,
`a2ui-agent-sdk` or `a2ui-a2a`. The checks core does better are re-implemented in ag2's validator:
the component graph (dangling references, orphans, cycles, duplicate ids), JSON Pointer paths in
data-model updates, and readable "unknown function" / "unknown catalog" errors.

`a2ui-core` is installed only in the `optionals` dependency group, where `test/a2ui/test_schema_drift.py`
compares the vendored schemas with the SDK's. It never becomes a runtime extra.

## Consequences

- ag2 follows upstream spec changes by updating its vendored files by hand. The drift test fails when the SDK's schemas
  differ from them in anything but descriptions and list order; each accepted difference is listed, with its reason, in
  the test's `KNOWN_DRIFT`.
- Validation stays strict, reports every error of a response in one retry, and accepts updates to
  surfaces created in earlier turns without seeding.
- Reopening SDK adoption, including the `ag2.a2ui.beta` subpackage, needs a new ADR that supersedes
  this one.

## Deviations from upstream

Upstream A2UI at `6e79b2e20fe2e1543aff3b9928613daa2b37189c` is the only authority. ag2 departs from it on purpose
in these places; line numbers are at that commit. Re-checked on 2026-11-20 against
`496df0e76e2ad9aba4cc2a12abe066f43d4f171d`, including its `conformance/` suites: the vendored schemas and catalogs are
unchanged, and the deviations below still hold. Suite paths below are at that later commit.

- **`root` in the same response.** `specification/v1_0/docs/a2ui_protocol.md` L180 says one of the components sent
  to a surface MUST have `"id": "root"`, and L787 (v0.9: `specification/v0_9/docs/a2ui_protocol.md` L331) lets the
  renderer buffer updates until it arrives. ag2 rejects a response that creates a surface and gives it components but
  no `root`: a stateless server cannot know whether a later response will send one, and a retry is cheaper for the
  model than a surface that never renders. Updates to a surface created earlier need no `root`.
- **Dangling references and orphans on a new surface.** v1.0 `a2ui_protocol.md` L223 asks renderers to handle
  references to components that "do not yet exist" with placeholders, and L787 has them render "skipping invalid
  references". For a surface the response creates, ag2 rejects a reference to a component the response does not send
  and a component not reachable from `root`, as `a2ui-core` does under `STRICT_VALIDATION`. Same reasoning as the
  `root` rule: a stateless server cannot wait for the rest of the tree, and a placeholder that never fills is a
  broken UI. Updates to a surface created earlier may still reference earlier components.
- **`callAgentFunction` without `catalogId`.** v1.0 `a2ui_protocol.md` L1495 says the renderer MUST set
  `callFunction.catalogId`. `renderer_to_agent.json` does not require it, and ag2 accepts the call without it: the
  MUST binds the renderer, and refusing the call would only fail the user's click.
- **`List.listStyle` for v0.9.1.** Upstream's `specification/v0_9_1/catalogs/basic/catalog.json` lacks
  `List.listStyle`, which a2ui-project/a2ui#3956 added only to `specification/v0_9/catalogs/basic/catalog.json`. Both
  files declare the same `$id`, so ag2 serves the newer v0.9 content for v0.9.1, as `a2ui-core` does
  (`ag2/a2ui/v0_9/spec/README.md`).
- **v1.0 surfaces from earlier responses.** v1.0 `a2ui_protocol.md` L185 / L503 resolve a component without
  `@catalogId` against its surface's `catalogId`. The server keeps no state between responses, so for a surface
  created in an earlier response it accepts an implicit component or call if any configured catalog defines it. With
  one catalog this is exact; with several, an explicit `@catalogId` / `catalogId` selects the definition.
- **`agentFunctionResponse` from the model.** v1.0 `agent_to_renderer.json` allows an `agentFunctionResponse`
  message from the agent. ag2 sends it itself, in reply to a `callAgentFunction` (L1486), so one the model writes
  would answer a call the model never saw; the validator asks for a retry instead.
- **Surfaces on a catalog the turn does not offer.** The schemas take any string as `createSurface.catalogId`. ag2
  asks for a retry when a surface names a catalog the turn does not offer (ADR 0041): the renderer has nothing to
  resolve its components against (L519).
- **Components and calls with no catalog to resolve against.** v1.0 L519 makes a component without `@catalogId` on
  a surface without `catalogId` a resolution error. The schemas accept it, and so do upstream cases
  `dynamic_value_validation.json#0-6,9,10`, which check message shape only; ag2 follows the prose and asks for a
  retry.
- **`callRendererFunction` and `allowedCallers`.** L302 says only `agentOnly` or `rendererOrAgent` functions can be
  called by the agent, and `openUrl` is `rendererOnly`. Upstream cases `call_function_message.json#0-2` call `openUrl`
  and are marked valid, because they check message shape only; `#3` calls `pingAgent`, which the basic catalog does
  not define. ag2 follows the prose and rejects all four.
- **Custom catalog `protocolVersion`.** v1.0 L635 makes a catalog without `protocolVersion` a `"0.9"` catalog and
  requires one for catalogs targeting 1.0, and ag2 rejects a catalog whose major.minor is not the served version's
  (ADR 0040). The reference SDK's `_settle_protocol_version` instead takes the agent's version for a catalog that
  states none, and upstream cases under `specification/v1_0/test/cases/` that check `catalog_definition.json`
  (`catalog_schema_relaxation`, `composition_constraints`, `function_definitions`) omit it, since that schema leaves
  it optional. ag2 follows the prose; the conformance probe states `"1.0"` for those cases, which then all agree.
- **Component discriminators.** v1.0 L607–608 (rule 4) gives every catalog component "a required property named
  `component` whose value is a constant (`const`) matching the key". The reference SDK takes the `const` as the
  type even when it differs from the key, and uses the key when there is none. ag2 rejects at load a component whose
  `component` is not a `const` equal to its key, looked up directly or in an `allOf` branch beside a trait (rule 5),
  since the prompt names components by key and the validator by const. Upstream cases
  `composition_constraints.json#0-1` leave `component` out of `required` and are valid; ag2 accepts them and, as the
  SDK does, adds `component` to `required` in the catalog it serves.
- **A2A DataParts with a single object.** The A2A extension (v1.0 `a2ui_extension_specification.md` L183, v0.9
  L178) says a DataPart's `data` "MUST be an array of messages". The reference SDK `a2ui_a2a` writes and reads one
  object per DataPart. ag2 follows the spec: `A2UIAgentExecutor` consumes a renderer DataPart whose `data` is a single
  envelope with a warning and acts on none of it, and `create_a2ui_parts(legacy_split=True)`, which writes that shape,
  is deprecated (removed in 1.4.0).
- **Upstream conformance data that breaks the prose.** `conformance/core/multi_catalog.yaml` puts `catalogId`
  without `@` on components (L179; the prose is `@catalogId`, v1.0 L503/L509), and its L155 catalog `$ref`s a
  `#/$defs/DynamicString` it does not declare ("Restricted `$ref` Targets", L583–586). `core/catalog.yaml`
  `test_v10_catalog_from_json_leaf_defs` loads a catalog with `metadata.displayName`, which the vendored
  `catalog_definition.json` forbids (`additionalProperties: false`). ag2 follows the prose and the schema.
- **No markup recovery or JSON repair.** `a2ui-agent-sdk` recovers a block without its open tag, tags in backticks
  and tags with attributes, and repairs trailing commas, smart quotes and raw newlines
  (`conformance/agent/direct_json/response_parser.yaml`, `compiler.yaml`). ag2 reads only the literal
  `<a2ui-json>` tags and strict JSON, and asks for a retry otherwise: a retry is cheap, and a guessed repair could
  send UI the model did not mean. It does read every block of a response and ignores a close tag inside a JSON
  string, as the SDK does.
- **SDK-only limits.** `blueprints/modules/a2ui_core.blueprint.md` L1004 caps nesting (50 levels, 5 nested function
  calls, 100 for expressions), and the SDK parses `formatString` templates and rejects `@index` outside a list
  template before sending. The prose sets no such limits and makes `@index` misuse a renderer `EXECUTION_ERROR`
  (v1.0 L319); ag2 checks none of them.
- **Agent function arguments.** v1.0 L1486 answers arguments that do not match the definition with
  `INVALID_FUNCTION_CALL`. ag2 checks the names, and then validates the values as it does a tool's: through the
  handler's Pydantic signature, which coerces `"25"` to an `int` parameter. A value that cannot be coerced is still
  `INVALID_FUNCTION_CALL`.
- **Catalog preference order.** `conformance/agent/request_processor.yaml` L181 orders the active catalogs by the
  renderer's `supportedCatalogIds`. In v1.0 ag2 keeps the basic catalog as the surface default whenever the renderer
  supports it (ADR 0040).
- **Legacy A2A MIME type.** `a2ui_a2a` still reads `application/json+a2ui`; ag2 reads only the
  `application/a2ui+json` the extension specification names.
