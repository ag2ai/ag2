# A2UI v1.0 vendored spec

These JSON schemas are vendored from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v1_0/json/*.json` and
  `catalogs/basic/v1/catalog.json`
- **Pinned commit:** `078b3d5c4fe5541e4390baad1ec5995c6bba6857` (upstream `main`,
  2026-11-08). It includes `TextField.value` being required in the v1 basic
  catalog (a2ui-project/a2ui#3868).
- **Spec status:** v1.0 is the **current** upstream version
  (`specification/versions.json`, a2ui-project/a2ui#3999).

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/v1/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The file
**contents are byte-faithful** to upstream — `$id`
(`https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json`),
`catalogId`, and all `$ref`s are unchanged. The canonical catalog id resolves
on a2ui.org and must match what renderers advertise in `supportedCatalogIds`.

Unlike v0.9, the schema files use the upstream *agent / renderer* names
(`agent_to_renderer.json`, `renderer_to_agent.json`, `agent_capabilities.json`,
`renderer_capabilities.json`, `renderer_data_model.json`) rather than
*server / client*.

## Wire version

The protocol is named `v1.0` (`A2UIVersion`, the `v1_0` spec directory, the A2A
extension URI), but its messages and version-keyed capability objects carry the
bare `"1.0"` (a2ui-project/a2ui#3726): `{"version": "1.0", ...}` and
`a2uiRendererCapabilities: {"1.0": {...}}`. v0.9.x keeps the `v` on the wire.

## What v1.0 changes over v0.9

See upstream `specification/v1_0/docs/evolution_guide.md`. The ones that shape
this integration:

- **Agent → renderer** messages are `createSurface`, `updateComponents`,
  `updateDataModel`, `deleteSurface`, `callRendererFunction` and
  `agentFunctionResponse`. `createSurface` may carry the initial `components`
  and `dataModel`; `theme` is gone and `catalogId` is optional.
- **Renderer → agent** messages are `action`, `callAgentFunction`,
  `rendererFunctionResponse` and `error`. The v0.9.x-draft `wantResponse` /
  `actionId` / `actionResponse` mechanism does not exist: request/response goes
  through `callAgentFunction` → `agentFunctionResponse`, correlated by
  `functionCallId`.
- Data bindings and function calls use the reserved `@path` / `@call` keys.
- `updateDataModel.value` is required; `null` deletes the key.
- Catalog guidance lives in the catalog's `instructions` field; there is no
  `basic_catalog_rules.txt`.
- A catalog may `$ref` only the components, functions and leaf `$defs` it
  declares (`#/components/...`, `#/functions/...`, `#/$defs/...`, never
  `anyComponent` / `anyFunction`) and the standard types listed under
  "Restricted `$ref` Targets" in `a2ui_protocol.md`, written as
  `common_types.json#/$defs/<Type>`. Any other `$ref`, such as one into another
  catalog (a2ui-project/a2ui#3575), makes it invalid. A custom v1.0 catalog that
  does so is rejected when it is loaded, by the same rules as upstream
  `specification/v1_0/test/run_tests.py`.
