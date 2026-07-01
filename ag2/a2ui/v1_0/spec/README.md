# A2UI v1.0 vendored spec

These JSON schemas are vendored from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v1_0/json/*.json` and
  `specification/v1_0/catalogs/basic/catalog.json`
- **Pinned commit:** `d8af0c56cd8e211b4f7d5c168efb56828d7aae5f`
- **Spec status:** v1.0 is an upstream **candidate** — re-pin and re-vendor when it stabilizes.

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The file
**contents are byte-faithful** to upstream — `$id`
(`https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json`),
`catalogId`, and all `$ref`s are unchanged. The canonical catalog id resolves
on a2ui.org and must match what renderers advertise in `supportedCatalogIds`.

## What v1.0 adds over v0.9 (server → client)

Two new message types beyond v0.9's four (`createSurface`, `updateComponents`,
`updateDataModel`, `deleteSurface`):

- `callFunction` — server-initiated client function call
  (required: `version`, `callFunction.call`, `functionCallId`).
- `actionResponse` — server response to a client-initiated action
  (required: `version`, `actionResponse`, `actionId`; `actionResponse` is
  `value` xor `error`).

Upstream v1.0 does not ship a `catalogs/basic/rules.txt` (v0.9 did); the prose
guidance lives in `docs/basic_catalog_implementation_guide.md` instead, so this
version has no `basic_catalog_rules.txt`.
