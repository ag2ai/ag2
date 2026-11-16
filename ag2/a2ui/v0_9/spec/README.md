# A2UI v0.9 vendored spec

These JSON schemas are vendored **as-is** from the upstream A2UI project.

- **Source repository:** https://github.com/a2ui-project/a2ui
- **Source paths:** `specification/v0_9/json/*.json` and
  `specification/v0_9/catalogs/basic/catalog.json` (+ `rules.txt`)
- **Pinned commit:** `6e79b2e20fe2e1543aff3b9928613daa2b37189c` (upstream `main`, 2026-11-16). It includes
  `List.listStyle` in the basic catalog (a2ui-project/a2ui#3956)
- **Spec status:** v0.9 is a **previous** upstream version; v0.9.1 is the patch
  over it and v1.0 is current.

## Local adaptations

Only the catalog file is renamed on disk (`catalogs/basic/catalog.json` →
`basic_catalog.json`) to match this module's per-version layout. The nine JSON
schema files and `basic_catalog_rules.txt` (from `catalogs/basic/rules.txt`)
are **byte-faithful** to the pinned commit — `$id`, `catalogId`, and all
`$ref`s are unchanged.

## Relationship to v0.9.1 / v1.0

v0.9.1 is served from these files too. Upstream's v0.9.1 files
(`specification/v0_9_1/`, commit `078b3d5c4fe5541e4390baad1ec5995c6bba6857`)
differ from these only in description text and in the messages' `version`
field, which accepts the enum `["v0.9", "v0.9.1"]` instead of `const: "v0.9"`.
`ag2/a2ui/schema_manager.py` (`_accept_wire_versions`) applies that widening at
load time. The schema and catalog `$id`s are the same (`v0_9`) in both.

v1.0 is a breaking revision with its own wire version (`"1.0"`), schemas and
catalog; see `../../v1_0/spec/README.md`.
