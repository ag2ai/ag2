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

v0.9.1 is served from these files too. At the pinned commit, upstream's v0.9.1 files
(`specification/v0_9_1/`, last changed in `ae97628de674407fd55eb76966d0b8eddb3a8497`) differ from these in
two places only:

- the messages' `version` field accepts the enum `["v0.9", "v0.9.1"]` instead of `const: "v0.9"`.
  `ag2/a2ui/schema_manager.py` (`_accept_wire_versions`) applies that widening at load time;
- upstream's v0.9.1 basic catalog lacks `List.listStyle` (a2ui-project/a2ui#3956 changed only the v0.9 copy).
  ag2 accepts and describes `listStyle` for v0.9.1 too. Both copies carry the same catalog `$id` (`v0_9`), so
  the newer content under that id wins, as in the official SDK (`a2ui-core`), which also serves v0.9.1 from its
  v0.9 models.

v1.0 is a breaking revision with its own wire version (`"1.0"`), schemas and
catalog; see `../../v1_0/spec/README.md`.
