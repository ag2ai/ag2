# ag2 publish

> Validate and publish artifacts to the AG2 registry via pull request.

## Problem

You've created an artifact (template, tool, dataset, agent, skills, or bundle)
and want to share it with the community. The AG2 artifacts registry
(`ag2ai/resource-hub`) accepts contributions via pull request, but manually
forking, cloning, copying files, and opening PRs is tedious.

`ag2 publish artifact` automates the entire flow: validate your artifact,
fork the repo, create a branch, copy files, and open a PR — all in one command.

## Commands

### `ag2 publish artifact` — Validate and publish

```bash
# Publish the artifact in the current directory
ag2 publish artifact

# Publish from a specific path
ag2 publish artifact ./my-template

# Validate only (no PR created)
ag2 publish artifact ./my-tool --dry-run

# Target a different repository
ag2 publish artifact . --repo myorg/artifacts

# Specify a custom branch name (default: add-<type>-<name>)
ag2 publish artifact . --branch my-custom-branch
```

**Arguments:**
- `path` — path to artifact directory (default: current directory)

**Options:**
- `--dry-run` — validate the artifact without publishing
- `--repo` / `-r` — target resource hub repository (default: `ag2ai/resource-hub`)
- `--branch` / `-b` — branch name (default: auto-generated as `add-<type>-<name>`)

## Validation

Before publishing (or when using `--dry-run`), the command validates:

**Required checks (errors):**
- `artifact.json` exists and is valid JSON
- Required fields are present and non-empty: `name`, `type`, `description`,
  `version`, `authors`
- `type` is one of: `template`, `tool`, `dataset`, `agent`, `skills`, `bundle`
- Type-specific config block exists (e.g. `template` type needs a `"template"` key,
  `tool` type needs a `"tool"` key)

**Recommended checks (warnings):**
- Expected directories exist (`scaffold/` for templates, `src/` for tools,
  `data/` for datasets)
- Skills directory contains at least one `SKILL.md` file
- Version follows semver format (`X.Y.Z`)
- Authors list is not empty
- Tags are present (helps discoverability)
- For `skills` type: `rules/` or `skills/` directories exist

## Publishing Flow

When not using `--dry-run`, the command runs this sequence:

1. **Validate** — run all checks above; abort on errors
2. **Check GitHub auth** — verify `gh auth status`; prompt to login if needed
3. **Fork repository** — `gh repo fork` (idempotent, skips if already forked)
4. **Clone fork** — shallow clone to a temp directory
5. **Create branch** — `git checkout -b add-<type>-<name>` (or custom `--branch`)
6. **Copy artifact** — copy to `<type-plural>/<name>/` in the repo
   (e.g. `templates/my-template/`, `tools/web-scraper/`).
   Strips `__pycache__`, `.pyc`, `.pytest_cache` automatically.
7. **Commit and push** — commit with message `Add <type>: <name>`, push branch
8. **Create PR** — opens a pull request with auto-generated body including
   description, type, version, authors, and tags from `artifact.json`

## Prerequisites

- **GitHub CLI (`gh`)** — must be installed and authenticated (`gh auth login`)
- **Git** — for branch creation, commit, and push
- A valid `artifact.json` in the artifact directory (create one with
  `ag2 create artifact`)

## Example Workflow

```bash
# 1. Scaffold a new tool artifact
ag2 create artifact tool web-scraper

# 2. Implement the tool and write skills
cd web-scraper
# ... edit src/web_scraper.py, skills/, artifact.json ...

# 3. Validate before publishing
ag2 publish artifact . --dry-run

# 4. Publish to the registry
ag2 publish artifact .
# → Validates, forks ag2ai/resource-hub, creates PR
```

## Implementation Notes

### Directory Mapping

Artifact types map to plural directories in the registry:

| Type | Registry Directory |
|------|-------------------|
| template | `templates/` |
| tool | `tools/` |
| dataset | `datasets/` |
| agent | `agents/` |
| skills | `skills/` |
| bundle | `bundles/` |

### PR Body

The auto-generated PR body includes metadata extracted from `artifact.json`:

```
## New <type>: <name>

<description>

- **Type:** <type>
- **Version:** <version>
- **Authors:** <authors>
- **Tags:** <tags>
```

### Dependencies

- `gh` CLI — for fork, clone, and PR creation
- `git` — for branch, commit, push
- No additional Python packages required
