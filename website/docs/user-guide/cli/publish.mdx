---
sidebarTitle: Publish
title: "ag2 publish"
description: "Validate and publish artifacts to the AG2 resource hub."
---

The `ag2 publish` command validates your artifact and submits it to the [AG2 Resource Hub](https://github.com/ag2ai/resource-hub) via a pull request.

## Prerequisites

- [GitHub CLI](https://cli.github.com/) (`gh`) installed and authenticated
- Git installed
- A valid `artifact.json` in your artifact directory

## Usage

```bash
ag2 publish <path_to_artifact>
```

## What It Does

1. **Validates** `artifact.json` — checks required fields, type-specific config, and structure
2. **Authenticates** with GitHub via `gh`
3. **Forks** `ag2ai/resource-hub` (idempotent)
4. **Clones** the fork to a temp directory
5. **Copies** your artifact to the correct directory (`tools/`, `templates/`, etc.)
6. **Commits and pushes** to a new branch
7. **Opens a pull request** with auto-generated description

## Validation

Before publishing, you can validate your artifact locally:

```bash
ag2 publish <path> --validate-only
```

**Validation checks include:**

- `artifact.json` is valid JSON with required fields (`name`, `type`, `version`, `description`)
- Type-specific config is present and valid
- Expected directories and files exist
- Version follows semver

## Scaffold an Artifact

Use [`ag2 create artifact`](create#ag2-create-artifact) to generate the required structure:

```bash
ag2 create artifact my-tool --type tool
# ... develop your tool ...
ag2 publish my-tool/
```
