"""ag2 publish — validate and publish artifacts to the AG2 registry."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import typer

from ..ui import console

app = typer.Typer(
    help="Publish artifacts to the AG2 registry.",
    rich_markup_mode="rich",
)

ARTIFACTS_REPO = "ag2ai/resource-hub"

# Required fields in every artifact.json
REQUIRED_FIELDS = ["name", "type", "description", "version", "authors"]

# Required structure per type
TYPE_REQUIREMENTS: dict[str, list[str]] = {
    "template": ["template"],
    "tool": ["tool"],
    "dataset": ["dataset"],
    "agent": ["agent"],
    "skills": [],
    "bundle": ["bundle"],
}

# Expected directories per type
TYPE_DIRS: dict[str, list[str]] = {
    "template": ["scaffold"],
    "tool": ["src"],
    "dataset": ["data"],
    "agent": [],
    "skills": [],
    "bundle": [],
}

# Type name → plural directory in resource hub repo
TYPE_TO_DIR = {
    "template": "templates",
    "tool": "tools",
    "dataset": "datasets",
    "agent": "agents",
    "skills": "skills",
    "bundle": "bundles",
}


class ValidationError:
    """A single validation issue."""

    def __init__(self, level: str, message: str):
        self.level = level  # "error" or "warning"
        self.message = message


def _validate_artifact(artifact_dir: Path) -> tuple[dict | None, list[ValidationError]]:
    """Validate an artifact directory. Returns (parsed manifest, errors)."""
    issues: list[ValidationError] = []

    # Check artifact.json exists
    manifest_path = artifact_dir / "artifact.json"
    if not manifest_path.exists():
        issues.append(ValidationError("error", "Missing artifact.json"))
        return None, issues

    # Parse JSON
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as e:
        issues.append(ValidationError("error", f"Invalid JSON in artifact.json: {e}"))
        return None, issues

    # Required fields
    for field in REQUIRED_FIELDS:
        if not manifest.get(field):
            issues.append(ValidationError("error", f"Missing or empty required field: {field}"))

    # Type validation
    artifact_type = manifest.get("type", "")
    if artifact_type not in TYPE_REQUIREMENTS:
        issues.append(ValidationError("error", f"Unknown artifact type: {artifact_type}"))
        return manifest, issues

    # Type-specific config block
    for required_key in TYPE_REQUIREMENTS[artifact_type]:
        if required_key not in manifest:
            issues.append(ValidationError("error", f"Missing '{required_key}' config for type '{artifact_type}'"))

    # Required directories
    for dirname in TYPE_DIRS.get(artifact_type, []):
        dir_path = artifact_dir / dirname
        if not dir_path.is_dir():
            issues.append(ValidationError("warning", f"Expected directory not found: {dirname}/"))

    # Skills directory (recommended for all non-bundle types)
    if artifact_type != "bundle":
        skills_config = manifest.get("skills", {})
        skills_dir_name = skills_config.get("dir", "skills/").rstrip("/")
        skills_path = artifact_dir / skills_dir_name
        if artifact_type == "skills":
            # Skills type: check for rules/ or skills/ at the root
            has_content = (artifact_dir / "rules").is_dir() or (artifact_dir / "skills").is_dir()
            if not has_content:
                issues.append(ValidationError("warning", "No rules/ or skills/ directories found"))
        elif not skills_path.is_dir():
            issues.append(ValidationError("warning", f"No skills directory at {skills_dir_name}/"))
        else:
            # Check skills have content
            skill_files = list(skills_path.rglob("SKILL.md"))
            flat_files = [f for f in skills_path.rglob("*.md") if f.name != "SKILL.md"]
            if not skill_files and not flat_files:
                issues.append(ValidationError("warning", "Skills directory exists but contains no SKILL.md files"))

    # Version format
    version = manifest.get("version", "")
    if version:
        parts = version.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            issues.append(ValidationError("warning", f"Version '{version}' is not valid semver (expected X.Y.Z)"))

    # Authors
    authors = manifest.get("authors", [])
    if not authors:
        issues.append(ValidationError("warning", "No authors specified"))

    # Tags
    tags = manifest.get("tags", [])
    if not tags:
        issues.append(ValidationError("warning", "No tags specified — helps discoverability"))

    return manifest, issues


def _run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=120,
    )


@app.command("artifact")
def publish_artifact(
    path: Path = typer.Argument(".", help="Path to artifact directory."),
    repo: str = typer.Option(ARTIFACTS_REPO, "--repo", "-r", help="Target resource hub repository."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate only, don't publish."),
    branch: str = typer.Option("", "--branch", "-b", help="Branch name (default: add-<type>-<name>)."),
) -> None:
    """Validate and publish an artifact to the AG2 registry via pull request.

    [dim]Validates the artifact structure, then forks the resource hub repo,
    copies the artifact, and opens a PR.[/dim]

    [dim]Examples:[/dim]
      [command]ag2 publish artifact ./my-template[/command]
      [command]ag2 publish artifact ./my-tool --dry-run[/command]
      [command]ag2 publish artifact . --repo myorg/artifacts[/command]
    """
    artifact_dir = path.resolve()
    if not artifact_dir.is_dir():
        console.print(f"[error]Not a directory: {artifact_dir}[/error]")
        raise typer.Exit(1)

    # --- Step 1: Validate ---
    console.print(f"\n[heading]Validating artifact:[/heading] {artifact_dir.name}\n")

    manifest, issues = _validate_artifact(artifact_dir)

    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    for issue in issues:
        if issue.level == "error":
            console.print(f"  [error]✗[/error] {issue.message}")
        else:
            console.print(f"  [warning]![/warning] {issue.message}")

    if errors:
        console.print(f"\n  [error]Validation failed with {len(errors)} error(s).[/error]")
        raise typer.Exit(1)

    if not issues:
        console.print("  [success]✓[/success] All checks passed")

    if warnings:
        console.print(f"\n  [warning]{len(warnings)} warning(s)[/warning] — consider fixing before publishing.")

    if manifest is None:
        raise typer.Exit(1)

    artifact_name = manifest["name"]
    artifact_type = manifest["type"]
    target_dir_name = TYPE_TO_DIR.get(artifact_type, f"{artifact_type}s")
    target_path = f"{target_dir_name}/{artifact_name}"

    console.print(f"\n  Artifact: [command]{artifact_name}[/command] ({artifact_type})")
    console.print(f"  Target:   [path]{target_path}/[/path] in {repo}")
    console.print(f"  Version:  {manifest.get('version', 'unknown')}")

    if dry_run:
        console.print("\n  [dim]Dry run — no changes made.[/dim]\n")
        raise typer.Exit(0)

    # --- Step 2: Check gh CLI ---
    console.print()
    try:
        result = _run_gh("auth", "status", check=False)
        if result.returncode != 0:
            console.print("[error]Not authenticated with GitHub CLI.[/error]")
            console.print("Run [command]gh auth login[/command] first.")
            raise typer.Exit(1)
    except FileNotFoundError:
        console.print("[error]GitHub CLI (gh) not found.[/error]")
        console.print("Install from: https://cli.github.com")
        raise typer.Exit(1)

    # --- Step 3: Fork and clone ---
    console.print("[heading]Publishing...[/heading]\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Fork (idempotent — does nothing if already forked)
        console.print("  Forking repository...")
        _run_gh("repo", "fork", repo, "--clone=false", check=False)

        # Clone the fork
        console.print("  Cloning fork...")
        clone_result = _run_gh("repo", "clone", repo, str(tmp_path / "artifacts"), "--", "--depth=1", check=False)
        if clone_result.returncode != 0:
            console.print(f"[error]Failed to clone: {clone_result.stderr.strip()}[/error]")
            raise typer.Exit(1)

        repo_path = tmp_path / "artifacts"

        # --- Step 4: Create branch and copy ---
        branch_name = branch or f"add-{artifact_type}-{artifact_name}"
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Copy artifact to correct location
        dest = repo_path / target_dir_name / artifact_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(artifact_dir, dest)

        # Remove any __pycache__, .pyc
        for pattern in ["__pycache__", "*.pyc", ".pytest_cache"]:
            for p in dest.rglob(pattern):
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.is_file():
                    p.unlink()

        console.print(f"  Copied to [path]{target_path}/[/path]")

        # --- Step 5: Commit and push ---
        subprocess.run(["git", "add", "."], cwd=repo_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", f"Add {artifact_type}: {artifact_name}"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        console.print("  Pushing branch...")
        push_result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if push_result.returncode != 0:
            console.print(f"[error]Push failed: {push_result.stderr.strip()}[/error]")
            raise typer.Exit(1)

        # --- Step 6: Create PR ---
        console.print("  Creating pull request...")
        description = manifest.get("description", "")
        pr_body = (
            f"## New {artifact_type}: {artifact_name}\n\n"
            f"{description}\n\n"
            f"- **Type:** {artifact_type}\n"
            f"- **Version:** {manifest.get('version', '0.1.0')}\n"
            f"- **Authors:** {', '.join(manifest.get('authors', []))}\n"
            f"- **Tags:** {', '.join(manifest.get('tags', []))}\n"
        )

        pr_result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--repo",
                repo,
                "--title",
                f"Add {artifact_type}: {artifact_name}",
                "--body",
                pr_body,
                "--head",
                branch_name,
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )

        if pr_result.returncode != 0:
            console.print(f"[error]PR creation failed: {pr_result.stderr.strip()}[/error]")
            console.print(f"Branch [command]{branch_name}[/command] has been pushed — create the PR manually.")
            raise typer.Exit(1)

        pr_url = pr_result.stdout.strip()
        console.print(f"\n  [success]✓[/success] Pull request created: {pr_url}")
        console.print()
