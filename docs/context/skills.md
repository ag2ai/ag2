# Skills

The local-skills subsystem (`autogen/beta/tools/skills/`) implements the
agentskills.io progressive-disclosure pattern: skills are directories discovered
on the filesystem, surfaced to the model as a catalog, and loaded / read /
executed on demand through tools.

## Language

**Skill**:
A directory containing a `SKILL.md` (with YAML frontmatter) plus optional bundled
files. The unit of progressive disclosure.

**Resource**:
A bundled file inside a skill directory that is **not** `SKILL.md` and **not**
under `scripts/`. Read on demand via `read_skill_resource` (e.g. `references/`,
`assets/`). Distinct from a Script.

**Script**:
An executable file under a skill's `scripts/` directory. Run on demand via
`run_skill_script`. Disjoint from a Resource — a file is one or the other, never
both.

**Runtime**:
The backend that owns a set of skills — it discovers them (`runtime.skills`) and
performs all IO on them (`read` / `read_resource` / `execute`). `LocalRuntime`
backs skills with the filesystem; a `MemoryRuntime` backs them with RAM. Skills,
Scripts, and Resources are inert descriptors; the Runtime does the reading and
running.
_Avoid_: store, source, provider (for this concept).

**Shadowing**:
When the same skill name exists in more than one Runtime, the **last** Runtime
passed to the Toolkit/Plugin wins (project overrides global). The rule is applied
uniformly to routing, reads, the catalog, and tool gating.
