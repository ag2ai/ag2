# Skills

The local-skills subsystem (`ag2/tools/skills/`) implements the
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
A runnable unit of a skill, run on demand via `run_skill_script`. Two forms: a
**file** under a skill's `scripts/` directory (LocalRuntime; invoked with
positional string args), or an **in-process callable** (MemoryRuntime; invoked
with named args). Disjoint from a Resource. An in-process Script carries a
JSON-schema for its parameters, disclosed inside the loaded skill content (not as
a separate tool).

**Runtime**:
The backend that owns a set of skills — it discovers them (`runtime.skills`) and
performs all IO on them (`read` / `read_resource` / `execute`). `LocalRuntime`
backs skills with the filesystem; a `MemoryRuntime` backs them with RAM. Skills,
Scripts, and Resources are inert descriptors; the Runtime does the reading and
running.
_Avoid_: store, source, provider (for this concept).

**MemorySkill**:
A Skill defined inline in code rather than discovered on disk — it carries its
instructions, Resources, and Scripts as in-memory values instead of files. Owned
by a `MemoryRuntime`. A MemorySkill's Scripts are in-process callables (not
`scripts/` files) run through the same `run_skill_script` tool; their parameter
JSON-schema is disclosed inside the loaded skill content.
_Avoid_: code skill, inline skill (informal).

**Shadowing**:
When the same skill name exists in more than one Runtime, the **last** Runtime
passed to the Toolkit/Plugin wins (project overrides global). The rule is applied
uniformly to routing, reads, the catalog, and tool gating.
