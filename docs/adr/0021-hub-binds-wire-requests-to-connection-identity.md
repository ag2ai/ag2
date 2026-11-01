---
status: accepted
date: 2026-09-30
---

# 0021. The hub authorizes wire requests against the connection's bound identities

## Context

A network client proves who it is once per connection: the `register` op
validates the passport's auth claim, and a `HelloFrame` does the same when an
existing identity re-attaches. Either binds the connection's endpoint to that
`agent_id` (`Hub.bind_endpoint`), and one connection may hold several agents.

The wire control plane (`Hub._dispatch_request_op`) then maps each request to a
hub method. Those methods take the identity they act on — `agent_id`,
`creator_id`, an envelope's `sender_id`, a task's owner — as a plain argument,
because the same methods back in-process callers, the hub's own sweepers, and
expectation handlers, none of which have a connection. The endpoint binding was
used only to route notifies, so a request could name any agent.

## Decision

**Every wire request is authorized in the dispatcher, against the set of agents
bound to the calling endpoint, before the hub method runs.** Hub methods keep
their signatures and stay unchecked.

`Hub._authorize_request` classifies each op:

- **Unscoped** — `register` and discovery reads (agents, resumes, skills,
  rules). A fresh connection runs these before its `HelloFrame`. `get_task` /
  `list_tasks` are filtered to the tasks the connection may see.
- **Agent-scoped** — the named agent must be bound: identity mutation,
  `unregister`, `create_channel`, `can_send`, `pending_turns_for`,
  `report_turn_failure`, `record_observation`, the `sender_id` of
  `post_envelope`, the owner in `observe_task`.
- **Channel-scoped** — a bound agent must be a participant: `get_channel`,
  `close_channel`, `read_wal`, `find_envelope_by_causation`. `list_channels`
  without an `agent_id` lists only those channels.
- **Task-scoped** — when the hub has observed the task, its owner must be bound.
  A checkpoint of a task the hub has not observed belongs to the agents bound to
  the connection that first wrote it (recorded next to the checkpoint); later
  reads and writes need one of them, and so does `observe_task` for that id,
  since observing makes the caller the task's owner.

An op in no class is rejected, so a new op is unreachable until it is
classified. A `ReceiptFrame` for an agent not bound to the connection is dropped.
Denials raise `AccessDeniedError`, returned as an `access_denied` response; the
connection stays open.

The binding is only as strong as admission, so a `HelloFrame` re-attaching an
existing identity is validated with the scheme that identity registered with; a
Hello naming another scheme — e.g. `none` in a registry that also holds `NoAuth`
— is refused with `auth_failed`.

Admission is also only as strong as the secrecy of the claim. The hub does not
keep an agent's `auth.claim` once registration validated it — adapters check a
Hello's claim against their own configuration (`ApiKeyAuth` against its keys),
not against a stored copy — so it is dropped before the passport is cached or
persisted, and `hydrate` drops and rewrites it in older stores. Passports in
wire responses (`register`, `get_agent`, `list_agents`) never carry a claim. A
client re-attaching with `attach` therefore supplies its own passport; the hub
cannot hand the credential back, and adapters validate a Hello's claim against
their own configuration because the stored passport's claim is always empty (the
`AuthAdapter` contract says so; `NoAuth` and `ApiKeyAuth` comply). (A `remote_agent`'s claim is proxy routing
data; it stays in-process and is stripped from wire responses too.)

A `kind="remote_agent"` passport is different: its `auth.scheme` is a routing
label for a `RemoteAgentProxy`, not a credential, so registration never
validates it. The federation operator normally registers such identities on
the hub directly (`Hub.register_identity`) and its proxy posts for them
in-process; they have no connection of their own. Registration skips auth for
them, so allowing it over the wire would let any admitted client take an unused
name that the auth registry could authenticate — a real agent's name — and
receive the traffic addressed to it. Wire registration of `remote_agent` is
therefore **off by default** (`Hub(allow_remote_agent_registration=False)`);
an operator who needs wire clients to register remote agents turns it on,
accepting that unused names are first-come and that the registrant's
`auth.scheme` / `auth.claim` become the routing data its proxy dispatches with. We preferred this over namespacing
remote names, which would change every federated agent's name. When enabled:

- registering a `remote_agent` needs a connection that already holds a
  non-remote agent of this hub. Those agents become its **owners**, persisted
  at `agents/{id}/owners.json` and reloaded on `hydrate`;
- a Hello for a `remote_agent` is accepted only on a connection holding one of
  its owners, and never on the strength of a scheme. One registered in-process
  has no owners and cannot be re-attached over the wire at all.

Without this, a connection with no credential could register a `remote_agent`
under an unused name, capturing traffic meant for it, or re-attach as one with
`auth_scheme="none"` wherever the registry holds `NoAuth`.

Every task id a wire request names (`task_id`, `metadata.task_id`,
`envelope.task_id`) must be one path segment — `[A-Za-z0-9][A-Za-z0-9._-]{0,127}`,
which covers `uuid4().hex` and ids like `task-1` — because task ids become
store paths. Anything else is a `ProtocolError`.

Every task-scoped op needs a bound agent. `fire_task_event` reaches listeners
verbatim — the audit log records terminal kinds keyed on the payload's
`owner_id` / `channel_id` / `outcome` / `capability`, and telemetry opens a task
span linked to the payload's channel — while the hub emits real terminal events
itself from `update_task`. So over the wire the only task event accepted is the
`TaskMirror`'s `mirror_failed` report, with exactly its payload keys (`op`,
`owner_id`, `channel_id`, `exc_type`, `exc_message`). Its `owner_id` must be
bound and, for an observed task, be that task's owner with the task's own
channel; for an unobserved task any `channel_id` must be one the owner
participates in. Telemetry still records `mirror_failed` as a failed task span.
For an unobserved task id whose checkpoint another connection wrote, the report is denied like any other access to that checkpoint. Its owner is always the reporting agent; for an observed task it carries that
task's own id and channel, while for an unobserved task it carries whatever
pattern-valid task id the reporter names and either no channel or one the
reporter is in.

A peer's cancel request is a protocol event, so it bypasses the recipient's
inbox cap; each sender may therefore send only one per task. The hub records
the senders per task in memory, checks and extends that record under the
channel's WAL lock (so concurrent duplicates cannot both pass and no request
reads the WAL), drops it when the task turns terminal, and rebuilds it from
active channels' WALs on `hydrate`.
`record_observation` deduplicates per `(owner_id, task_id)`, so an agent
reporting a task id first cannot suppress the real owner's observation.

Inbound frames are decoded JSON, so field types are not guaranteed. A
`HelloFrame`, `ReceiptFrame` or `RequestFrame` whose fields have the wrong JSON
type gets an `ErrorFrame(code="protocol_error")` and the connection stays open,
matching how an unknown op or a failed handshake is answered.

Agent ids are what bindings and routing key on, so a name can never stand in for
one: `get_agent` and the workflow adapter's handoff routing resolve an exact id
before a name, and registering a name that equals an agent id or has its shape
(32 lowercase hex) is a `ProtocolError`. Names that merely resemble an id or
another name (upper-case hex, dashed UUIDs, trailing spaces, look-alike
characters) are still accepted: they cannot misroute, but can mislead a reader.

A `HelloFrame` whose `auth_claim` is not a JSON object is refused with
`auth_failed` before any adapter sees it.

Over the wire, `ag2.channel.*` events other than `invite.ack` / `invite.reject`
(invite, opened, closed, expired) are emitted only by the hub and rejected from
clients: they drive the channel state machine and clients (e.g. `delegate`)
treat `closed` as terminal, so a forged one ends waits early.

Independently of the connection, `post_envelope` accepts events only from the
channel's participants, in the hub rather than per adapter, so an adapter that
forgets the check cannot let outsiders inject content. This covers protocol
events too — an outsider's `ag2.channel.invite.reject` would otherwise fail
another agent's handshake. Hub-generated protocol envelopes carry the creator,
and invitees are participants from creation. The one exception is
`ag2.task.cancel_request` shaped as the `tasks` tool sends it, which any peer
may post: its `task_id` names a live task the hub has observed in that same,
active channel, its audience is exactly `[task owner]`, and its `event_data` is
exactly `{"task_id", "reason"}` with a string `reason`. For that to mean anything, a task's channel
must be trustworthy, so `observe_task` over the wire requires the owner to be a
participant of the task's `channel_id`.

## Consequences

- The trust boundary is the wire. An in-process `HubClient` with a hub reference,
  and code holding the `Hub`, act as any agent — they already run inside the
  hub's process.
- A channel participant reads the whole WAL, including envelopes whose
  `audience` excludes it: the client re-folds adapter state from the full WAL,
  so filtering by audience would diverge it from the hub's fold.
- Re-attaching an agent from a new connection moves its authority there; the old
  connection's late requests and receipts for it are rejected or dropped.
- Task records are visible over the wire only to the task's owner and to the
  participants of its channel: `get_task` answers `not_found` for any other
  task, and `list_tasks` leaves them out. A delegator polls the tasks it
  delegated because it shares their channel; `scope="all"` lists what the
  connection may see, not every task on the hub. A task with no channel is
  visible to its owner alone.
- A wire envelope's `audience` may name only participants of its channel, and
  a peer cancel request's `reason` is at most 500 characters, as is the free
  text of a `mirror_failed` payload (longer is clipped). `report_turn_failure`
  needs the named agent to participate in the channel.
- A checkpoint written only in-process (or before writers were recorded) has no
  recorded writer: any connection with a bound agent can read it, and the first
  one to write it over the wire claims it.
- Recording a checkpoint's first writer is serialised by a hub lock, so two
  connections racing to checkpoint the same new id cannot both claim it.
  Squatting stays: an id nobody has checkpointed yet goes to whoever writes it
  first.
- `get_resume` and `get_skill` are discovery reads (the `peers` tool uses
  them), and `get_rule` is read by `attach` before the Hello, so none is
  scoped to a bound agent.
- A registry holding `NoAuth` admits any client as any agent, so `serve_ws`
  raises `ValueError` for such a hub on a non-loopback host. Operators who
  accept that pass `allow_unauthenticated=True`, which logs a warning instead.
