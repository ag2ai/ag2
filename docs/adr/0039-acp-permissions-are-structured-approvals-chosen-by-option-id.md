---
status: accepted
date: 2026-11-21
---

# 0039. ACP permissions are structured approvals, chosen by `option_id`

## Context

An ACP agent asks before a sensitive action with `session/request_permission`.
The request carries the tool call and a list of options, each with an
`optionId`, a `name` and a `kind` (`allow_once`, `allow_always`, `reject_once`
or `reject_always`). The spec leaves the choice to the user: the client shows
the options and answers with the `optionId` of the one picked. The agent owns
what an option means. An `allow_always` option's `name` says what it will
remember (Gemini CLI: "Allow for this session", keyed to the command's root),
and `kind` is only a UI hint.

Under `permission_policy="ask"`, AG2 put a `yes/no` question to the human and
answered any yes with the agent's *first* allow option. Gemini CLI lists
`allow_always` first, so one "yes" to `git status` approved every later `git`
command for the session (GHSA-3r6w-7pcg-c4x2). Picking the option by kind
instead would fix the order bug but keep the cause: AG2 collapsing the agent's
options into yes/no and guessing which one the human meant.

AG2 already asks about tool calls through `ToolApprovalRequest`, which
`approval_required()` uses and AG-UI renders as an approval interrupt with a
boolean answer.

## Decision

1. **Under `"ask"`, the request goes to the human as an
   `ACPPermissionRequest(ToolApprovalRequest)`** carrying the agent's options,
   the tool call's `kind`, `locations` and `raw_input`, and its title as
   `content`. AG2 renders nothing itself; the hook or transport does.
2. **A selection is typed, not a string.** The human picks an option with an
   `ApprovalChoiceMessage(HumanMessage)` holding its `option_id`, and that
   option is selected, rejections included. Ids and words share one string
   space and the agent picks the ids, so a selection and a yes or no must not
   be told apart by their text: an agent offering an allow option with id
   `"n"` would turn AG-UI's refuse button into a grant, and a rejecting option
   with id `"y"` would read as a yes. `Context.ask_message()` returns the whole
   reply so the asker can see its type.
3. **A plain approval answer is a word, never an id, and is never widened.**
   So that hooks and UIs that only know tool approvals keep working,
   `approval_required()`'s words are accepted: `yes` / `y` / `1` select the
   one-time allow option (or reject if there is none), `always` selects the
   remembered one, anything else rejects.
4. **`"auto"` and `"deny"` select the one-time option**, and the remembered one
   only when the agent offers no other. The agent's option order never decides.
5. **`ToolApprovalRequest.choices` (a list of `ApprovalChoice(value, label)`) is the
   transport-neutral way to offer more than yes or no.** AG-UI adds an object
   branch to the interrupt's `responseSchema` whose `choice` is one titled
   constant per choice, and a pick is resumed as `{"choice": value}`. A pick has
   its own wire shape rather than being inferred from a string matching a
   value: existing clients answer in words (a typed refusal included), and a
   string that happened to equal a value would otherwise change meaning. So a
   boolean reaches the asker as `"y"` / `"n"`, a string as itself, and only a
   `{"choice": ...}` payload naming an offered value as an
   `ApprovalChoiceMessage`; anything else under that shape is refused. `ACPPermissionRequest` fills
   `choices` from its options, so `ag_ui` needs no knowledge of ACP.

## Consequences

- A hook that returns an option's id as a plain string rejects; it must
  return an `ApprovalChoiceMessage` to select it.
- A hook that only prints `content` shows the title, not the options. Showing
  them is the hook's or UI's job; the fields are on the event.
- The words `ok`, `allow`, `approve`, `allow_once` and `allow_always`, accepted
  before this change, are still accepted, with the meaning above.
- `cancelled` is still sent when no offered option fits (for example `"deny"`
  with no reject option). The spec reserves it for a cancelled turn, but it is
  the only outcome left.
- `approval_required()` can offer `always` to AG-UI clients through `choices`;
  that is a separate change.
