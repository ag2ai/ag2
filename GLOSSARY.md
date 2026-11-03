# AG2

Domain language for agent delegation. See the
[subagent event routing guide](docs/design/subagent-eventbus-v2.md) for the API.

## Language

**Delegation**:
A particular assignment of work from a parent agent to a subagent. Repeated
assignments to the same agent are distinct delegations.

**Subagent**:
An agent performing work delegated by another agent; that agent may in turn
delegate work to descendants.

**Delegation subtree**:
A delegation together with all delegations descended from it.

**Uplink**:
The upward flow of a delegation subtree's events to its ancestors.

**Downlink**:
The delivery of an event from an ancestor to a selected subagent delegation.

**Event origin**:
The agent that directly authored an event, regardless of how many ancestors
subsequently receive it.

**Observation envelope**:
The `ChildEvent` in which an ancestor receives a descendant's operational
event, so that the event cannot trigger work in the ancestor.

**Inert event**:
An event that cannot trigger work in an ancestor (usage records and task
lifecycle events), so ancestors receive it as itself, stamped with the
delegation it came from.
_Avoid_: Unwrapped event, raw event

**Scoped value**:
A value declared with explicit rules for its visibility and inheritance within
a delegation tree.

**Local variable**:
A variable visible only to its owning agent during an invocation.

**Copied variable**:
A variable whose value is inherited independently by each descendant, with
changes confined to the descendant's branch.

**Shared variable**:
A variable declared by the root whose current value is shared throughout a
delegation tree; changes survive a subagent's failure.
_Avoid_: Inherited variable, write-back variable

**Source usage record**:
The record of one unit of token-consuming agent work, attributable to its
originating agent through the delegation stamped on it; appearances of that
record at ancestors describe the same expenditure.
