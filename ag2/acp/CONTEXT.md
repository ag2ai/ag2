# ACP

AG2's integration with the [Agent Client Protocol](https://agentclientprotocol.com). The
protocol runs in two directions and AG2 implements both, which is the single largest source
of confusion in this area: AG2 can *drive* an external agent, and AG2 can *be* the agent
somebody else drives. Nearly every term below means something different depending on which
direction is meant, so name the direction first.

## Language

### Roles

**ACP Client**:
The side that launches the other process and drives the conversation. Editors and IDEs are
the usual example. AG2 plays this role when it runs an external CLI coding agent.
_Avoid_: host, caller, consumer

**ACP Agent**:
The side that answers. It is launched by the ACP Client as a subprocess and speaks over the
pipe between them. AG2 plays this role when an external Client drives an AG2 agent.
_Avoid_: ACP Server, endpoint, provider — ACP defines no "server", and borrowing the word
from MCP or A2A points at a role the protocol does not have.

**Serving direction**:
AG2 in the ACP Agent role — an AG2 agent exposed to an outside Client.
_Avoid_: server side, inbound

**Consuming direction**:
AG2 in the ACP Client role — AG2 driving an outside CLI coding agent.
_Avoid_: client side, outbound

### Serving direction

**Connection**:
One Client's link to the AG2 agent, lasting from process start to disconnect. It owns the
authorization state and the set of sessions; nothing meaningful is shared between two
connections.

**Session**:
One independent conversation inside a Connection, with its own message history and its own
context variables. Sessions never see each other's messages. They separate conversations,
not tenants — every session runs the same underlying agent, so its tools and knowledge are
reachable from all of them equally.
_Avoid_: thread, chat, conversation

**Prompt**:
One request from the Client to run the agent. Exactly one prompt is one ordinary AG2 turn,
through the same path an off-protocol `ask()` takes. Two prompts arriving on one Session
queue rather than interleave.

**Session Update**:
A notification pushed to the Client *while* a prompt is running — generated text, reasoning,
tool calls, tool results — in the order they happened.
_Avoid_: event, stream chunk, progress

**Prompt Content**:
The declaration of which non-text input kinds this agent accepts — image, audio, embedded
document. Declared by whoever builds the agent rather than inferred, because the answer
depends on the model behind it.

**Human-Input Channel**:
Whatever can put a question from a running turn to an actual person — the `hitl_hook` an
embedding application passes to `ACPAgent`, or nothing at all. ACP elicitation is not wired,
so the protocol is never that channel in this version. A turn whose question the channel
cannot answer fails, and the protocol error carries `data["category"] == "human_input"` so a
Client can tell it apart from any other internal error.
_Avoid_: HITL, approval flow, elicitation — elicitation is a specific ACP feature this
version does not implement, and borrowing its name for the hook claims otherwise.

**Client Capability**:
Something the Client offers *to* the Agent — filesystem reads and writes against the
editor's live buffers, terminals, permission prompts. Optional in the protocol, and AG2's
serving direction uses none of them: an AG2 agent reaches files through its own tools.

### Consuming direction

**Tool Gateway**:
The mechanism by which AG2 makes its own locally-executable tools callable by an external
CLI agent. It exists only in the consuming direction. Nothing in the serving direction has
a gateway, and an AG2 agent being served does not connect to MCP servers the Client declares.
_Avoid_: tool bridge, tool proxy, tool server
