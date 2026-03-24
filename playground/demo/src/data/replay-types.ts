/** Replay data types for the AG2 demo playground. */

/** Top-level replay file for one example */
export interface ReplayData {
  example: ExampleMeta
  scenarios: ScenarioReplay[]
}

export interface ExampleMeta {
  id: string
  title: string
  category: string
  description: string
  themes: Array<'network' | 'distributed' | 'autonomy'>
}

/** For distributed examples: server topology */
export interface ServerDef {
  id: string
  label: string
  agents: string[]
  color: string
  x: number
  y: number
  w: number
  h: number
}

export interface ServerConnection {
  from: string
  to: string
  label: string
}

/** A single scenario replay */
export interface ScenarioReplay {
  id: number
  title: string
  inputMessage: string
  agents: AgentDef[]
  servers?: ServerDef[]
  serverConnections?: ServerConnection[]
  events: ReplayEvent[]
  totalDurationMs: number
}

/** Agent definition for the graph */
export interface AgentDef {
  id: string
  name: string
  role: string
  tools: string[]
  server?: string
  color: string
  x: number
  y: number
}

/** All event types */
export type ReplayEvent =
  | DelegationRequestEvent
  | DelegationResultEvent
  | ToolCallEvent
  | ToolResultEvent
  | ModelResponseEvent
  | DiscoverAgentsEvent
  | HubConnectEvent
  | SchedulerTriggerEvent

interface BaseEvent {
  id: string
  timestamp: number
  agent: string
}

export interface DelegationRequestEvent extends BaseEvent {
  type: 'delegation-request'
  source: string
  target: string
  channel: 'local' | 'http'
  taskPreview: string
}

export interface DelegationResultEvent extends BaseEvent {
  type: 'delegation-result'
  source: string
  target: string
  resultPreview: string
}

export interface ToolCallEvent extends BaseEvent {
  type: 'tool-call'
  toolName: string
  args: string
}

export interface ToolResultEvent extends BaseEvent {
  type: 'tool-result'
  toolName: string
  result: string
}

export interface ModelResponseEvent extends BaseEvent {
  type: 'model-response'
  contentPreview: string
}

export interface DiscoverAgentsEvent extends BaseEvent {
  type: 'discover-agents'
  capability: string
  results: string[]
}

export interface HubConnectEvent extends BaseEvent {
  type: 'hub-connect'
  targetServer: string
  agentsDiscovered: string[]
}

export interface SchedulerTriggerEvent extends BaseEvent {
  type: 'scheduler-trigger'
  target: string
  watch: string
  task: string
}
