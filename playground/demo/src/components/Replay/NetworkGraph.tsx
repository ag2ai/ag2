import { motion } from 'framer-motion'
import type { AgentDef, ServerDef, ServerConnection, ReplayEvent } from '../../data/replay-types'

const COLORS: Record<string, { fill: string; stroke: string; text: string; glow: string }> = {
  blue:    { fill: '#3b82f620', stroke: '#3b82f6', text: '#60a5fa', glow: '#3b82f640' },
  cyan:    { fill: '#06b6d420', stroke: '#06b6d4', text: '#22d3ee', glow: '#06b6d440' },
  emerald: { fill: '#10b98120', stroke: '#10b981', text: '#34d399', glow: '#10b98140' },
  amber:   { fill: '#f59e0b20', stroke: '#f59e0b', text: '#fbbf24', glow: '#f59e0b40' },
  rose:    { fill: '#f4364820', stroke: '#f43648', text: '#fb7185', glow: '#f4364840' },
  violet:  { fill: '#8b5cf620', stroke: '#8b5cf6', text: '#a78bfa', glow: '#8b5cf640' },
  yellow:  { fill: '#eab30820', stroke: '#eab308', text: '#facc15', glow: '#eab30840' },
}

function getColor(name: string) {
  return COLORS[name] || COLORS.blue
}

interface NetworkGraphProps {
  agents: AgentDef[]
  servers?: ServerDef[]
  serverConnections?: ServerConnection[]
  activeAgent: string | null
  activeDelegations: Array<{ source: string; target: string; channel: 'local' | 'http' }>
  visibleEvents: ReplayEvent[]
}

export function NetworkGraph({ agents, servers, serverConnections, activeAgent, activeDelegations, visibleEvents }: NetworkGraphProps) {
  const svgW = 500
  const svgH = 260

  // Which agents have appeared in events so far
  const appearedAgents = new Set(visibleEvents.map(e => e.agent))
  // Also add targets of delegations and hub-connect discovered agents
  for (const e of visibleEvents) {
    if (e.type === 'delegation-request') { appearedAgents.add(e.target); appearedAgents.add(e.source) }
    if (e.type === 'delegation-result') { appearedAgents.add(e.target); appearedAgents.add(e.source) }
    if (e.type === 'hub-connect') { for (const a of e.agentsDiscovered) appearedAgents.add(a) }
  }

  // Check which agents are "busy" (have a tool call without a result yet)
  const busyAgents = new Set<string>()
  const toolCalls = new Map<string, string>()
  for (const e of visibleEvents) {
    if (e.type === 'tool-call') toolCalls.set(e.agent, e.toolName)
    if (e.type === 'tool-result') toolCalls.delete(e.agent)
  }
  for (const [agent] of toolCalls) busyAgents.add(agent)

  const agentPos = (id: string) => {
    const a = agents.find(a => a.id === id)
    if (!a) return { x: 250, y: 130 }
    return { x: (a.x / 100) * svgW, y: (a.y / 100) * svgH }
  }

  return (
    <svg viewBox={`0 0 ${svgW} ${svgH}`} className="w-full" style={{ maxHeight: '340px' }}>
      <defs>
        <marker id="arrowLocal" viewBox="0 0 8 6" refX="8" refY="3" markerWidth="8" markerHeight="6" orient="auto">
          <path d="M0 0 L8 3 L0 6z" fill="#8b5cf6" />
        </marker>
        <marker id="arrowHttp" viewBox="0 0 8 6" refX="8" refY="3" markerWidth="8" markerHeight="6" orient="auto">
          <path d="M0 0 L8 3 L0 6z" fill="#06b6d4" />
        </marker>
        <marker id="arrowReturn" viewBox="0 0 8 6" refX="8" refY="3" markerWidth="8" markerHeight="6" orient="auto">
          <path d="M0 0 L8 3 L0 6z" fill="#6b728080" />
        </marker>
      </defs>

      {/* Server boxes */}
      {servers?.map(server => {
        const c = getColor(server.color)
        const sx = (server.x / 100) * svgW
        const sy = (server.y / 100) * svgH
        const sw = (server.w / 100) * svgW
        const sh = (server.h / 100) * svgH
        return (
          <g key={server.id}>
            <motion.rect
              x={sx + 4} y={sy + 4} width={sw - 8} height={sh - 8} rx="8"
              fill="none" stroke={c.stroke} strokeWidth="1" strokeDasharray="6 4" opacity={0.25}
              initial={{ opacity: 0 }} animate={{ opacity: 0.25 }} transition={{ duration: 0.5 }}
            />
            <text x={sx + sw / 2} y={sy + 16} textAnchor="middle"
              className="text-[8px] font-mono font-bold" fill={c.text} opacity={0.6}>
              {server.label}
            </text>
          </g>
        )
      })}

      {/* Server connections (HttpChannel labels) */}
      {serverConnections?.map((conn, i) => {
        const fromServer = servers?.find(s => s.id === conn.from)
        const toServer = servers?.find(s => s.id === conn.to)
        if (!fromServer || !toServer) return null
        const fromCx = ((fromServer.x + fromServer.w / 2) / 100) * svgW
        const fromCy = ((fromServer.y + fromServer.h / 2) / 100) * svgH
        const toCx = ((toServer.x + toServer.w / 2) / 100) * svgW
        const toCy = ((toServer.y + toServer.h / 2) / 100) * svgH
        const midX = (fromCx + toCx) / 2
        const midY = (fromCy + toCy) / 2
        return (
          <g key={i}>
            <line x1={fromCx} y1={fromCy} x2={toCx} y2={toCy}
              stroke="#06b6d4" strokeWidth="0.8" strokeDasharray="3 3" opacity={0.2} />
            <text x={midX} y={midY - 4} textAnchor="middle"
              className="text-[7px] font-mono" fill="#06b6d4" opacity={0.4}>
              {conn.label}
            </text>
          </g>
        )
      })}

      {/* Delegation arrows */}
      {activeDelegations.map((del, i) => {
        const from = agentPos(del.source)
        const to = agentPos(del.target)
        const isHttp = del.channel === 'http'
        const dx = to.x - from.x
        const dy = to.y - from.y
        const len = Math.sqrt(dx * dx + dy * dy)
        const nx = dx / len
        const ny = dy / len
        const r = 20
        const x1 = from.x + nx * r
        const y1 = from.y + ny * r
        const x2 = to.x - nx * (r + 8)
        const y2 = to.y - ny * (r + 8)

        return (
          <g key={`del-${i}`}>
            <motion.line
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke={isHttp ? '#06b6d4' : '#8b5cf6'}
              strokeWidth={isHttp ? 2 : 1.5}
              strokeDasharray={isHttp ? '6 3' : 'none'}
              markerEnd={isHttp ? 'url(#arrowHttp)' : 'url(#arrowLocal)'}
              initial={{ opacity: 0 }} animate={{ opacity: 0.8 }} transition={{ duration: 0.3 }}
            />
            {/* Flowing dot */}
            <motion.circle r="3" fill={isHttp ? '#06b6d4' : '#8b5cf6'}
              animate={{
                cx: [x1, x2],
                cy: [y1, y2],
                opacity: [0.9, 0.9, 0],
              }}
              transition={{ repeat: Infinity, duration: 1.2, ease: 'linear' }}
            />
            {isHttp && (
              <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle"
                className="text-[7px] font-mono font-bold" fill="#06b6d4">
                HTTP
              </text>
            )}
          </g>
        )
      })}

      {/* Completed delegation return arrows (subtle) */}
      {visibleEvents
        .filter((e): e is import('../../data/replay-types').DelegationResultEvent => e.type === 'delegation-result')
        .map((e, i) => {
          const from = agentPos(e.target)
          const to = agentPos(e.source)
          const dx = to.x - from.x
          const dy = to.y - from.y
          const len = Math.sqrt(dx * dx + dy * dy)
          if (len === 0) return null
          const nx = dx / len
          const ny = dy / len
          const r = 20
          const offset = 4
          const perpX = -ny * offset
          const perpY = nx * offset
          return (
            <line key={`ret-${i}`}
              x1={from.x + nx * r + perpX} y1={from.y + ny * r + perpY}
              x2={to.x - nx * (r + 8) + perpX} y2={to.y - ny * (r + 8) + perpY}
              stroke="#6b7280" strokeWidth="0.8" strokeDasharray="2 2" opacity={0.25}
              markerEnd="url(#arrowReturn)"
            />
          )
        })}

      {/* Agent nodes */}
      {agents.map(agent => {
        const c = getColor(agent.color)
        const pos = agentPos(agent.id)
        const isActive = activeAgent === agent.id
        const appeared = appearedAgents.has(agent.id)
        const isBusy = busyAgents.has(agent.id)
        const nodeOpacity = appeared ? 1 : 0.3

        return (
          <g key={agent.id} opacity={nodeOpacity}>
            {/* Glow for active agent */}
            {isActive && (
              <motion.circle cx={pos.x} cy={pos.y} r="24" fill="none" stroke={c.stroke} strokeWidth="1.5"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: [0.3, 0.6, 0.3], scale: [0.95, 1.05, 0.95] }}
                transition={{ repeat: Infinity, duration: 1.5 }}
              />
            )}

            {/* Main circle */}
            <circle cx={pos.x} cy={pos.y} r="18"
              fill={isActive ? c.glow : c.fill}
              stroke={c.stroke}
              strokeWidth={isActive ? 2 : 1}
            />

            {/* Agent name */}
            <text x={pos.x} y={pos.y + 1} textAnchor="middle"
              className="text-[9px] font-mono font-bold" fill={c.text}>
              {agent.name.length > 10 ? agent.name.slice(0, 9) + '…' : agent.name}
            </text>

            {/* Role below */}
            <text x={pos.x} y={pos.y + 32} textAnchor="middle"
              className="text-[7px] font-mono" fill="#6b7280">
              {agent.role}
            </text>

            {/* Tool call badge */}
            {isBusy && (
              <motion.g
                initial={{ scale: 0 }} animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 400, damping: 15 }}
              >
                <circle cx={pos.x + 14} cy={pos.y - 14} r="6" fill="#f59e0b" />
                <text x={pos.x + 14} y={pos.y - 11} textAnchor="middle"
                  className="text-[7px] font-bold" fill="#000">
                  T
                </text>
              </motion.g>
            )}
          </g>
        )
      })}
    </svg>
  )
}
