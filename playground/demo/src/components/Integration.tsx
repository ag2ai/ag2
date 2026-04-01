import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

/* ── Ecosystem Hub Visualization ──────────────────────────────────────── */

function EcosystemDiagram() {
  const hubX = 300, hubY = 200

  // Arrange entities in rings around AG2 Hub
  const innerRing = [
    { label: 'A2A', sub: 'Channel', x: 300, y: 80, color: '#06b6d4', size: 28 },
    { label: 'MCP', sub: 'Server/Client', x: 460, y: 150, color: '#10b981', size: 28 },
    { label: 'AGNTCY', sub: 'Infra layer', x: 460, y: 280, color: '#a855f7', size: 24 },
    { label: 'HTTP', sub: 'Transport', x: 300, y: 330, color: '#f59e0b', size: 22 },
    { label: 'gRPC', sub: 'Transport', x: 140, y: 280, color: '#f59e0b', size: 22 },
    { label: 'NATS', sub: 'Transport', x: 140, y: 150, color: '#f59e0b', size: 22 },
  ]

  const outerRing = [
    { label: 'Google ADK', x: 195, y: 38, color: '#3b82f6' },
    { label: 'LangGraph', x: 405, y: 38, color: '#3b82f6' },
    { label: 'Claude Code', x: 540, y: 110, color: '#f97316' },
    { label: 'ChatGPT', x: 555, y: 190, color: '#10b981' },
    { label: 'Cursor', x: 540, y: 270, color: '#8b5cf6' },
    { label: 'AWS Bedrock', x: 470, y: 350, color: '#f97316' },
    { label: 'OpenAI SDK', x: 300, y: 385, color: '#10b981' },
    { label: 'CrewAI', x: 130, y: 350, color: '#ec4899' },
    { label: 'Devin', x: 50, y: 270, color: '#8b5cf6' },
    { label: 'VS Code', x: 40, y: 190, color: '#3b82f6' },
    { label: 'Codex', x: 50, y: 110, color: '#10b981' },
  ]

  return (
    <FadeIn>
      <div className="relative max-w-2xl mx-auto">
        <svg viewBox="0 0 600 420" className="w-full overflow-visible">
          {/* Outer glow rings */}
          {[140, 170, 200].map((r, i) => (
            <motion.circle key={r} cx={hubX} cy={hubY} r={r}
              fill="none" stroke="#3b82f6" strokeWidth="0.4"
              animate={{ opacity: [0, 0.08, 0], r: [r - 5, r, r + 5] }}
              transition={{ repeat: Infinity, duration: 4, delay: i * 1.2, ease: 'easeOut' }} />
          ))}

          {/* Connection lines: inner ring to hub */}
          {innerRing.map((node, i) => (
            <motion.line key={`inner-${i}`}
              x1={hubX} y1={hubY} x2={node.x} y2={node.y}
              stroke={node.color} strokeWidth="1.5" opacity="0.25"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
              viewport={{ once: true }} transition={{ delay: 0.3 + i * 0.08 }} />
          ))}

          {/* Connection lines: outer ring to nearest inner ring node */}
          {outerRing.map((outer, i) => {
            // Find nearest inner ring node
            let minDist = Infinity, nearest = innerRing[0]
            for (const inner of innerRing) {
              const d = Math.sqrt((outer.x - inner.x) ** 2 + (outer.y - inner.y) ** 2)
              if (d < minDist) { minDist = d; nearest = inner }
            }
            return (
              <motion.line key={`outer-${i}`}
                x1={nearest.x} y1={nearest.y} x2={outer.x} y2={outer.y}
                stroke={outer.color} strokeWidth="0.6" opacity="0.15"
                strokeDasharray="3 4"
                initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
                viewport={{ once: true }} transition={{ delay: 0.6 + i * 0.05 }} />
            )
          })}

          {/* AG2 Hub center */}
          <motion.circle cx={hubX} cy={hubY} r="42"
            fill="url(#ecosystemHubGrad)" stroke="#3b82f6" strokeWidth="2"
            initial={{ scale: 0 }} whileInView={{ scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 0.2, type: 'spring', stiffness: 150 }} />
          <motion.circle cx={hubX} cy={hubY} r="42"
            fill="none" stroke="#60a5fa" strokeWidth="0.5"
            animate={{ r: [42, 45, 42] }}
            transition={{ repeat: Infinity, duration: 3, ease: 'easeInOut' }} />
          <text x={hubX} y={hubY - 4} textAnchor="middle" className="fill-white text-[13px] font-mono font-bold">AG2 V2</text>
          <text x={hubX} y={hubY + 12} textAnchor="middle" className="fill-blue-300 text-[9px] font-mono">Network Hub</text>

          {/* Inner ring nodes */}
          {innerRing.map((node, i) => (
            <motion.g key={node.label}
              initial={{ opacity: 0, scale: 0 }} whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }} transition={{ delay: 0.4 + i * 0.08, type: 'spring' }}>
              <circle cx={node.x} cy={node.y} r={node.size} fill={node.color + '15'} stroke={node.color} strokeWidth="1.5" />
              <text x={node.x} y={node.y - 2} textAnchor="middle" className="text-[9px] font-mono font-bold" fill={node.color}>{node.label}</text>
              <text x={node.x} y={node.y + 10} textAnchor="middle" className="text-[7px] font-mono fill-gray-500">{node.sub}</text>
            </motion.g>
          ))}

          {/* Outer ring nodes */}
          {outerRing.map((node, i) => (
            <motion.g key={node.label}
              initial={{ opacity: 0, scale: 0 }} whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }} transition={{ delay: 0.7 + i * 0.04, type: 'spring' }}>
              <circle cx={node.x} cy={node.y} r="16" fill={node.color + '10'} stroke={node.color} strokeWidth="0.8" />
              <text x={node.x} y={node.y + 3} textAnchor="middle" className="text-[6px] font-mono" fill={node.color}>{node.label}</text>
            </motion.g>
          ))}

          {/* Flowing data between hub and inner ring */}
          {innerRing.slice(0, 3).map((node, i) => (
            <motion.circle key={`flow-${i}`} r="2.5" fill={node.color}
              animate={{ cx: [hubX, node.x], cy: [hubY, node.y], opacity: [0.7, 0] }}
              transition={{ repeat: Infinity, duration: 2 + i * 0.4, delay: i * 0.6, ease: 'easeOut' }} />
          ))}

          <defs>
            <radialGradient id="ecosystemHubGrad" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0.05" />
            </radialGradient>
          </defs>
        </svg>
      </div>
    </FadeIn>
  )
}

/* ── Protocol Stack ───────────────────────────────────────────────────── */

function ProtocolStackDiagram() {
  const layers = [
    { label: 'AG2 Network', sub: 'Routing, State, Scheduling, Observability', color: 'from-blue-500 to-violet-500', borderColor: 'border-blue-400/40', bgColor: 'bg-blue-500/10', textColor: 'text-white', highlight: true },
    { label: 'A2A', sub: 'Agent-to-Agent (horizontal)', color: 'from-cyan-500 to-blue-500', borderColor: 'border-cyan-500/30', bgColor: 'bg-cyan-500/5', textColor: 'text-cyan-400' },
    { label: 'MCP', sub: 'Agent-to-Tool (vertical)', color: 'from-emerald-500 to-teal-500', borderColor: 'border-emerald-500/30', bgColor: 'bg-emerald-500/5', textColor: 'text-emerald-400' },
  ]

  return (
    <div className="max-w-md mx-auto">
      <div className="space-y-2">
        {layers.map((layer, i) => (
          <motion.div
            key={layer.label}
            initial={{ opacity: 0, scale: 0.9, y: 20 }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 + i * 0.15, duration: 0.5 }}
            className={`relative px-6 py-4 rounded-xl border ${layer.borderColor} ${layer.bgColor} text-center ${layer.highlight ? 'ring-1 ring-blue-400/20' : ''}`}
          >
            <div className={`font-bold font-mono ${layer.textColor} text-lg`}>{layer.label}</div>
            <div className="text-xs text-gray-500 font-mono mt-1">{layer.sub}</div>
            {layer.highlight && (
              <motion.div
                className="absolute -right-2 -top-2 w-5 h-5 rounded-full bg-blue-500 flex items-center justify-center"
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ repeat: Infinity, duration: 2 }}
              >
                <span className="text-[8px] text-white font-bold">!</span>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-600 mt-4 font-mono">
        The emerging agent protocol stack
      </p>
    </div>
  )
}

/* ── Integration Cards ────────────────────────────────────────────────── */

interface IntegrationCardProps {
  name: string; role: string; howItIntegrates: string; icon: React.ReactNode
  color: string; bgColor: string; borderColor: string; adoption: string; delay: number
}

function IntegrationCard({ name, role, howItIntegrates, icon, color, bgColor, borderColor, adoption, delay }: IntegrationCardProps) {
  return (
    <FadeIn delay={delay}>
      <div className={`p-5 rounded-xl border ${borderColor} ${bgColor} h-full`}>
        <div className="flex items-start gap-3 mb-3">
          <div className={`w-9 h-9 rounded-lg ${color} flex items-center justify-center shrink-0`}>{icon}</div>
          <div>
            <h4 className="text-sm font-bold text-white">{name}</h4>
            <p className="text-[10px] text-gray-500 font-mono">{role}</p>
          </div>
        </div>
        <p className="text-sm text-gray-400 leading-relaxed mb-3">{howItIntegrates}</p>
        <div className="text-[10px] text-gray-600 font-mono">{adoption}</div>
      </div>
    </FadeIn>
  )
}

/* ── Main Section ─────────────────────────────────────────────────────── */

export function Integration() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="integration">
      <SectionLabel number="09" title="Integration" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">The hub that connects</span>{' '}
        <span className="text-gray-500">the entire ecosystem.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        MCP for vertical (agent-to-tool). A2A for horizontal (agent-to-agent).
        AG2 for the <span className="text-white">infrastructure layer</span> that makes them
        work together — with any framework, any model, any transport.
      </p>

      {/* Ecosystem visualization */}
      <EcosystemDiagram />

      <div className="mt-12 grid md:grid-cols-2 gap-8">
        {/* Left: Protocol stack */}
        <FadeIn>
          <div className="p-6 rounded-2xl bg-gray-900/50 border border-gray-800">
            <h3 className="text-lg font-bold text-white text-center mb-6">The Protocol Stack</h3>
            <ProtocolStackDiagram />
          </div>
        </FadeIn>

        {/* Right: How integration works */}
        <FadeIn delay={0.1}>
          <div className="p-6 rounded-2xl bg-gray-900/50 border border-gray-800">
            <h3 className="text-lg font-bold text-white mb-6">How It Connects</h3>
            <div className="space-y-4">
              {[
                { protocol: 'A2A', how: 'Plugs in as a Channel backend. AG2 routes, observes, and controls traffic. A2A carries it cross-framework. Any A2A-compliant agent becomes a network peer.', color: 'text-cyan-400', border: 'border-cyan-500/20' },
                { protocol: 'MCP', how: 'Expose Actors as MCP servers — accessible from Claude, ChatGPT, Cursor, VS Code. Actors also consume 10,000+ existing MCP servers as tools.', color: 'text-emerald-400', border: 'border-emerald-500/20' },
                { protocol: 'AGNTCY', how: 'Provides lower-level infra (discovery, identity, messaging). AG2 provides the framework layer above — complementary, not overlapping.', color: 'text-purple-400', border: 'border-purple-500/20' },
                { protocol: 'Any HTTP API', how: 'Any agent behind an HTTP endpoint can be wrapped as a RemoteAgent and join the network. No SDK required — just an endpoint.', color: 'text-amber-400', border: 'border-amber-500/20' },
              ].map(item => (
                <div key={item.protocol} className={`p-3 rounded-lg border ${item.border} bg-gray-950/50`}>
                  <div className={`text-xs font-mono font-bold ${item.color} mb-1`}>{item.protocol}</div>
                  <p className="text-xs text-gray-400 leading-relaxed">{item.how}</p>
                </div>
              ))}
            </div>
          </div>
        </FadeIn>
      </div>

      {/* Integration targets */}
      <FadeIn className="mt-16 mb-8">
        <h3 className="text-2xl font-bold text-white text-center mb-2">Frameworks & Products That Connect</h3>
        <p className="text-gray-500 text-center text-sm max-w-lg mx-auto">
          Two protocols — A2A as Channel + MCP as Server — unlock the entire ecosystem
        </p>
      </FadeIn>

      {/* Primary integrations */}
      <div className="grid md:grid-cols-2 gap-6 mb-8">
        <FadeIn delay={0.1}>
          <div className="p-6 rounded-2xl border border-cyan-500/20 bg-cyan-500/5 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                  VIA A2A
                </span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">Agent Frameworks</h3>
              <p className="text-sm text-gray-400 leading-relaxed mb-4">
                Any A2A-compliant agent becomes a peer in the AG2 network.
                AG2's Hub routes and monitors; A2A carries the traffic cross-framework.
              </p>
              <div className="flex flex-wrap gap-2">
                {['Google ADK', 'LangGraph', 'AWS Bedrock', 'OpenAI Agents SDK', 'CrewAI', 'Semantic Kernel'].map(name => (
                  <span key={name} className="text-[10px] font-mono text-gray-500 bg-gray-800 px-2 py-1 rounded">{name}</span>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>

        <FadeIn delay={0.2}>
          <div className="p-6 rounded-2xl border border-emerald-500/20 bg-emerald-500/5 relative overflow-hidden">
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                  VIA MCP
                </span>
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">AI Products & IDEs</h3>
              <p className="text-sm text-gray-400 leading-relaxed mb-4">
                Expose AG2 Actors as MCP servers — instantly accessible from every MCP-enabled product.
                Actors can also consume 10,000+ existing MCP servers as tools.
              </p>
              <div className="flex flex-wrap gap-2">
                {['Claude Code', 'ChatGPT', 'Codex', 'Cursor', 'VS Code', 'Windsurf', 'Devin'].map(name => (
                  <span key={name} className="text-[10px] font-mono text-gray-500 bg-gray-800 px-2 py-1 rounded">{name}</span>
                ))}
              </div>
            </div>
          </div>
        </FadeIn>
      </div>

      {/* Secondary integrations */}
      <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-4">
        <IntegrationCard
          name="Claude Code" role="Coding Agent"
          howItIntegrates="AG2 network as MCP server. Claude Code invokes entire multi-agent workflows via tool calls."
          icon={<span className="text-sm">C</span>}
          color="bg-orange-500/15 text-orange-400" bgColor="bg-gray-900/50" borderColor="border-gray-800"
          adoption="Hooks + MCP + Plugins" delay={0.1}
        />
        <IntegrationCard
          name="OpenAI Codex" role="Coding Agent"
          howItIntegrates="Bidirectional: AG2 as MCP server for Codex, Codex as MCP server for AG2 Actors."
          icon={<span className="text-sm">O</span>}
          color="bg-green-500/15 text-green-400" bgColor="bg-gray-900/50" borderColor="border-gray-800"
          adoption="MCP + AGENTS.md" delay={0.15}
        />
        <IntegrationCard
          name="Google ADK" role="Agent Framework"
          howItIntegrates="Native A2A support. AG2 A2AChannel enables direct peer communication with ADK agents."
          icon={<span className="text-sm">G</span>}
          color="bg-blue-500/15 text-blue-400" bgColor="bg-gray-900/50" borderColor="border-gray-800"
          adoption="A2A native" delay={0.2}
        />
        <IntegrationCard
          name="Devin" role="AI Engineer"
          howItIntegrates="Devin API wrapped as AG2 Actor. AG2 MCP servers available via Devin's MCP support."
          icon={<span className="text-sm">D</span>}
          color="bg-purple-500/15 text-purple-400" bgColor="bg-gray-900/50" borderColor="border-gray-800"
          adoption="API + MCP" delay={0.25}
        />
      </div>

      {/* Key insight */}
      <FadeIn className="mt-16 text-center">
        <div className="inline-block p-6 rounded-2xl bg-gradient-to-r from-blue-500/5 to-violet-500/5 border border-blue-500/10">
          <p className="text-gray-400 text-sm max-w-lg leading-relaxed">
            <span className="text-white font-medium">Two integrations unlock the world.</span>{' '}
            A2A as Channel = cross-framework interop.
            MCP as server = accessible from every AI product.
            AG2 becomes the infrastructure hub connecting them all.
          </p>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
