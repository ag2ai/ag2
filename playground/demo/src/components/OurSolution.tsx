import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn, ScaleIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

function PositioningDiagram() {
  return (
    <ScaleIn delay={0.2}>
      <div className="relative max-w-2xl mx-auto">
        <svg viewBox="0 0 600 320" className="w-full">
          {/* Y axis label */}
          <text x="10" y="160" textAnchor="middle" className="fill-gray-600 text-[10px] font-mono" transform="rotate(-90, 10, 160)">
            Runtime Capability
          </text>
          {/* X axis label */}
          <text x="320" y="310" textAnchor="middle" className="fill-gray-600 text-[10px] font-mono">
            Distribution Capability
          </text>

          {/* Grid */}
          <line x1="60" y1="20" x2="60" y2="280" stroke="#1e293b" strokeWidth="1" />
          <line x1="60" y1="280" x2="580" y2="280" stroke="#1e293b" strokeWidth="1" />

          {/* Orchestration camp */}
          <motion.g initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 0.3 }}>
            <rect x="100" y="40" width="160" height="100" rx="8" fill="#f97316" fillOpacity="0.06" stroke="#f97316" strokeWidth="1" strokeOpacity="0.2" />
            <text x="180" y="80" textAnchor="middle" className="fill-orange-400 text-[12px] font-semibold">Orchestration</text>
            <text x="180" y="100" textAnchor="middle" className="fill-gray-500 text-[9px] font-mono">LangGraph, CrewAI</text>
            <text x="180" y="116" textAnchor="middle" className="fill-gray-500 text-[9px] font-mono">OpenAI Agents SDK</text>
          </motion.g>

          {/* Protocol camp */}
          <motion.g initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 0.5 }}>
            <rect x="320" y="180" width="140" height="80" rx="8" fill="#3b82f6" fillOpacity="0.06" stroke="#3b82f6" strokeWidth="1" strokeOpacity="0.2" />
            <text x="390" y="215" textAnchor="middle" className="fill-blue-400 text-[12px] font-semibold">Protocols</text>
            <text x="390" y="235" textAnchor="middle" className="fill-gray-500 text-[9px] font-mono">A2A, MCP</text>
          </motion.g>

          {/* AGNTCY */}
          <motion.g initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 0.7 }}>
            <rect x="460" y="160" width="100" height="80" rx="8" fill="#a855f7" fillOpacity="0.06" stroke="#a855f7" strokeWidth="1" strokeOpacity="0.2" />
            <text x="510" y="195" textAnchor="middle" className="fill-purple-400 text-[12px] font-semibold">AGNTCY</text>
            <text x="510" y="215" textAnchor="middle" className="fill-gray-500 text-[9px] font-mono">Infra only</text>
          </motion.g>

          {/* AG2 V2 */}
          <motion.g initial={{ opacity: 0, scale: 0 }} whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 1.0, type: 'spring', stiffness: 200 }}>
            <rect x="300" y="30" width="200" height="110" rx="12"
              fill="url(#ag2gradient)" fillOpacity="0.12"
              stroke="url(#ag2gradient)" strokeWidth="2" strokeOpacity="0.5" />
            <text x="400" y="65" textAnchor="middle" className="fill-white text-[16px] font-bold">AG2 V2</text>
            <text x="400" y="85" textAnchor="middle" className="fill-gray-400 text-[10px] font-mono">Framework + Infrastructure</text>
            <text x="400" y="105" textAnchor="middle" className="fill-gray-400 text-[10px] font-mono">Runtime + Distribution</text>
            <text x="400" y="125" textAnchor="middle" className="fill-blue-400 text-[9px] font-mono">Protocol-agnostic</text>
          </motion.g>

          <defs>
            <linearGradient id="ag2gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="100%" stopColor="#8b5cf6" />
            </linearGradient>
          </defs>
        </svg>
      </div>
    </ScaleIn>
  )
}

function StackLayerDiagram() {
  const layers = [
    { label: 'Your Application', sub: 'Agent workflows, business logic', color: '#8b5cf6', highlight: false },
    { label: 'AG2 V2 Network', sub: 'Routing + State + Scheduling + Observability', color: '#3b82f6', highlight: true },
    { label: 'Wire Protocols', sub: 'A2A (agent-to-agent) + MCP (agent-to-tool)', color: '#06b6d4', highlight: false },
    { label: 'Transport', sub: 'HTTP, WebSocket, gRPC, Redis, NATS', color: '#10b981', highlight: false },
  ]

  return (
    <div className="max-w-md mx-auto mt-10">
      <div className="space-y-1.5">
        {layers.map((layer, i) => (
          <motion.div
            key={layer.label}
            initial={{ opacity: 0, scale: 0.9, y: 15 }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 + i * 0.12, duration: 0.5 }}
            className={`relative px-5 py-3 rounded-lg text-center border ${
              layer.highlight
                ? 'border-blue-400/40 bg-blue-500/10 ring-1 ring-blue-400/20'
                : 'border-gray-700/50 bg-gray-900/50'
            }`}
          >
            <div className={`font-bold font-mono text-sm ${layer.highlight ? 'text-white' : 'text-gray-400'}`}>
              {layer.label}
            </div>
            <div className="text-[10px] text-gray-500 font-mono mt-0.5">{layer.sub}</div>
            {layer.highlight && (
              <motion.div
                className="absolute -left-2 top-1/2 -translate-y-1/2 w-1 h-8 rounded-full bg-blue-500"
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity, duration: 2 }}
              />
            )}
          </motion.div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-600 mt-3 font-mono">
        AG2 V2 is the missing runtime layer in this stack
      </p>
    </div>
  )
}

export function OurSolution() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="solution">
      <SectionLabel number="03" title="Our Answer" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
          The operating system for agent networks.
        </span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        AG2 V2 is to AI agents what <span className="text-white">TCP/IP + HTTP was to computers</span>:
        the infrastructure layer that turns isolated machines into a connected network.
        We build the runtime between wire protocols and application code — so any agent,
        from any framework, can discover, communicate, and coordinate with any other.
      </p>

      <PositioningDiagram />
      <StackLayerDiagram />

      {/* Five capabilities — the real differentiators */}
      <div className="mt-24">
        <FadeIn>
          <h3 className="text-2xl md:text-3xl font-bold text-white text-center mb-4">
            What AG2 V2 actually delivers
          </h3>
          <p className="text-gray-500 text-center max-w-2xl mx-auto mb-12">
            Not just another framework. A fundamentally different architectural layer
            that no one else is building.
          </p>
        </FadeIn>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            {
              title: 'Network-Native Agents',
              desc: 'Agents are born networked. An Actor registers capabilities with a Hub, gets discovered by peers, and communicates through typed channels — no hardcoded connections. Like DNS + HTTP for agents.',
              gradient: 'from-cyan-500 to-blue-500',
              bgHover: 'hover:border-cyan-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <circle cx="16" cy="16" r="6" stroke="#06b6d4" strokeWidth="1.5" />
                  <circle cx="16" cy="16" r="12" stroke="#06b6d4" strokeWidth="1" strokeDasharray="3 3" opacity="0.4" />
                  <circle cx="6" cy="10" r="3" fill="#06b6d420" stroke="#06b6d4" strokeWidth="1" />
                  <circle cx="26" cy="10" r="3" fill="#06b6d420" stroke="#06b6d4" strokeWidth="1" />
                  <circle cx="16" cy="28" r="3" fill="#06b6d420" stroke="#06b6d4" strokeWidth="1" />
                </svg>
              ),
            },
            {
              title: 'Controlled Choreography',
              desc: 'Neither rigid orchestration nor uncontrolled peer-to-peer. The Hub sets policies (topology, rate limits, auth) — Actors operate autonomously within them. Like BGP: set the rules, let nodes route.',
              gradient: 'from-blue-500 to-violet-500',
              bgHover: 'hover:border-blue-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <rect x="8" y="8" width="16" height="16" rx="3" stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="3 3" opacity="0.4" />
                  <circle cx="10" cy="10" r="3" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1" />
                  <circle cx="22" cy="10" r="3" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1" />
                  <circle cx="10" cy="22" r="3" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1" />
                  <circle cx="22" cy="22" r="3" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1" />
                  <circle cx="16" cy="16" r="4" fill="#3b82f630" stroke="#3b82f6" strokeWidth="1.5" />
                </svg>
              ),
            },
            {
              title: 'Run Anywhere, Same Code',
              desc: 'Every infrastructure concern is a swappable protocol. LocalChannel in dev, RedisChannel in production. MemoryStore to PostgresStore. Zero code changes — just swap the backend.',
              gradient: 'from-violet-500 to-purple-500',
              bgHover: 'hover:border-violet-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <rect x="4" y="6" width="10" height="10" rx="2" stroke="#8b5cf6" strokeWidth="1.5" />
                  <rect x="18" y="6" width="10" height="10" rx="2" stroke="#8b5cf6" strokeWidth="1.5" />
                  <rect x="11" y="18" width="10" height="10" rx="2" stroke="#8b5cf6" strokeWidth="1.5" />
                  <path d="M14 11h4M9 16v2M23 16v5h-2" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
                </svg>
              ),
            },
            {
              title: 'Full-Stack Observability',
              desc: 'Every delegation is traced end-to-end. Every message carries context (trace ID, causation chain, priority). Observers monitor in real-time and can steer agents through signals — not just log them.',
              gradient: 'from-emerald-500 to-teal-500',
              bgHover: 'hover:border-emerald-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <circle cx="16" cy="16" r="5" fill="none" stroke="#10b981" strokeWidth="1.5" />
                  <circle cx="16" cy="16" r="2" fill="#10b981" />
                  <circle cx="16" cy="16" r="10" fill="none" stroke="#10b981" strokeWidth="0.8" opacity="0.3" />
                  <path d="M16 2v4M16 26v4M2 16h4M26 16h4" stroke="#10b981" strokeWidth="1" opacity="0.4" strokeLinecap="round" />
                </svg>
              ),
            },
            {
              title: 'Universal Integration',
              desc: 'A2A plugs in as a Channel. MCP remains agent-to-tool. AGNTCY provides lower-level infra. AG2 is complementary, not competing — the layer above that ties everything together.',
              gradient: 'from-amber-500 to-orange-500',
              bgHover: 'hover:border-amber-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <path d="M8 8h6v6H8zM18 8h6v6h-6zM8 18h6v6H8zM18 18h6v6h-6z" stroke="#f59e0b" strokeWidth="1.5" rx="1" />
                  <path d="M14 11h4M14 21h4M11 14v4M25 14v4" stroke="#f59e0b" strokeWidth="1" opacity="0.5" />
                </svg>
              ),
            },
            {
              title: 'AI-First Design',
              desc: 'Coding agents are primary consumers. Complete type hints, self-documenting errors, introspectable state, convention-based defaults. The API is designed to be read by machines, not just humans.',
              gradient: 'from-pink-500 to-rose-500',
              bgHover: 'hover:border-pink-500/30',
              icon: (
                <svg viewBox="0 0 32 32" fill="none" className="w-8 h-8">
                  <rect x="4" y="6" width="24" height="20" rx="3" stroke="#ec4899" strokeWidth="1.5" />
                  <path d="M10 14l3 3-3 3M16 20h6" stroke="#ec4899" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ),
            },
          ].map((item, i) => (
            <FadeIn key={item.title} delay={0.08 * i}>
              <div className={`p-6 rounded-2xl bg-gray-900/50 border border-gray-800 h-full transition-colors duration-300 ${item.bgHover}`}>
                <div className="mb-4">{item.icon}</div>
                <h4 className="text-lg font-bold text-white mb-2">{item.title}</h4>
                <p className="text-sm text-gray-400 leading-relaxed">{item.desc}</p>
              </div>
            </FadeIn>
          ))}
        </div>
      </div>

      {/* OSS to Cloud bridge */}
      <FadeIn className="mt-16">
        <div className="p-6 md:p-8 rounded-2xl bg-gradient-to-r from-blue-950/30 to-violet-950/30 border border-blue-500/10">
          <div className="flex flex-col md:flex-row items-center gap-6 md:gap-10">
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="w-16 h-16 rounded-xl bg-gray-800 border border-gray-700 flex items-center justify-center mb-2">
                  <span className="text-2xl font-bold text-white">OSS</span>
                </div>
                <span className="text-xs text-gray-500">In-memory defaults</span>
              </div>
              <motion.div
                className="flex items-center"
                initial={{ width: 0, opacity: 0 }}
                whileInView={{ width: 'auto', opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.5, duration: 0.5 }}
              >
                <div className="w-24 h-px bg-gradient-to-r from-gray-600 to-blue-500" />
                <svg viewBox="0 0 12 12" className="w-3 h-3 text-blue-400 -ml-1">
                  <path d="M2 6h8M7 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </motion.div>
              <div className="text-center">
                <div className="w-16 h-16 rounded-xl bg-gradient-to-br from-blue-600 to-violet-600 flex items-center justify-center mb-2 shadow-lg shadow-blue-500/20">
                  <span className="text-lg font-bold text-white">Cloud</span>
                </div>
                <span className="text-xs text-gray-500">Swap backends</span>
              </div>
            </div>

            <div className="flex-1 md:pl-6 md:border-l border-gray-800">
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-white font-medium">Same application code, different backends.</span>{' '}
                Every infrastructure concern — Channel, Registry, StateStore, Cache, Lock — is a protocol
                with in-memory defaults. AG2 Cloud swaps in distributed backends. Zero code changes.
              </p>
            </div>
          </div>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
