import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

/* ── Animated failure diagrams for each camp ──────────────────────────── */

function OrchestrationFailure() {
  // Show exponential complexity: adding agents creates n^2 connections through orchestrator
  const agents = [
    { x: 50, y: 130 }, { x: 130, y: 140 }, { x: 210, y: 130 },
    { x: 90, y: 190 }, { x: 170, y: 195 },
  ]
  const orchX = 130

  return (
    <svg viewBox="0 0 260 220" className="w-full max-w-[260px] mx-auto">
      {/* Orchestrator — overloaded */}
      <motion.rect x="80" y="30" width="100" height="40" rx="6"
        fill="#1e293b" stroke="#f97316" strokeWidth="1.5"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x={orchX} y="55" textAnchor="middle" className="fill-orange-400 text-[10px] font-mono font-bold">Orchestrator</text>

      {/* Stress pulse on orchestrator */}
      <motion.circle cx={orchX} cy="50" r="4" fill="#ef4444"
        animate={{ opacity: [0.3, 0.9, 0.3], scale: [1, 1.5, 1] }}
        transition={{ repeat: Infinity, duration: 1.5, ease: 'easeInOut' }} />

      {/* All lines funnel through orchestrator */}
      {agents.map((agent, i) => (
        <g key={i}>
          <motion.line x1={orchX} y1="70" x2={agent.x} y2={agent.y}
            stroke="#f97316" strokeWidth="1" strokeDasharray="3 3"
            initial={{ pathLength: 0, opacity: 0 }}
            whileInView={{ pathLength: 1, opacity: 0.4 }}
            viewport={{ once: true }} transition={{ delay: 0.3 + i * 0.1 }} />
          <motion.g animate={{ x: [0, 1.5, -1.5, 0] }}
            transition={{ repeat: Infinity, duration: 2, delay: i * 0.4, ease: 'easeInOut' }}>
            <rect x={agent.x - 25} y={agent.y} width="50" height="28" rx="5"
              fill="#1e293b" stroke="#475569" strokeWidth="1" />
            <text x={agent.x} y={agent.y + 18} textAnchor="middle" className="fill-gray-400 text-[8px] font-mono">
              Agent {i + 1}
            </text>
          </motion.g>
        </g>
      ))}

      {/* Bottleneck label */}
      <motion.text x={orchX} y="18" textAnchor="middle"
        className="fill-red-400 text-[8px] font-mono"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ repeat: Infinity, duration: 2 }}>
        BOTTLENECK
      </motion.text>
    </svg>
  )
}

function ProtocolFailure() {
  // Show messages flying between agents but no coordination — some lost, some colliding
  const agents = [
    { x: 40, y: 60, label: 'A' },
    { x: 220, y: 60, label: 'B' },
    { x: 130, y: 160, label: 'C' },
  ]

  return (
    <svg viewBox="0 0 260 200" className="w-full max-w-[260px] mx-auto">
      {/* Agents */}
      {agents.map((a, i) => (
        <motion.g key={a.label}
          initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }} transition={{ delay: 0.2 + i * 0.1 }}>
          <circle cx={a.x} cy={a.y} r="20" fill="#3b82f610" stroke="#3b82f6" strokeWidth="1.2" />
          <text x={a.x} y={a.y + 4} textAnchor="middle" className="fill-blue-400 text-[10px] font-mono font-bold">{a.label}</text>
        </motion.g>
      ))}

      {/* Messages flying — some arrive, some fade out (lost) */}
      {[
        { x1: 60, y1: 60, x2: 200, y2: 60, color: '#3b82f6', lost: false },
        { x1: 200, y1: 70, x2: 60, y2: 70, color: '#3b82f6', lost: true },
        { x1: 60, y1: 75, x2: 115, y2: 145, color: '#06b6d4', lost: false },
        { x1: 145, y1: 145, x2: 200, y2: 75, color: '#06b6d4', lost: true },
      ].map((msg, i) => (
        <motion.circle key={i} r="3" fill={msg.color}
          animate={{
            cx: [msg.x1, msg.x2],
            cy: [msg.y1, msg.y2],
            opacity: msg.lost ? [0.8, 0.8, 0] : [0.8, 0.8, 0.8, 0],
          }}
          transition={{ repeat: Infinity, duration: msg.lost ? 1.5 : 2, delay: i * 0.6, ease: 'linear' }} />
      ))}

      {/* Question marks — no state, no guarantees */}
      {[{ x: 130, y: 50 }, { x: 80, y: 120 }, { x: 180, y: 120 }].map((p, i) => (
        <motion.text key={i} x={p.x} y={p.y} textAnchor="middle"
          className="fill-yellow-400/60 text-[14px] font-bold"
          animate={{ opacity: [0, 0.6, 0], y: [p.y, p.y - 8, p.y] }}
          transition={{ repeat: Infinity, duration: 2.5, delay: i * 0.8 }}>
          ?
        </motion.text>
      ))}

      {/* Labels */}
      <text x="130" y="195" textAnchor="middle" className="fill-gray-600 text-[8px] font-mono">
        No state. No delivery guarantees. No visibility.
      </text>
    </svg>
  )
}

function InfrastructureFailure() {
  // Show massive spec stack with no runtime
  const specs = ['OASF', 'SLIM', 'ACP', 'Directory', 'Identity', 'Observe']

  return (
    <svg viewBox="0 0 260 200" className="w-full max-w-[260px] mx-auto">
      {/* Stack of spec boxes */}
      {specs.map((spec, i) => (
        <motion.g key={spec}
          initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }} transition={{ delay: 0.15 + i * 0.08 }}>
          <rect x="60" y={20 + i * 26} width="140" height="22" rx="4"
            fill="#a855f710" stroke="#a855f7" strokeWidth="0.8" />
          <text x="130" y={35 + i * 26} textAnchor="middle" className="fill-purple-400 text-[8px] font-mono">{spec}</text>
        </motion.g>
      ))}

      {/* Empty box on top — "Your runtime here?" */}
      <motion.g
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.8 }}>
        <rect x="40" y="175" width="180" height="24" rx="6"
          fill="none" stroke="#ef4444" strokeWidth="1.5" strokeDasharray="4 4" />
        <text x="130" y="191" textAnchor="middle" className="fill-red-400 text-[9px] font-mono font-bold">
          Runtime? Scheduling? Coordination?
        </text>
      </motion.g>

      {/* Arrow pointing to gap */}
      <motion.path d="M225 187 L205 187" stroke="#ef4444" strokeWidth="1.5" markerEnd="url(#arrowRed)"
        animate={{ opacity: [0.4, 1, 0.4] }} transition={{ repeat: Infinity, duration: 1.5 }} />

      <defs>
        <marker id="arrowRed" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#ef4444" />
        </marker>
      </defs>
    </svg>
  )
}

/* ── Camp Card ────────────────────────────────────────────────────────── */

interface CampCardProps {
  title: string
  subtitle: string
  examples: string
  analogy: string
  icon: React.ReactNode
  failure: React.ReactNode
  color: string
  borderColor: string
  bgColor: string
  problems: { title: string; desc: string }[]
  verdict: string
  verdictColor: string
  delay: number
}

function CampCard({
  title, subtitle, examples, analogy, icon, failure, color, borderColor, bgColor,
  problems, verdict, verdictColor, delay
}: CampCardProps) {
  return (
    <FadeIn delay={delay}>
      <div className={`relative h-full p-6 rounded-2xl border ${borderColor} ${bgColor} flex flex-col`}>
        <div className={`w-12 h-12 rounded-xl ${color} flex items-center justify-center mb-4`}>
          {icon}
        </div>
        <h3 className="text-xl font-bold text-white mb-1">{title}</h3>
        <p className="text-sm text-gray-400 mb-1">{subtitle}</p>
        <p className="text-xs text-gray-600 font-mono mb-3">{examples}</p>

        {/* Animated failure visualization */}
        <div className="mb-5 py-3 px-2 rounded-xl bg-gray-950/50">
          {failure}
        </div>

        {/* Real-world analogy */}
        <div className="mb-5 p-3 rounded-lg bg-gray-800/30 border border-gray-800">
          <p className="text-xs text-gray-500 italic leading-relaxed">
            <span className="text-gray-400 not-italic font-medium">Analogy: </span>
            {analogy}
          </p>
        </div>

        <div className="space-y-3 flex-1">
          {problems.map((problem, i) => (
            <div key={i} className="flex items-start gap-2.5">
              <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-red-400/60">
                <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.2" />
                <path d="M5.5 5.5l5 5M10.5 5.5l-5 5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
              </svg>
              <div>
                <p className="text-sm text-white font-medium leading-snug">{problem.title}</p>
                <p className="text-xs text-gray-500 mt-0.5">{problem.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div className={`mt-6 pt-4 border-t border-gray-800 text-sm font-bold ${verdictColor}`}>
          {verdict}
        </div>
      </div>
    </FadeIn>
  )
}

export function MarketSolutions() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="market">
      <SectionLabel number="02" title="Landscape" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Three camps.</span>{' '}
        <span className="text-gray-500">All insufficient.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        The industry has converged on three approaches to multi-agent systems.
        Each solves part of the puzzle — but each has a{' '}
        <span className="text-white">structural limitation</span> that
        prevents it from becoming the foundation agents need.
      </p>

      <div className="grid md:grid-cols-3 gap-6 mb-4">
        <CampCard
          title="Orchestration"
          subtitle="Central brain controls all agents"
          examples="LangGraph, CrewAI, OpenAI Agents SDK"
          analogy="Like a phone switchboard operator manually connecting every call. Works at 10 calls. Collapses at 10,000."
          icon={
            <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6 text-orange-400">
              <circle cx="12" cy="6" r="3" stroke="currentColor" strokeWidth="1.5" />
              <path d="M12 9v3m-6 3a3 3 0 100 6m12-6a3 3 0 100 6M6 15l6-3 6 3" stroke="currentColor" strokeWidth="1.5" />
            </svg>
          }
          failure={<OrchestrationFailure />}
          color="bg-orange-500/10"
          borderColor="border-orange-500/15"
          bgColor="bg-orange-950/10"
          problems={[
            {
              title: 'Single-process, single-machine',
              desc: 'All agents must live in one deployment. No cross-framework, no cross-machine collaboration.',
            },
            {
              title: 'Exponential complexity',
              desc: 'Adding each agent multiplies handoff logic. 5 agents = manageable. 50 = unmaintainable.',
            },
            {
              title: 'Wrong model for scale',
              desc: 'The internet doesn\'t orchestrate computers. Cell networks don\'t route every call through one tower. Central control is structurally unable to scale.',
            },
          ]}
          verdict="Doesn't scale. By design."
          verdictColor="text-orange-400"
          delay={0.1}
        />

        <CampCard
          title="Protocols"
          subtitle="Wire formats for message passing"
          examples="A2A (Google), MCP (Anthropic)"
          analogy="Like defining envelope sizes and postal codes — essential for mail delivery, but someone still needs to build the post office, hire carriers, and track packages."
          icon={
            <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6 text-blue-400">
              <path d="M4 12h16M4 6h16M4 18h16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          }
          failure={<ProtocolFailure />}
          color="bg-blue-500/10"
          borderColor="border-blue-500/15"
          bgColor="bg-blue-950/10"
          problems={[
            {
              title: 'No coordination layer',
              desc: 'A2A defines how agents talk, not how they coordinate. State, scheduling, routing — all left to the implementer.',
            },
            {
              title: 'Vertical vs. horizontal',
              desc: 'MCP connects agents to tools (vertical) but provides no agent-to-agent coordination (horizontal).',
            },
            {
              title: 'Fire-and-forget by default',
              desc: 'No delivery guarantees, no state consistency, no traffic control. Every production feature is built ad-hoc behind the server.',
            },
          ]}
          verdict="Necessary. But not sufficient."
          verdictColor="text-blue-400"
          delay={0.2}
        />

        <CampCard
          title="Infrastructure"
          subtitle="Top-down standards stack"
          examples="AGNTCY (Linux Foundation)"
          analogy="Like designing the entire interstate highway system, including quantum-proof tollbooths, before anyone has built a car that can use it."
          icon={
            <svg viewBox="0 0 24 24" fill="none" className="w-6 h-6 text-purple-400">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          }
          failure={<InfrastructureFailure />}
          color="bg-purple-500/10"
          borderColor="border-purple-500/15"
          bgColor="bg-purple-950/10"
          problems={[
            {
              title: 'Premature complexity',
              desc: '36 repos, 6+ specs, 4 languages, quantum-safe crypto — zero known production deployment. Solving 2030 problems with 2030 complexity today.',
            },
            {
              title: 'Infrastructure without runtime',
              desc: 'Agents can discover each other — but no task coordination, no execution loop, no scheduling, no developer experience.',
            },
            {
              title: 'The WS-* pattern',
              desc: 'Top-down specs from enterprise vendors (Cisco, Dell, Oracle). History: CORBA lost to REST. WS-* lost to JSON. Grassroots wins.',
            },
          ]}
          verdict="Right diagnosis. Wrong treatment."
          verdictColor="text-purple-400"
          delay={0.3}
        />
      </div>

      {/* Gap visualization */}
      <FadeIn className="mt-16">
        <div className="text-center">
          <motion.div
            className="inline-flex items-center gap-3 px-6 py-4 rounded-2xl bg-gradient-to-r from-blue-500/10 to-violet-500/10 border border-blue-500/20"
            whileInView={{ scale: [0.95, 1] }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <span className="text-lg md:text-xl font-bold bg-gradient-to-r from-blue-400 to-violet-400 bg-clip-text text-transparent">
              What's missing is the infrastructure layer for agents.
            </span>
          </motion.div>
          <p className="text-gray-500 mt-4 text-lg max-w-2xl mx-auto">
            Not more orchestration. Not bare wire protocols. Not premature standards.
            The <span className="text-gray-300">runtime layer</span> between the protocol and the application —
            the nervous system that lets autonomous agents discover, communicate, and coordinate at scale.
          </p>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
