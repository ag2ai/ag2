import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

interface Phase {
  label: string
  title: string
  status: 'done' | 'current' | 'next' | 'future'
  items: string[]
  color: string
  borderColor: string
  bgColor: string
}

const phases: Phase[] = [
  {
    label: 'Phase 1-3',
    title: 'Framework Complete',
    status: 'done',
    items: [
      '15 primitives (Watch, Signal, Priority, Envelope, Channel, KnowledgeStore, CompactStrategy, AggregateStrategy, Infra)',
      '7 building blocks (Actor, Hub, Observer, Scheduler, Assembler, AssemblyPolicies, Network)',
      'Composition layer (Pipeline, Fanout, Conditional)',
      'Agent Harness (persistence, assembly, maintenance)',
      'HttpChannel, topics pub/sub, built-in observers & plugins',
    ],
    color: 'text-emerald-400',
    borderColor: 'border-emerald-500/30',
    bgColor: 'bg-emerald-500/5',
  },
  {
    label: 'Next',
    title: 'End-to-End Examples',
    status: 'current',
    items: [
      'Real agents with real LLM calls',
      'Multi-actor workflow demos',
      'Performance benchmarks',
      'Developer documentation',
      'Internal demo & team alignment',
    ],
    color: 'text-blue-400',
    borderColor: 'border-blue-500/30',
    bgColor: 'bg-blue-500/5',
  },
  {
    label: 'Near-term',
    title: 'Protocol Integration',
    status: 'next',
    items: [
      'A2A Channel backend (cross-framework interop)',
      'MCP server exposure (AG2 Actors as MCP tools)',
      'OpenTelemetry export (Envelope tracing → Jaeger/Grafana)',
    ],
    color: 'text-violet-400',
    borderColor: 'border-violet-500/30',
    bgColor: 'bg-violet-500/5',
  },
  {
    label: 'AG2 Cloud',
    title: 'Distributed Backends',
    status: 'future',
    items: [
      'RedisChannel / NatsChannel (persistent transport)',
      'EtcdRegistry / ConsulRegistry (distributed discovery)',
      'RedisStateStore / PostgresStateStore (persistent state)',
      'Durable scheduler (Temporal/Celery backend)',
      'Hub replication with leader election',
    ],
    color: 'text-amber-400',
    borderColor: 'border-amber-500/30',
    bgColor: 'bg-amber-500/5',
  },
]

const statusConfig = {
  done: { label: 'DONE', bg: 'bg-emerald-500/15', text: 'text-emerald-400', border: 'border-emerald-500/30' },
  current: { label: 'IN PROGRESS', bg: 'bg-blue-500/15', text: 'text-blue-400', border: 'border-blue-500/30' },
  next: { label: 'PLANNED', bg: 'bg-violet-500/15', text: 'text-violet-400', border: 'border-violet-500/30' },
  future: { label: 'AG2 CLOUD', bg: 'bg-amber-500/15', text: 'text-amber-400', border: 'border-amber-500/30' },
}

export function Roadmap() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="roadmap">
      <SectionLabel number="10" title="Roadmap" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Where we're going.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        The framework is functionally complete. Every deferred item is a{' '}
        <span className="text-white">backend swap behind an existing protocol</span> —
        same application code, different infrastructure.
      </p>

      {/* Timeline */}
      <div className="relative">
        {/* Vertical line */}
        <div className="absolute left-[19px] md:left-1/2 top-0 bottom-0 w-px bg-gray-800 md:-translate-x-px" />

        <div className="space-y-8">
          {phases.map((phase, i) => {
            const sc = statusConfig[phase.status]
            const isLeft = i % 2 === 0

            return (
              <motion.div
                key={phase.label}
                initial={{ opacity: 0, x: isLeft ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className={`relative flex items-start gap-6 md:gap-0 ${isLeft ? 'md:flex-row' : 'md:flex-row-reverse'}`}
              >
                {/* Timeline dot */}
                <div className="absolute left-[12px] md:left-1/2 md:-translate-x-1/2 w-4 h-4 rounded-full bg-gray-950 border-2 border-gray-600 z-10 mt-6">
                  {phase.status === 'done' && (
                    <div className="w-full h-full rounded-full bg-emerald-500" />
                  )}
                  {phase.status === 'current' && (
                    <motion.div
                      className="w-full h-full rounded-full bg-blue-500"
                      animate={{ opacity: [0.5, 1, 0.5] }}
                      transition={{ repeat: Infinity, duration: 2 }}
                    />
                  )}
                </div>

                {/* Content */}
                <div className={`ml-10 md:ml-0 md:w-[calc(50%-2rem)] ${isLeft ? 'md:pr-8' : 'md:pl-8'}`}>
                  <div className={`p-5 rounded-xl border ${phase.borderColor} ${phase.bgColor}`}>
                    <div className="flex items-center gap-3 mb-3">
                      <span className={`text-xs font-mono font-bold ${phase.color}`}>
                        {phase.label}
                      </span>
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${sc.bg} ${sc.text} ${sc.border}`}>
                        {sc.label}
                      </span>
                    </div>
                    <h3 className="text-lg font-bold text-white mb-3">{phase.title}</h3>
                    <ul className="space-y-1.5">
                      {phase.items.map(item => (
                        <li key={item} className="flex items-start gap-2 text-sm text-gray-400">
                          <span className={`mt-1.5 w-1.5 h-1.5 rounded-full shrink-0 ${
                            phase.status === 'done' ? 'bg-emerald-500' : 'bg-gray-600'
                          }`} />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Spacer for the other side */}
                <div className="hidden md:block md:w-[calc(50%-2rem)]" />
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* Key principle */}
      <FadeIn className="mt-20">
        <div className="p-8 rounded-2xl bg-gradient-to-r from-blue-950/30 to-violet-950/30 border border-blue-500/10 text-center">
          <h3 className="text-xl font-bold text-white mb-3">The Design Principle</h3>
          <div className="grid sm:grid-cols-3 gap-6 max-w-2xl mx-auto">
            {[
              { from: 'LocalChannel', to: 'RedisChannel', protocol: 'Channel' },
              { from: 'LocalRegistry', to: 'EtcdRegistry', protocol: 'Registry' },
              { from: 'MemoryStateStore', to: 'RedisStateStore', protocol: 'StateStore' },
            ].map(item => (
              <div key={item.protocol} className="text-center">
                <div className="text-xs text-gray-500 font-mono mb-2">{item.protocol}</div>
                <div className="flex items-center justify-center gap-2">
                  <span className="text-xs text-gray-400 font-mono bg-gray-800 px-2 py-1 rounded">
                    {item.from}
                  </span>
                  <svg viewBox="0 0 12 12" className="w-3 h-3 text-blue-400 shrink-0">
                    <path d="M2 6h8M7 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="text-xs text-blue-400 font-mono bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20">
                    {item.to}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-gray-500 text-sm mt-6 max-w-md mx-auto">
            Same application code. Different backends.
            Zero user code changes when cloud backends ship.
          </p>
        </div>
      </FadeIn>

      {/* Closing */}
      <FadeIn className="mt-32 text-center">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
        >
          <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
            <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-cyan-400 bg-clip-text text-transparent">
              The infrastructure layer agents need.
            </span>
          </h2>
          <p className="text-xl text-gray-500 max-w-2xl mx-auto">
            Not more orchestration. Not bare protocols. Not premature standards.
            The network framework that makes autonomous agents production-ready.
          </p>
        </motion.div>
      </FadeIn>
    </AnimatedSection>
  )
}
