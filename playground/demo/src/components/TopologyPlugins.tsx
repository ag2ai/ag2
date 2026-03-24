import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

/* -- Plugin duality cards ------------------------------------------------- */

const pluginTypes = [
  {
    type: 'Routing Plugins',
    icon: '\u2192',
    desc: 'Sit in the delegation path. Transform, reject, reroute, or replicate envelopes before the target agent runs.',
    color: 'text-pink-400',
    border: 'border-pink-500/20',
    bg: 'bg-pink-500/5',
    iconBg: 'bg-pink-500/10 border-pink-500/20',
    examples: [
      { name: 'RateLimiter', role: 'Reject delegations exceeding per-sender rate limits' },
      { name: 'LoadBalancer', role: 'Reroute to least-loaded replica' },
      { name: 'RouteMap', role: 'Enforce allowed delegation paths' },
      { name: 'CoRouter', role: 'Replicate delegation to additional targets via RouteDecision' },
    ],
  },
  {
    type: 'System Plugins',
    icon: '\u25C9',
    desc: 'Observe the Hub stream independently. Monitor traffic, track metrics, manage resources. Never in the delegation path.',
    color: 'text-cyan-400',
    border: 'border-cyan-500/20',
    bg: 'bg-cyan-500/5',
    iconBg: 'bg-cyan-500/10 border-cyan-500/20',
    examples: [
      { name: 'TelemetryPlugin', role: 'Track delegation counts, latency, targets' },
      { name: 'AutoScaler', role: 'Scale actors based on queue depth' },
      { name: 'CircuitBreaker', role: 'Disable failing actors after threshold' },
      { name: 'DealTracker', role: 'Domain-specific delegation audit trail' },
    ],
  },
]

/* -- Topology types ------------------------------------------------------- */

const topologyTypes = [
  {
    name: 'Pipeline',
    desc: 'Sequential. Each plugin sees the output of the previous.',
    analogy: 'Like nn.Sequential',
    color: 'text-blue-400',
    border: 'border-blue-500/20',
    bg: 'bg-blue-500/5',
    visual: ['Auth', '\u2192', 'RateLimit', '\u2192', 'Telemetry'],
  },
  {
    name: 'Fanout',
    desc: 'Parallel side-effects. All plugins see copies concurrently.',
    analogy: 'Like port mirroring',
    color: 'text-emerald-400',
    border: 'border-emerald-500/20',
    bg: 'bg-emerald-500/5',
    visual: ['\u251C', 'AuditLog', '\u2502', 'Metrics', '\u2524'],
  },
  {
    name: 'Conditional',
    desc: 'Branching. Route to different topologies based on predicates.',
    analogy: 'Like if/else',
    color: 'text-amber-400',
    border: 'border-amber-500/20',
    bg: 'bg-amber-500/5',
    visual: ['priority?', '\u2192', 'FastTrack', '|', 'Standard'],
  },
]

/* -- RouteDecision patterns ----------------------------------------------- */

const routePatterns = [
  {
    name: 'Forward',
    code: 'return envelope',
    desc: 'Pass through (possibly modified)',
    icon: '\u2192',
  },
  {
    name: 'Reject',
    code: 'return None',
    desc: 'Drop the delegation',
    icon: '\u2715',
  },
  {
    name: 'Multicast',
    code: 'RouteDecision(primary, additional=[...])',
    desc: 'Forward primary + replicate to others',
    icon: '\u21D2',
  },
  {
    name: 'Reject + Notify',
    code: 'RouteDecision(None, additional=[alert])',
    desc: 'Reject but trigger side-effects',
    icon: '\u26A0',
  },
]

/* -- Diagrams ------------------------------------------------------------- */

function TopologyDiagram() {
  return (
    <svg viewBox="0 0 540 220" className="w-full max-w-2xl">
      {/* Pipeline container */}
      <motion.rect
        x="10" y="10" width="520" height="200" rx="12"
        fill="none" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4 4" opacity="0.2"
        initial={{ opacity: 0 }} whileInView={{ opacity: 0.2 }}
        viewport={{ once: true }} transition={{ delay: 0.1 }}
      />
      <motion.text
        x="30" y="32" className="fill-blue-400/40 text-[10px] font-mono font-bold"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.15 }}
      >
        Pipeline
      </motion.text>

      {/* Step 1: Auth plugin */}
      <motion.rect
        x="30" y="55" width="80" height="40" rx="6"
        fill="#ec489920" stroke="#ec4899" strokeWidth="1"
        initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.2, type: 'spring' }}
      />
      <text x="70" y="78" textAnchor="middle" className="fill-pink-400 text-[9px] font-mono font-bold">Auth</text>
      <text x="70" y="110" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">check perms</text>

      {/* Arrow 1-2 */}
      <motion.line
        x1="112" y1="75" x2="138" y2="75"
        stroke="#475569" strokeWidth="1"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.3 }}
      />

      {/* Step 2: CoRouter (multicast) */}
      <motion.rect
        x="140" y="55" width="90" height="40" rx="6"
        fill="#f59e0b20" stroke="#f59e0b" strokeWidth="1.5"
        initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.35, type: 'spring' }}
      />
      <text x="185" y="72" textAnchor="middle" className="fill-amber-400 text-[9px] font-mono font-bold">CoRouter</text>
      <text x="185" y="84" textAnchor="middle" className="fill-amber-400/60 text-[7px] font-mono">RouteDecision</text>

      {/* Additional envelope branching down */}
      <motion.path
        d="M185 95 L185 145 L305 145"
        fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="3 3"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.55, duration: 0.5 }}
      />
      <motion.rect
        x="305" y="130" width="90" height="30" rx="5"
        fill="#f59e0b10" stroke="#f59e0b" strokeWidth="0.8" strokeDasharray="3 3"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.7 }}
      />
      <text x="350" y="149" textAnchor="middle" className="fill-amber-400/70 text-[8px] font-mono">+ audit agent</text>
      <text x="350" y="178" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">additional (fire-and-forget)</text>

      {/* Arrow 2-3 */}
      <motion.line
        x1="232" y1="75" x2="258" y2="75"
        stroke="#475569" strokeWidth="1"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.4 }}
      />

      {/* Step 3: Fanout (side-effects) */}
      <motion.rect
        x="260" y="45" width="110" height="60" rx="6"
        fill="#10b98115" stroke="#10b981" strokeWidth="1"
        initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.45, type: 'spring' }}
      />
      <text x="315" y="62" textAnchor="middle" className="fill-emerald-400/50 text-[7px] font-mono">Fanout</text>
      <text x="295" y="80" textAnchor="middle" className="fill-emerald-400 text-[8px] font-mono">Logger</text>
      <text x="340" y="80" textAnchor="middle" className="fill-emerald-400 text-[8px] font-mono">Metrics</text>
      <text x="315" y="95" textAnchor="middle" className="fill-gray-600 text-[6px] font-mono">parallel side-effects</text>

      {/* Arrow 3-4 */}
      <motion.line
        x1="372" y1="75" x2="398" y2="75"
        stroke="#475569" strokeWidth="1"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.5 }}
      />

      {/* Step 4: RateLimiter */}
      <motion.rect
        x="400" y="55" width="100" height="40" rx="6"
        fill="#ec489920" stroke="#ec4899" strokeWidth="1"
        initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.55, type: 'spring' }}
      />
      <text x="450" y="78" textAnchor="middle" className="fill-pink-400 text-[9px] font-mono font-bold">RateLimiter</text>
      <text x="450" y="110" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">reject if over limit</text>

      {/* Primary flow label */}
      <motion.text
        x="270" y="30" textAnchor="middle" className="fill-gray-500 text-[8px] font-mono"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.6 }}
      >
        primary envelope flows through chain
      </motion.text>

      {/* Flowing envelope on primary path */}
      <motion.rect
        width="8" height="6" rx="1.5" fill="#3b82f6"
        animate={{ x: [25, 500], opacity: [0.9, 0.9, 0.9, 0.9, 0] }}
        transition={{ repeat: Infinity, duration: 4, ease: 'linear' }}
        y="72"
      />
    </svg>
  )
}

/* -- Main section --------------------------------------------------------- */

export function TopologyPlugins() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="topology">
      <SectionLabel number="06" title="Topology & Plugins" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Full routing layer.</span>{' '}
        <span className="text-gray-500">Not just a filter.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        Every delegation flows through a composable plugin pipeline.
        Plugins can <span className="text-pink-400">transform</span>,{' '}
        <span className="text-pink-400">reject</span>,{' '}
        <span className="text-pink-400">reroute</span>, or{' '}
        <span className="text-amber-400">replicate</span> envelopes.
        Topologies compose plugins into{' '}
        <span className="text-white">sequential, parallel, and conditional</span> routing
        patterns that nest freely.
      </p>

      {/* Plugin duality */}
      <div className="grid md:grid-cols-2 gap-6 mb-16">
        {pluginTypes.map((pt, i) => (
          <FadeIn key={pt.type} delay={i * 0.1}>
            <div className={`h-full p-6 rounded-2xl border ${pt.border} ${pt.bg}`}>
              <div className="flex items-center gap-3 mb-4">
                <div className={`w-10 h-10 rounded-lg ${pt.iconBg} border flex items-center justify-center`}>
                  <span className={`${pt.color} font-mono font-bold text-lg`}>{pt.icon}</span>
                </div>
                <div>
                  <h3 className="text-lg font-bold text-white">{pt.type}</h3>
                  <p className="text-xs text-gray-500">{pt.desc}</p>
                </div>
              </div>
              <div className="space-y-2">
                {pt.examples.map(ex => (
                  <div key={ex.name} className="flex items-start gap-3 p-2 rounded-lg bg-gray-900/50 border border-gray-800">
                    <span className={`text-xs font-mono font-bold ${pt.color} bg-gray-800 px-2 py-0.5 rounded shrink-0 mt-0.5`}>
                      {ex.name}
                    </span>
                    <p className="text-xs text-gray-400">{ex.role}</p>
                  </div>
                ))}
              </div>
            </div>
          </FadeIn>
        ))}
      </div>

      {/* Topology types */}
      <FadeIn>
        <h3 className="text-2xl font-bold text-white mb-2">Composable Topologies</h3>
        <p className="text-gray-500 mb-8 max-w-2xl">
          Three composition patterns that nest freely. A Topology is itself usable
          wherever a plugin is expected.
        </p>
      </FadeIn>

      <div className="grid md:grid-cols-3 gap-4 mb-16">
        {topologyTypes.map((tt, i) => (
          <motion.div
            key={tt.name}
            className={`p-5 rounded-xl border ${tt.border} ${tt.bg}`}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 + i * 0.1 }}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className={`text-lg font-bold font-mono ${tt.color}`}>{tt.name}</span>
              <span className="text-[10px] text-gray-600 font-mono">{tt.analogy}</span>
            </div>
            <p className="text-sm text-gray-400 mb-4">{tt.desc}</p>
            <div className="flex items-center gap-1 font-mono text-xs">
              {tt.visual.map((v, j) => (
                <span key={j} className={v.length <= 2 ? 'text-gray-600' : `${tt.color} bg-gray-900/50 px-2 py-0.5 rounded`}>
                  {v}
                </span>
              ))}
            </div>
          </motion.div>
        ))}
      </div>

      {/* RouteDecision */}
      <FadeIn>
        <div className="p-8 md:p-10 rounded-2xl bg-gradient-to-br from-amber-950/20 to-pink-950/20 border border-amber-500/15 mb-16">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-3 py-1 rounded-full border border-amber-500/20">
              NEW
            </span>
            <h3 className="text-2xl font-bold text-white">RouteDecision</h3>
          </div>
          <p className="text-gray-400 mb-8 max-w-2xl">
            The routing primitive that makes the topology a full routing layer.
            Without it, <span className="font-mono text-gray-300">process()</span> is
            single-in, single-out &mdash; a packet filter, not a router.
            RouteDecision lets plugins express{' '}
            <span className="text-amber-400">multicast</span>,{' '}
            <span className="text-amber-400">co-routing</span>,{' '}
            <span className="text-amber-400">broadcast</span>, and{' '}
            <span className="text-amber-400">reject-with-notification</span> patterns.
          </p>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
            {routePatterns.map((rp, i) => (
              <motion.div
                key={rp.name}
                className="p-4 rounded-xl bg-gray-950/60 border border-gray-800"
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 + i * 0.08 }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">{rp.icon}</span>
                  <span className="text-sm font-bold text-white">{rp.name}</span>
                </div>
                <p className="text-xs text-gray-500 mb-2">{rp.desc}</p>
                <code className="text-[10px] font-mono text-amber-400/80 break-all">{rp.code}</code>
              </motion.div>
            ))}
          </div>

          <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800">
            <p className="text-xs text-gray-500 mb-1 font-mono uppercase tracking-wider">Composition rule</p>
            <p className="text-sm text-gray-400 leading-relaxed">
              Additional envelopes <span className="text-amber-400">propagate upward</span> through topology
              composition. Only the primary flows through the pipeline chain. Each additional envelope is dispatched
              as an independent delegation through the full Hub path &mdash; depth tracking, topology, events.
            </p>
          </div>
        </div>
      </FadeIn>

      {/* Visual: full topology pipeline diagram */}
      <FadeIn>
        <h3 className="text-lg font-bold text-white mb-2 text-center">How It Flows</h3>
        <p className="text-sm text-gray-500 text-center mb-6 max-w-lg mx-auto">
          A delegation through a Pipeline with multicast, side-effects, and rate limiting
        </p>
        <div className="flex justify-center">
          <TopologyDiagram />
        </div>
      </FadeIn>

      {/* Key insight */}
      <FadeIn className="mt-16 text-center">
        <div className="inline-flex flex-col items-center gap-2 p-6 rounded-2xl bg-gray-900/50 border border-gray-800">
          <p className="text-gray-400 text-sm max-w-lg leading-relaxed">
            Developers build routing policies as{' '}
            <span className="text-pink-400 font-mono">plugins</span>.
            The framework provides{' '}
            <span className="text-blue-400 font-mono">Pipeline</span>,{' '}
            <span className="text-emerald-400 font-mono">Fanout</span>,{' '}
            <span className="text-amber-400 font-mono">Conditional</span>, and{' '}
            <span className="text-amber-400 font-mono">RouteDecision</span> as building blocks.
            Any routing strategy is expressible &mdash; the framework doesn't prescribe patterns.
          </p>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
