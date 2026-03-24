import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

/* ── Theme deep-dive sections ─────────────────────────────────────────── */

function NetworkTheme() {
  return (
    <FadeIn>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-cyan-950/20 to-blue-950/20 border border-cyan-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
            <span className="text-cyan-400 font-mono font-bold text-lg">~</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Network</h3>
            <p className="text-xs text-gray-500">How agents find and talk to each other</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {/* Left: What we built */}
          <div>
            <h4 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Today, connecting two agents means hardcoding URLs, writing custom HTTP clients,
              and hoping the other side speaks the same protocol. There's no standard way to say
              "I need an agent that can do X" and have one show up.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              AG2 V2 builds a complete networking stack: agents register capabilities with a{' '}
              <span className="text-cyan-400 font-mono">Hub</span>, get discovered dynamically,
              and communicate through typed{' '}
              <span className="text-cyan-400 font-mono">Channels</span> wrapped in{' '}
              <span className="text-cyan-400 font-mono">Envelopes</span> that carry addressing,
              tracing, and delivery metadata.
            </p>

            <div className="space-y-3">
              {[
                { component: 'Hub', role: 'Discovery, routing, and plugin pipeline — the network center' },
                { component: 'Channel', role: 'Transport abstraction — Local, HTTP, Buffered, Priority (future: Redis, NATS, A2A)' },
                { component: 'Envelope', role: 'Wire format with addressing, tracing IDs, priority, TTL, and delivery requirements' },
                { component: 'Topology', role: 'Composable routing policies — Pipeline, Fanout, Conditional — that nest freely' },
              ].map(item => (
                <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                  <span className="text-xs font-mono font-bold text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                  <p className="text-xs text-gray-400">{item.role}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Why it matters + example */}
          <div>
            <h4 className="text-sm font-semibold text-cyan-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Dynamic discovery', detail: 'Agents find each other by capability, not by hardcoded address. Add a new agent to the network — existing agents discover it automatically.' },
                { benefit: 'Transport-agnostic', detail: 'Same code works in-process (LocalChannel), across machines (HttpChannel), or across organizations (future A2AChannel). Swap the transport, not the logic.' },
                { benefit: 'Full traffic control', detail: 'Every delegation flows through a composable plugin pipeline — auth, rate limiting, telemetry, approval gates. Infrastructure concerns are centralized, not scattered across agents.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-cyan-400">
                    <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-white">{item.benefit}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Example reference */}
            <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800">
              <p className="text-xs text-gray-500 mb-2 font-mono uppercase tracking-wider">In Practice</p>
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-white font-medium">Emergency Dispatch:</span> A dispatch agent says
                "I need medical + police capabilities." The Hub discovers EMS, Hospital, and Police actors —
                whether they're in the same process or running as separate services across machines via HttpChannel.
                Delegation chains flow automatically: Dispatch → EMS → Hospital, with full telemetry at every hop.
              </p>
            </div>
          </div>
        </div>

        {/* Network visualization */}
        <div className="mt-8 flex justify-center">
          <NetworkFlowDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

function DistributionTheme() {
  return (
    <FadeIn delay={0.1}>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-amber-950/20 to-orange-950/20 border border-amber-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
            <span className="text-amber-400 font-mono font-bold text-lg">::</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Distribution</h3>
            <p className="text-xs text-gray-500">How agents work across processes, machines, and organizations</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Every existing framework assumes agents live in the same Python process.
              But production systems span machines, regions, and organizations. Moving from
              a prototype to distributed deployment currently requires a complete rewrite.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              AG2 V2 is distributed from day one. Every component is backed by a{' '}
              <span className="text-amber-400">swappable protocol</span>. Development uses in-memory defaults.
              Production swaps in distributed backends. The architecture follows the
              same pattern everywhere: <span className="text-white">protocol → in-memory default → cloud backend</span>.
            </p>

            <div className="space-y-3">
              {[
                { from: 'LocalChannel', to: 'HttpChannel → Redis/NATS', protocol: 'Channel' },
                { from: 'MemoryRegistry', to: 'Etcd/Consul', protocol: 'Registry' },
                { from: 'MemoryStateStore', to: 'Redis/Postgres', protocol: 'StateStore' },
                { from: 'In-memory Lock', to: 'Distributed Lock', protocol: 'Lock' },
              ].map(item => (
                <div key={item.protocol} className="flex items-center gap-2 p-2 rounded-lg bg-gray-900/50 border border-gray-800">
                  <span className="text-[10px] font-mono text-gray-400 bg-gray-800 px-2 py-0.5 rounded shrink-0">{item.from}</span>
                  <svg viewBox="0 0 12 12" className="w-3 h-3 text-amber-400 shrink-0">
                    <path d="M2 6h8M7 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded shrink-0">{item.to}</span>
                  <span className="text-[9px] text-gray-600 font-mono ml-auto">{item.protocol}</span>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Prototype to production without rewrite', detail: 'Build locally with in-memory everything. Deploy distributed by swapping configuration. Your agent logic never changes.' },
                { benefit: 'Cross-process and cross-machine', detail: 'Hub.serve() exposes a network endpoint. Hub.connect() discovers remote agents. RemoteAgent proxies make remote agents look local.' },
                { benefit: 'Infrastructure protocols define the boundary', detail: 'StateStore, Cache, Lock, Registry — each is a protocol with clear semantics. AG2 Cloud provides production backends. Third parties can implement their own.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-amber-400">
                    <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-white">{item.benefit}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800">
              <p className="text-xs text-gray-500 mb-2 font-mono uppercase tracking-wider">In Practice</p>
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-white font-medium">Smart Building:</span> HVAC and Energy agents run on a climate server.
                Security and Maintenance run on an operations server. A controller connects to both via{' '}
                <span className="text-amber-400 font-mono">Hub.connect()</span>. The scheduler triggers autonomous health checks
                every 10 seconds. Same code runs in-process for testing or distributed across buildings in production.
              </p>
            </div>
          </div>
        </div>

        {/* Distribution visualization */}
        <div className="mt-8 flex justify-center">
          <DistributedDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

function AutonomyTheme() {
  return (
    <FadeIn delay={0.2}>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-emerald-950/20 to-teal-950/20 border border-emerald-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center">
            <span className="text-emerald-400 font-mono font-bold text-lg">*</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Autonomy</h3>
            <p className="text-xs text-gray-500">How agents act independently, react to events, and self-correct</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Most agent frameworks are request-response: a human sends a message, the agent replies.
              But real-world agents need to operate autonomously — monitoring systems, reacting to events,
              running on schedules, and self-correcting when things go wrong.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              AG2 V2 gives agents a nervous system. <span className="text-emerald-400 font-mono">Observers</span> watch
              the event stream using composable <span className="text-emerald-400 font-mono">Watches</span>,
              emit <span className="text-emerald-400 font-mono">Signals</span> that reshape agent behavior in real-time,
              and the <span className="text-emerald-400 font-mono">Scheduler</span> drives time-based autonomy.
            </p>

            <div className="space-y-3">
              {[
                { component: 'Observer', role: 'Monitors the stream, emits signals. Built-in: TokenMonitor, LoopDetector, ContextHarness' },
                { component: 'Watch', role: 'Unified reactive conditions — event, batch, interval, cron, delay + composites (AllOf, AnyOf, Sequence)' },
                { component: 'Signal', role: 'Structured notifications: INFO/WARNING/CRITICAL (advisory) and FATAL (mechanical halt)' },
                { component: 'Scheduler', role: 'Manages Watch lifecycle — arms, disarms, deduplicates. Drives autonomous behavior loops' },
              ].map(item => (
                <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                  <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                  <p className="text-xs text-gray-400">{item.role}</p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-emerald-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Agents that operate without humans', detail: 'Scheduler triggers agents on intervals or cron. An SRE bot checks system health every 10 seconds. A building manager runs energy audits hourly. No human in the loop.' },
                { benefit: 'Real-time guardrails', detail: 'Observers monitor token spend, detect loops, enforce content policies — and steer agents via signals. A FATAL signal mechanically halts execution. Safety is architectural, not bolted-on.' },
                { benefit: 'Self-aware agents', detail: 'Signals give agents awareness of their environment. A budget monitor warns at 80% spend. A loop detector breaks infinite delegation chains. Agents adapt their behavior based on system state.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-emerald-400">
                    <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    <path d="M6 8l2 2 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-white">{item.benefit}</p>
                    <p className="text-xs text-gray-500 mt-0.5">{item.detail}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800">
              <p className="text-xs text-gray-500 mb-2 font-mono uppercase tracking-wider">In Practice</p>
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-white font-medium">Safety Harness:</span> A research agent operates with multiple
                observers watching for budget overruns, repetitive behavior, and content policy violations.
                When the budget monitor detects 90% spend, it emits a CRITICAL signal. The agent sees the warning
                and wraps up. At 100%, a FATAL signal mechanically halts execution — no LLM decision needed.
              </p>
            </div>
          </div>
        </div>

        {/* Autonomy visualization */}
        <div className="mt-8 flex justify-center">
          <AutonomyFlowDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Animated diagrams ────────────────────────────────────────────────── */

function NetworkFlowDiagram() {
  return (
    <svg viewBox="0 0 500 120" className="w-full max-w-xl">
      {/* Flow: Actor A → Hub (plugin pipeline) → Channel → Actor B */}
      {[
        { x: 40, w: 70, label: 'Actor A', sub: 'delegate_to("B")', color: '#8b5cf6' },
        { x: 150, w: 60, label: 'Hub', sub: 'route + wrap', color: '#3b82f6' },
        { x: 240, w: 90, label: 'Plugins', sub: 'auth → rate → telemetry', color: '#f59e0b' },
        { x: 360, w: 60, label: 'Channel', sub: 'transport', color: '#06b6d4' },
        { x: 450, w: 70, label: 'Actor B', sub: 'actor.ask()', color: '#10b981' },
      ].map((step, i) => (
        <g key={step.label}>
          {i > 0 && (
            <motion.line
              x1={[40, 150, 240, 360, 450][i] - 15} y1="55"
              x2={step.x - step.w / 2 + 5} y2="55"
              stroke="#475569" strokeWidth="1"
              initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
              viewport={{ once: true }} transition={{ delay: 0.2 + i * 0.12 }} />
          )}
          <motion.rect
            x={step.x - step.w / 2} y="35" width={step.w} height="40" rx="6"
            fill={step.color + '15'} stroke={step.color} strokeWidth="1"
            initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }} transition={{ delay: 0.15 + i * 0.12, type: 'spring' }} />
          <text x={step.x} y="58" textAnchor="middle" className="text-[9px] font-mono font-bold" fill={step.color}>{step.label}</text>
          <text x={step.x} y="70" textAnchor="middle" className="text-[7px] font-mono fill-gray-500">{step.sub}</text>
        </g>
      ))}

      {/* Flowing envelope */}
      <motion.rect width="8" height="6" rx="1.5" fill="#f59e0b"
        animate={{ x: [30, 470], opacity: [0.9, 0.9, 0.9, 0.9, 0] }}
        transition={{ repeat: Infinity, duration: 3, ease: 'linear' }} y="52" />

      <text x="250" y="100" textAnchor="middle" className="fill-gray-600 text-[8px] font-mono">
        Every delegation: wrapped, routed, traced, delivered
      </text>
    </svg>
  )
}

function DistributedDiagram() {
  return (
    <svg viewBox="0 0 500 140" className="w-full max-w-xl">
      {/* Machine 1 */}
      <motion.rect x="10" y="10" width="200" height="120" rx="8"
        fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 4" opacity="0.3"
        initial={{ opacity: 0 }} whileInView={{ opacity: 0.3 }} viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x="110" y="28" textAnchor="middle" className="fill-amber-400/50 text-[8px] font-mono">Machine A</text>

      {/* Machine 2 */}
      <motion.rect x="290" y="10" width="200" height="120" rx="8"
        fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 4" opacity="0.3"
        initial={{ opacity: 0 }} whileInView={{ opacity: 0.3 }} viewport={{ once: true }} transition={{ delay: 0.3 }} />
      <text x="390" y="28" textAnchor="middle" className="fill-amber-400/50 text-[8px] font-mono">Machine B</text>

      {/* Hub A */}
      <motion.circle cx="60" cy="75" r="18" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1"
        initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: 0.4, type: 'spring' }} />
      <text x="60" y="79" textAnchor="middle" className="fill-blue-400 text-[8px] font-mono font-bold">Hub</text>

      {/* Actors on Machine A */}
      {[{ x: 130, y: 55 }, { x: 170, y: 90 }].map((p, i) => (
        <motion.circle key={i} cx={p.x} cy={p.y} r="14" fill="#8b5cf620" stroke="#8b5cf6" strokeWidth="1"
          initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: 0.5 + i * 0.1 }} />
      ))}

      {/* Hub B */}
      <motion.circle cx="340" cy="75" r="18" fill="#3b82f620" stroke="#3b82f6" strokeWidth="1"
        initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: 0.5, type: 'spring' }} />
      <text x="340" y="79" textAnchor="middle" className="fill-blue-400 text-[8px] font-mono font-bold">Hub</text>

      {/* Actors on Machine B */}
      {[{ x: 410, y: 55 }, { x: 450, y: 90 }].map((p, i) => (
        <motion.circle key={i} cx={p.x} cy={p.y} r="14" fill="#10b98120" stroke="#10b981" strokeWidth="1"
          initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: 0.6 + i * 0.1 }} />
      ))}

      {/* HttpChannel connection */}
      <motion.line x1="78" y1="75" x2="322" y2="75" stroke="#06b6d4" strokeWidth="1.5"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ delay: 0.7, duration: 0.6 }} />

      {/* Flowing data dots */}
      <motion.circle r="3" fill="#06b6d4"
        animate={{ cx: [80, 320], opacity: [0.8, 0.8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: 'linear' }} cy="75" />
      <motion.circle r="3" fill="#f59e0b"
        animate={{ cx: [320, 80], opacity: [0.8, 0.8, 0] }}
        transition={{ repeat: Infinity, duration: 2, delay: 1, ease: 'linear' }} cy="75" />

      {/* Channel label */}
      <motion.text x="200" y="68" textAnchor="middle" className="fill-cyan-400 text-[8px] font-mono font-bold"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.9 }}>
        HttpChannel
      </motion.text>

      {/* Local connections */}
      {[{ x1: 60, y1: 75, x2: 130, y2: 55 }, { x1: 60, y1: 75, x2: 170, y2: 90 },
        { x1: 340, y1: 75, x2: 410, y2: 55 }, { x1: 340, y1: 75, x2: 450, y2: 90 }].map((l, i) => (
        <line key={i} {...l} stroke="#475569" strokeWidth="0.8" opacity="0.4" />
      ))}
    </svg>
  )
}

function AutonomyFlowDiagram() {
  return (
    <svg viewBox="0 0 500 130" className="w-full max-w-xl">
      {/* Actor */}
      <motion.rect x="180" y="10" width="140" height="50" rx="8"
        fill="#8b5cf620" stroke="#8b5cf6" strokeWidth="1.5"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x="250" y="32" textAnchor="middle" className="fill-violet-400 text-[10px] font-mono font-bold">Actor</text>
      <text x="250" y="48" textAnchor="middle" className="fill-gray-500 text-[7px] font-mono">autonomous agent</text>

      {/* Observer watching */}
      <motion.g initial={{ opacity: 0, x: -20 }} whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }} transition={{ delay: 0.4 }}>
        <circle cx="60" cy="35" r="22" fill="#10b98115" stroke="#10b981" strokeWidth="1" />
        <motion.circle cx="60" cy="35" r="6" fill="none" stroke="#10b981" strokeWidth="1.5"
          animate={{ r: [6, 8, 6] }} transition={{ repeat: Infinity, duration: 2 }} />
        <circle cx="60" cy="35" r="3" fill="#10b981" />
        <text x="60" y="65" textAnchor="middle" className="fill-emerald-500 text-[8px] font-mono">Observer</text>
      </motion.g>

      {/* Signal arrow: Observer → Actor */}
      <motion.path d="M82 35 L178 35" stroke="#f59e0b" strokeWidth="1.5" markerEnd="url(#sigArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.6 }} />
      <motion.text x="130" y="28" textAnchor="middle" className="fill-amber-400 text-[7px] font-mono"
        animate={{ opacity: [0.4, 1, 0.4] }} transition={{ repeat: Infinity, duration: 2 }}>
        Signal
      </motion.text>

      {/* Scheduler */}
      <motion.g initial={{ opacity: 0, x: 20 }} whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }} transition={{ delay: 0.5 }}>
        <rect x="390" y="15" width="80" height="40" rx="6" fill="#06b6d415" stroke="#06b6d4" strokeWidth="1" />
        <text x="430" y="39" textAnchor="middle" className="fill-cyan-400 text-[8px] font-mono font-bold">Scheduler</text>
        <text x="430" y="65" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">IntervalWatch(10s)</text>
      </motion.g>

      {/* Scheduler → Actor */}
      <motion.path d="M390 35 L322 35" stroke="#06b6d4" strokeWidth="1" strokeDasharray="3 3"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.7 }} />

      {/* Event stream */}
      <motion.path d="M30 100 Q130 80, 250 100 Q370 120, 470 100"
        fill="none" stroke="#3b82f6" strokeWidth="1" opacity="0.3"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ duration: 1, delay: 0.3 }} />
      <text x="250" y="120" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">Event Stream</text>

      {/* Events on stream */}
      {[80, 170, 250, 340, 420].map((x, i) => (
        <motion.circle key={i} cx={x} cy={100} r="3" fill="#3b82f6" opacity="0.4"
          initial={{ scale: 0 }} whileInView={{ scale: 1 }}
          viewport={{ once: true }} transition={{ delay: 0.5 + i * 0.1, type: 'spring' }} />
      ))}

      {/* Observer eye scanning stream */}
      <line x1="60" y1="57" x2="60" y2="95" stroke="#10b981" strokeWidth="0.8" strokeDasharray="2 2" opacity="0.3" />

      <defs>
        <marker id="sigArrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#f59e0b" />
        </marker>
      </defs>
    </svg>
  )
}

/* ── Delegation Flow ──────────────────────────────────────────────────── */

function DelegationFlow() {
  return (
    <FadeIn className="mt-16">
      <div className="p-8 rounded-2xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-2 text-center">How It All Comes Together</h3>
        <p className="text-sm text-gray-500 text-center mb-6 max-w-lg mx-auto">
          A single delegation demonstrates all three themes working in concert
        </p>
        <div className="flex flex-col md:flex-row items-center justify-center gap-3 md:gap-2 text-sm">
          {[
            { label: 'Actor A', sub: 'delegate_to("B", task)', color: 'border-violet-500/30 bg-violet-500/5 text-violet-400', theme: 'autonomy' },
            { label: 'Hub', sub: 'discover + wrap', color: 'border-blue-500/30 bg-blue-500/5 text-blue-400', theme: 'network' },
            { label: 'Plugins', sub: 'auth → rate → log', color: 'border-amber-500/30 bg-amber-500/5 text-amber-400', theme: 'network' },
            { label: 'Channel', sub: 'transport envelope', color: 'border-cyan-500/30 bg-cyan-500/5 text-cyan-400', theme: 'distributed' },
            { label: 'Actor B', sub: 'actor.ask(task)', color: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400', theme: 'autonomy' },
          ].map((step, i) => (
            <div key={step.label} className="flex items-center gap-2">
              {i > 0 && (
                <svg viewBox="0 0 16 16" className="w-4 h-4 text-gray-600 shrink-0 hidden md:block">
                  <path d="M4 8h8M9 5l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
              <motion.div
                className={`px-4 py-3 rounded-lg border ${step.color} text-center min-w-[120px]`}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 + i * 0.1 }}
              >
                <div className="font-mono font-bold text-sm">{step.label}</div>
                <div className="text-[10px] text-gray-500 mt-0.5 font-mono">{step.sub}</div>
              </motion.div>
            </div>
          ))}
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Main section ─────────────────────────────────────────────────────── */

export function CoreConcepts() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="concepts">
      <SectionLabel number="05" title="Deep Dive" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Three themes.</span>{' '}
        <span className="text-gray-500">One unified design.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-4 leading-relaxed">
        AG2 V2 tackles agent fragmentation through three complementary capabilities.
        Each theme has dedicated primitives and building blocks.
        Together they enable agents that are <span className="text-cyan-400">networked</span>,{' '}
        <span className="text-amber-400">distributed</span>, and{' '}
        <span className="text-emerald-400">autonomous</span>.
      </p>
      <p className="text-gray-500 max-w-3xl mb-16">
        Click into each theme to see the components, the benefits, and real examples.
      </p>

      {/* Three theme deep dives */}
      <div className="space-y-8">
        <NetworkTheme />
        <DistributionTheme />
        <AutonomyTheme />
      </div>

      {/* Delegation flow */}
      <DelegationFlow />
    </AnimatedSection>
  )
}
