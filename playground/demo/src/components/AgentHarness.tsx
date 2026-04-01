import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

/* ── Knowledge Store ─────────────────────────────────────────────────── */

function KnowledgeStoreTheme() {
  return (
    <FadeIn>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-rose-950/20 to-pink-950/20 border border-rose-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center">
            <span className="text-rose-400 font-mono font-bold text-lg">/</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Persistence</h3>
            <p className="text-xs text-gray-500">How agents remember across conversations</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Today's agents are goldfish. Every conversation starts from scratch.
              Knowledge gathered in a 3-hour research session evaporates when the context window resets.
              There's no standard way for an agent to persist what it learns.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              The <span className="text-rose-400 font-mono">KnowledgeStore</span> gives each actor a virtual
              filesystem — read, write, list, delete with Unix paths. LLMs already understand filesystem
              semantics. Any backend (memory, disk, S3, Redis) implements the same 5-method protocol.
            </p>

            <div className="space-y-3">
              {[
                { component: 'KnowledgeStore', role: 'Protocol: 5 methods (read/write/list/delete/exists). Virtual filesystem per actor.' },
                { component: 'MemoryKnowledgeStore', role: 'In-memory default. Dict-backed. Zero config for development and testing.' },
                { component: 'EventLogWriter', role: 'Utility: persists stream events as WAL entries to /log/. Auto-runs on conversation end.' },
                { component: 'StoreBootstrap', role: 'Protocol: initializes store structure on first use. Default creates /log/, /artifacts/, /memory/.' },
              ].map(item => (
                <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                  <span className="text-xs font-mono font-bold text-rose-400 bg-rose-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                  <p className="text-xs text-gray-400">{item.role}</p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-rose-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Cross-conversation memory', detail: 'An actor picks up where it left off. Research findings, user preferences, working state — all persist between sessions. No more "start over every time."' },
                { benefit: 'LLM-native interface', detail: 'Filesystem semantics are in every LLM\'s training data. The knowledge tool maps directly to read/write/list/delete. No new concepts to teach the model.' },
                { benefit: 'Backend-agnostic', detail: 'MemoryKnowledgeStore for dev. DiskKnowledgeStore for single-machine. S3KnowledgeStore for cloud. Same actor code, different persistence.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-rose-400">
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
                <span className="text-white font-medium">Research Assistant:</span> An actor researches quantum computing for hours,
                saving findings to <span className="text-rose-400 font-mono">/artifacts/</span>. The conversation ends.
                Next session, it reads <span className="text-rose-400 font-mono">/memory/working.md</span> and picks up
                exactly where it left off — with full context of past discoveries.
              </p>
            </div>
          </div>
        </div>

        {/* Knowledge Store visualization */}
        <div className="mt-8 flex justify-center">
          <KnowledgeStoreDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Assembly ─────────────────────────────────────────────────────────── */

function AssemblyTheme() {
  return (
    <FadeIn delay={0.1}>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-indigo-950/20 to-blue-950/20 border border-indigo-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <span className="text-indigo-400 font-mono font-bold text-lg">&gt;</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Assembly</h3>
            <p className="text-xs text-gray-500">How agents compose what the LLM sees</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Before every LLM call, an agent needs to decide: which events to show, what knowledge to inject,
              how to trim for token limits. Hard-coding this makes agents brittle. Every new capability
              means rewriting the context logic.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              The <span className="text-indigo-400 font-mono">Assembler</span> runs a pipeline of{' '}
              <span className="text-indigo-400 font-mono">AssemblyPolicy</span> transforms before each LLM call.
              Each policy sees (prompts, events) and returns modified (prompts, events).
              Policies compose left-to-right — add, remove, reorder without touching others.
            </p>

            <div className="space-y-3">
              {[
                { component: 'ConversationPolicy', role: 'Filters to conversation + tool events only. The default — matches current Agent behavior.' },
                { component: 'NetworkPolicy', role: 'Includes delegation results, signals, scheduler events, topic messages. Formats for LLM readability.' },
                { component: 'SlidingWindowPolicy', role: 'Keeps the last N events. Optional transparency note to LLM about omitted events.' },
                { component: 'TokenBudgetPolicy', role: 'Keeps events within a token budget. Character-based estimation. Retains most recent first.' },
                { component: 'EpisodicMemoryPolicy', role: 'Injects past conversation summaries from /memory/conversations/. Cross-session context.' },
                { component: 'WorkingMemoryPolicy', role: 'Injects /memory/working.md. Persistent actor state across conversations.' },
                { component: 'TopicInboxPolicy', role: 'Injects unread messages from subscribed Hub topics. Network-wide knowledge flow.' },
              ].map(item => (
                <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                  <span className="text-xs font-mono font-bold text-indigo-400 bg-indigo-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                  <p className="text-xs text-gray-400">{item.role}</p>
                </div>
              ))}
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-indigo-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Composable context control', detail: 'Each policy is independent. Mix ConversationPolicy + EpisodicMemoryPolicy + TokenBudgetPolicy. Add or remove policies without touching agent logic.' },
                { benefit: 'Replaces monolithic harness', detail: 'The old ContextHarness was a single class doing everything. Policies decompose it into focused, testable transforms. Each can be developed and tested independently.' },
                { benefit: 'Order-aware validation', detail: 'The assembler validates policy ordering at construction time. Injection policies should come before reduction policies — the framework warns if you get it wrong.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-indigo-400">
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
                <span className="text-white font-medium">Multi-session Assistant:</span> An actor configured with{' '}
                <span className="text-indigo-400 font-mono">[ConversationPolicy, WorkingMemoryPolicy, EpisodicMemoryPolicy, TokenBudgetPolicy]</span>.
                Before each LLM call: filter to conversation events, inject working memory, inject last 3 conversation
                summaries, then trim to fit the token budget. Four independent transforms, zero coupling.
              </p>
            </div>
          </div>
        </div>

        {/* Assembly pipeline visualization */}
        <div className="mt-8 flex justify-center">
          <AssemblyPipelineDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Maintenance (Compact + Aggregate) ───────────────────────────────── */

function MaintenanceTheme() {
  return (
    <FadeIn delay={0.2}>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-teal-950/20 to-emerald-950/20 border border-teal-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
            <span className="text-teal-400 font-mono font-bold text-lg">&amp;</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Maintenance</h3>
            <p className="text-xs text-gray-500">How agents stay healthy over time</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-teal-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Long-running agents hit two walls: context windows fill up (performance degrades),
              and raw event history becomes useless noise. Without maintenance, agents either crash
              or produce increasingly poor results.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              Two complementary strategies:{' '}
              <span className="text-teal-400 font-mono">CompactStrategy</span> reduces the active context
              to respect system constraints.{' '}
              <span className="text-teal-400 font-mono">AggregateStrategy</span> creates organized knowledge
              from raw experience. Compaction removes. Aggregation creates.
            </p>

            <div className="space-y-4">
              <div>
                <h5 className="text-xs font-semibold text-teal-400 mb-2">Compaction</h5>
                <div className="space-y-2">
                  {[
                    { component: 'TailWindowCompact', role: 'Keep last N events, drop the rest. Zero LLM cost. Simplest strategy.' },
                    { component: 'SummarizeCompact', role: 'Summarize old events via LLM, keep recent. Creates CompactionSummary event. One LLM call.' },
                    { component: 'CompactTrigger', role: 'Deterministic triggers: max_events, max_tokens. Fires when ANY threshold is exceeded.' },
                  ].map(item => (
                    <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                      <span className="text-xs font-mono font-bold text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                      <p className="text-xs text-gray-400">{item.role}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h5 className="text-xs font-semibold text-teal-400 mb-2">Aggregation</h5>
                <div className="space-y-2">
                  {[
                    { component: 'ConversationSummaryAggregate', role: 'Summarize conversation → /memory/conversations/. One LLM call per aggregation.' },
                    { component: 'WorkingMemoryAggregate', role: 'Merge new events into /memory/working.md. Maintains persistent actor state.' },
                    { component: 'AggregateTrigger', role: 'Deterministic: every_n_turns, every_n_events, on_end (conversation finish).' },
                  ].map(item => (
                    <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                      <span className="text-xs font-mono font-bold text-teal-400 bg-teal-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                      <p className="text-xs text-gray-400">{item.role}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-teal-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Infinite effective context', detail: 'Compaction keeps the active window healthy. Aggregation preserves knowledge in the store. An agent can run for hours — old context is summarized, not lost.' },
                { benefit: 'Deterministic triggers', detail: 'No guessing when to compact or aggregate. Measurable thresholds (event count, token estimate, turn count) fire automatically. The agent can also trigger manually via the memory tool.' },
                { benefit: 'Cost-transparent', detail: 'LLM-based strategies emit CompactionCompleted and AggregationCompleted events with usage metadata. Observers and plugins can track costs across the network.' },
              ].map(item => (
                <div key={item.benefit} className="flex items-start gap-3">
                  <svg viewBox="0 0 16 16" className="w-4 h-4 mt-0.5 shrink-0 text-teal-400">
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
                <span className="text-white font-medium">Long-running Analyst:</span> An actor processes data for hours.
                After 150 events, <span className="text-teal-400 font-mono">TailWindowCompact</span> fires — drops old events
                (persisting them to <span className="text-teal-400 font-mono">/log/</span>), keeps the last 100. On conversation end,{' '}
                <span className="text-teal-400 font-mono">WorkingMemoryAggregate</span> distills everything into{' '}
                <span className="text-teal-400 font-mono">/memory/working.md</span>. Next session starts with full context,
                zero bloat.
              </p>
            </div>
          </div>
        </div>

        {/* Maintenance flow visualization */}
        <div className="mt-8 flex justify-center">
          <MaintenanceFlowDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Animated diagrams ────────────────────────────────────────────────── */

function KnowledgeStoreDiagram() {
  const entries = [
    { path: '/', indent: 0, icon: 'dir', color: '#f43f5e' },
    { path: 'log/', indent: 1, icon: 'dir', color: '#64748b' },
    { path: 'stream-a1b2.jsonl', indent: 2, icon: 'file', color: '#64748b' },
    { path: 'stream-d4e5.jsonl', indent: 2, icon: 'file', color: '#64748b' },
    { path: 'artifacts/', indent: 1, icon: 'dir', color: '#f59e0b' },
    { path: 'report.md', indent: 2, icon: 'file', color: '#f59e0b' },
    { path: 'dataset.csv', indent: 2, icon: 'file', color: '#f59e0b' },
    { path: 'memory/', indent: 1, icon: 'dir', color: '#8b5cf6' },
    { path: 'working.md', indent: 2, icon: 'file', color: '#8b5cf6' },
    { path: 'conversations/', indent: 2, icon: 'dir', color: '#8b5cf6' },
  ]

  return (
    <svg viewBox="0 0 500 200" className="w-full max-w-xl">
      {/* Store container */}
      <motion.rect x="20" y="10" width="220" height="180" rx="8"
        fill="none" stroke="#f43f5e" strokeWidth="1" strokeDasharray="4 4" opacity="0.3"
        initial={{ opacity: 0 }} whileInView={{ opacity: 0.3 }} viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x="130" y="28" textAnchor="middle" className="fill-rose-400/50 text-[8px] font-mono">KnowledgeStore</text>

      {/* File tree */}
      {entries.map((entry, i) => (
        <motion.g key={entry.path}
          initial={{ opacity: 0, x: -10 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 + i * 0.06 }}
        >
          {entry.icon === 'dir' ? (
            <rect x={35 + entry.indent * 18} y={36 + i * 15} width="6" height="5" rx="1" fill={entry.color} opacity="0.6" />
          ) : (
            <rect x={36 + entry.indent * 18} y={36 + i * 15} width="4" height="5" rx="0.5" fill={entry.color} opacity="0.4" />
          )}
          <text x={46 + entry.indent * 18} y={41 + i * 15}
            className="text-[7px] font-mono" fill={entry.color} opacity="0.8">
            {entry.path}
          </text>
        </motion.g>
      ))}

      {/* Backend swap */}
      {[
        { y: 50, label: 'MemoryKnowledgeStore', sub: 'dict[str, str]', active: true },
        { y: 90, label: 'DiskKnowledgeStore', sub: 'local filesystem', active: false },
        { y: 130, label: 'S3KnowledgeStore', sub: 'object store', active: false },
        { y: 170, label: 'RedisKnowledgeStore', sub: 'low-latency', active: false },
      ].map((backend, i) => (
        <motion.g key={backend.label}
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 + i * 0.1 }}
        >
          <rect x="290" y={backend.y - 12} width="180" height="28" rx="6"
            fill={backend.active ? '#f43f5e15' : '#1e293b'}
            stroke={backend.active ? '#f43f5e' : '#334155'} strokeWidth="1" />
          <text x="320" y={backend.y + 2} className="text-[8px] font-mono font-bold"
            fill={backend.active ? '#f43f5e' : '#64748b'}>{backend.label}</text>
          <text x="320" y={backend.y + 12} className="text-[6px] font-mono"
            fill="#475569">{backend.sub}</text>
          {backend.active && (
            <circle cx="300" cy={backend.y} r="3" fill="#f43f5e" />
          )}
          {!backend.active && (
            <circle cx="300" cy={backend.y} r="3" fill="none" stroke="#334155" strokeWidth="1" />
          )}
        </motion.g>
      ))}

      {/* Arrow from store to backends */}
      <motion.line x1="240" y1="100" x2="285" y2="100"
        stroke="#f43f5e" strokeWidth="1" opacity="0.4"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.5 }} />
      <text x="262" y="96" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">swap</text>
    </svg>
  )
}

function AssemblyPipelineDiagram() {
  const policies = [
    { label: 'Conversation', sub: 'filter events', color: '#6366f1' },
    { label: 'Working\nMemory', sub: 'inject state', color: '#8b5cf6' },
    { label: 'Episodic\nMemory', sub: 'inject history', color: '#a855f7' },
    { label: 'Token\nBudget', sub: 'trim to fit', color: '#c084fc' },
  ]

  return (
    <svg viewBox="0 0 500 100" className="w-full max-w-xl">
      {/* Input */}
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.2 }}>
        <rect x="10" y="25" width="55" height="40" rx="6"
          fill="#1e293b" stroke="#475569" strokeWidth="1" />
        <text x="37" y="42" textAnchor="middle" className="fill-gray-400 text-[7px] font-mono font-bold">prompts</text>
        <text x="37" y="55" textAnchor="middle" className="fill-gray-500 text-[6px] font-mono">events</text>
      </motion.g>

      {/* Arrow to pipeline */}
      <motion.line x1="67" y1="45" x2="85" y2="45"
        stroke="#475569" strokeWidth="1"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.3 }} />

      {/* Policy pipeline */}
      {policies.map((policy, i) => (
        <motion.g key={policy.label}
          initial={{ opacity: 0, scale: 0.8 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.35 + i * 0.12, type: 'spring' }}
        >
          <rect x={88 + i * 88} y="20" width="78" height="50" rx="6"
            fill={policy.color + '15'} stroke={policy.color} strokeWidth="1" />
          <text x={127 + i * 88} y="43" textAnchor="middle"
            className="text-[8px] font-mono font-bold" fill={policy.color}>
            {policy.label.split('\n').map((line, j) => (
              <tspan key={j} x={127 + i * 88} dy={j === 0 ? 0 : 10}>{line}</tspan>
            ))}
          </text>
          <text x={127 + i * 88} y="62" textAnchor="middle"
            className="text-[6px] font-mono fill-gray-500">{policy.sub}</text>

          {/* Arrow between policies */}
          {i < policies.length - 1 && (
            <line x1={168 + i * 88} y1="45" x2={88 + (i + 1) * 88} y2="45"
              stroke="#475569" strokeWidth="1" />
          )}
        </motion.g>
      ))}

      {/* Arrow to LLM */}
      <motion.line x1={168 + (policies.length - 1) * 88} y1="45" x2={178 + (policies.length - 1) * 88} y2="45"
        stroke="#475569" strokeWidth="1"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.8 }} />

      {/* LLM call */}
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.85 }}>
        <rect x="455" y="25" width="40" height="40" rx="6"
          fill="#3b82f620" stroke="#3b82f6" strokeWidth="1.5" />
        <text x="475" y="49" textAnchor="middle" className="fill-blue-400 text-[8px] font-mono font-bold">LLM</text>
      </motion.g>

      {/* Flowing dot */}
      <motion.circle r="3" fill="#6366f1" cy="45"
        animate={{ cx: [15, 490], opacity: [0.8, 0.8, 0.8, 0.8, 0] }}
        transition={{ repeat: Infinity, duration: 3.5, ease: 'linear' }} />

      {/* Label */}
      <text x="250" y="90" textAnchor="middle" className="fill-gray-600 text-[8px] font-mono">
        Each policy: (prompts, events) in, (prompts, events) out
      </text>
    </svg>
  )
}

function MaintenanceFlowDiagram() {
  return (
    <svg viewBox="0 0 500 140" className="w-full max-w-xl">
      {/* Active stream */}
      <motion.rect x="20" y="15" width="140" height="50" rx="8"
        fill="#14b8a615" stroke="#14b8a6" strokeWidth="1"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x="90" y="35" textAnchor="middle" className="fill-teal-400 text-[9px] font-mono font-bold">Event Stream</text>
      <text x="90" y="50" textAnchor="middle" className="fill-gray-500 text-[7px] font-mono">150 events (threshold!)</text>

      {/* Compact arrow */}
      <motion.path d="M165 30 L220 30" stroke="#14b8a6" strokeWidth="1.5" markerEnd="url(#tealArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.4 }} />
      <text x="193" y="24" textAnchor="middle" className="fill-teal-400/60 text-[7px] font-mono">compact</text>

      {/* Compacted stream */}
      <motion.rect x="225" y="15" width="100" height="50" rx="8"
        fill="#14b8a610" stroke="#14b8a6" strokeWidth="1" strokeDasharray="3 3"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.5 }} />
      <text x="275" y="35" textAnchor="middle" className="fill-teal-400 text-[9px] font-mono font-bold">Compacted</text>
      <text x="275" y="50" textAnchor="middle" className="fill-gray-500 text-[7px] font-mono">100 events</text>

      {/* Dropped events to log */}
      <motion.path d="M90 68 L90 95 L160 95" stroke="#64748b" strokeWidth="1" strokeDasharray="3 3"
        fill="none"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.6 }} />

      <motion.rect x="165" y="82" width="100" height="30" rx="6"
        fill="#1e293b" stroke="#334155" strokeWidth="1"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.7 }} />
      <text x="215" y="98" textAnchor="middle" className="fill-gray-500 text-[7px] font-mono">/log/stream.jsonl</text>
      <text x="215" y="108" textAnchor="middle" className="fill-gray-600 text-[6px] font-mono">persisted to store</text>

      {/* Aggregate arrow */}
      <motion.path d="M330 40 L370 40" stroke="#8b5cf6" strokeWidth="1.5" markerEnd="url(#purpleArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.7 }} />
      <text x="350" y="34" textAnchor="middle" className="fill-violet-400/60 text-[7px] font-mono">aggregate</text>

      {/* Knowledge store outputs */}
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.8 }}>
        <rect x="375" y="10" width="115" height="30" rx="6"
          fill="#8b5cf615" stroke="#8b5cf6" strokeWidth="1" />
        <text x="432" y="25" textAnchor="middle" className="fill-violet-400 text-[7px] font-mono font-bold">/memory/working.md</text>
        <text x="432" y="35" textAnchor="middle" className="fill-gray-500 text-[6px] font-mono">persistent state</text>

        <rect x="375" y="48" width="115" height="30" rx="6"
          fill="#8b5cf610" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="3 3" />
        <text x="432" y="63" textAnchor="middle" className="fill-violet-400/70 text-[7px] font-mono">/memory/conversations/</text>
        <text x="432" y="73" textAnchor="middle" className="fill-gray-600 text-[6px] font-mono">episode summaries</text>
      </motion.g>

      {/* Cycle arrow */}
      <motion.path d="M432 82 Q432 120, 275 120 Q120 120, 90 68"
        fill="none" stroke="#14b8a6" strokeWidth="1" strokeDasharray="4 4" opacity="0.3"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 1, duration: 0.8 }} />
      <text x="270" y="132" textAnchor="middle" className="fill-gray-600 text-[7px] font-mono">
        next conversation injects from store
      </text>

      <defs>
        <marker id="tealArrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#14b8a6" />
        </marker>
        <marker id="purpleArrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#8b5cf6" />
        </marker>
      </defs>
    </svg>
  )
}

/* ── Network Context Sharing ──────────────────────────────────────────── */

function NetworkContextTheme() {
  return (
    <FadeIn delay={0.3}>
      <div className="relative p-8 md:p-10 rounded-2xl bg-gradient-to-br from-amber-950/20 to-orange-950/20 border border-amber-500/15">
        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
            <span className="text-amber-400 font-mono font-bold text-lg">#</span>
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Network Context</h3>
            <p className="text-xs text-gray-500">How agents share knowledge across the network</p>
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h4 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-4">The Problem & Our Solution</h4>
            <p className="text-sm text-gray-400 leading-relaxed mb-4">
              Individual agent memory isn't enough. In a network, agents need to broadcast discoveries,
              subscribe to relevant streams, and query each other's knowledge — all without tight coupling.
              Without a shared context layer, multi-agent systems devolve into isolated silos or brittle
              point-to-point wiring.
            </p>
            <p className="text-sm text-gray-400 leading-relaxed mb-6">
              The Hub provides <span className="text-amber-400 font-mono">topic-based pub/sub</span> with
              cursor tracking and <span className="text-amber-400 font-mono">cross-actor knowledge queries</span> with
              access control. A unified <span className="text-amber-400 font-mono">network tool</span> gives
              every actor seven actions — publish, subscribe, topics, discover, request, query, query_list —
              through a single tool call.
            </p>

            <div className="space-y-4">
              <div>
                <h5 className="text-xs font-semibold text-amber-400 mb-2">Topics & Pub/Sub</h5>
                <div className="space-y-2">
                  {[
                    { component: 'Hub.publish()', role: 'Broadcast a TopicMessage to a named topic. Any subscribed actor receives it.' },
                    { component: 'Hub.subscribe_topic()', role: 'Subscribe an actor. Cursor starts at current end — no replay of old messages.' },
                    { component: 'Hub.read_topic()', role: 'Read new messages since last cursor position. Advances cursor after read.' },
                    { component: 'Hub.peek_topic()', role: 'Read without advancing cursor. Used by TopicInboxPolicy to respect overflow strategy.' },
                    { component: 'TopicMessage', role: 'Event: topic, sender, message, data dict. Emitted on Hub stream for observers.' },
                  ].map(item => (
                    <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                      <span className="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                      <p className="text-xs text-gray-400">{item.role}</p>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h5 className="text-xs font-semibold text-amber-400 mb-2">Cross-Actor Knowledge Queries</h5>
                <div className="space-y-2">
                  {[
                    { component: 'Hub.query_knowledge()', role: 'Read from another actor\'s KnowledgeStore. Only succeeds if path matches exposed prefixes.' },
                    { component: 'Hub.list_knowledge()', role: 'List entries in another actor\'s store. Same access control via exposed_paths.' },
                    { component: 'exposed_paths', role: 'Registered at Hub.register() time. Default: private. Expose /artifacts/ to share, keep /memory/ hidden.' },
                  ].map(item => (
                    <div key={item.component} className="flex items-start gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                      <span className="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded shrink-0 mt-0.5">{item.component}</span>
                      <p className="text-xs text-gray-400">{item.role}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold text-amber-400 uppercase tracking-wider mb-4">Why This Matters</h4>
            <div className="space-y-4 mb-6">
              {[
                { benefit: 'Decoupled broadcasting', detail: 'Topics are named channels. Publishers don\'t know who subscribes. Subscribers don\'t know who publishes. Add or remove actors without rewiring. True pub/sub semantics.' },
                { benefit: 'Cursor-tracked delivery', detail: 'Each actor has an independent cursor per topic. Read advances it. Peek doesn\'t. No messages lost, no duplicates. Actors can go offline and catch up on reconnect.' },
                { benefit: 'Access-controlled queries', detail: 'Knowledge stores are private by default. Actors explicitly expose path prefixes at registration. An analyst can query /artifacts/ but not /memory/. Zero-trust by default.' },
                { benefit: 'Overflow strategies', detail: 'When topic messages pile up faster than an actor processes them, TopicInboxPolicy handles it: NEWEST (drop old), OLDEST (defer new), or SUMMARY (LLM-compress the backlog). No lost context.' },
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

            {/* Unified network tool */}
            <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800 mb-4">
              <p className="text-xs text-gray-500 mb-3 font-mono uppercase tracking-wider">Unified Network Tool</p>
              <p className="text-xs text-gray-400 mb-3">
                Hub injects a single <span className="text-amber-400 font-mono">network</span> tool into every registered actor.
                Seven actions through one tool:
              </p>
              <div className="grid grid-cols-2 gap-1.5">
                {[
                  { action: 'publish', desc: 'broadcast to topic' },
                  { action: 'subscribe', desc: 'join a topic' },
                  { action: 'topics', desc: 'list active topics' },
                  { action: 'discover', desc: 'find agents by capability' },
                  { action: 'request', desc: 'delegate task to agent' },
                  { action: 'query', desc: 'read another\'s knowledge' },
                  { action: 'query_list', desc: 'list another\'s entries' },
                ].map(item => (
                  <div key={item.action} className="flex items-center gap-2 py-1">
                    <span className="text-[10px] font-mono font-bold text-amber-400">{item.action}</span>
                    <span className="text-[10px] text-gray-600">{item.desc}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-4 rounded-xl bg-gray-950/60 border border-gray-800">
              <p className="text-xs text-gray-500 mb-2 font-mono uppercase tracking-wider">In Practice</p>
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-white font-medium">Research Team:</span> Three actors —
                Crawler, Analyst, Writer. Crawler publishes raw findings to{' '}
                <span className="text-amber-400 font-mono">#discoveries</span>.
                Analyst subscribes, processes findings, writes reports to{' '}
                <span className="text-amber-400 font-mono">/artifacts/</span> (exposed).
                Writer queries Analyst's <span className="text-amber-400 font-mono">/artifacts/report.md</span> to
                draft the final document. No direct wiring — just topics and exposed paths.
              </p>
            </div>
          </div>
        </div>

        {/* Network context visualization */}
        <div className="mt-8 flex justify-center">
          <NetworkContextDiagram />
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Scenario showcase ───────────────────────────────────────────────── */

function NetworkContextDiagram() {
  const actors = [
    { name: 'Crawler', x: 60, y: 30, color: '#f59e0b' },
    { name: 'Analyst', x: 60, y: 110, color: '#f59e0b' },
    { name: 'Writer', x: 390, y: 110, color: '#f59e0b' },
  ]

  return (
    <svg viewBox="0 0 500 170" className="w-full max-w-xl">
      {/* Hub */}
      <motion.rect x="170" y="15" width="160" height="140" rx="10"
        fill="none" stroke="#f59e0b" strokeWidth="1" strokeDasharray="4 4" opacity="0.2"
        initial={{ opacity: 0 }} whileInView={{ opacity: 0.2 }}
        viewport={{ once: true }} transition={{ delay: 0.2 }} />
      <text x="250" y="32" textAnchor="middle" className="fill-amber-400/40 text-[8px] font-mono">Hub</text>

      {/* Topic channel */}
      <motion.rect x="185" y="45" width="130" height="28" rx="6"
        fill="#f59e0b15" stroke="#f59e0b" strokeWidth="1"
        initial={{ opacity: 0, scale: 0.9 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.3 }} />
      <text x="250" y="57" textAnchor="middle" className="fill-amber-400 text-[9px] font-mono font-bold">#discoveries</text>
      <text x="250" y="68" textAnchor="middle" className="fill-gray-500 text-[6px] font-mono">topic with cursors</text>

      {/* Knowledge query path */}
      <motion.rect x="185" y="95" width="130" height="28" rx="6"
        fill="#8b5cf610" stroke="#8b5cf6" strokeWidth="1" strokeDasharray="3 3"
        initial={{ opacity: 0, scale: 0.9 }} whileInView={{ opacity: 1, scale: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.4 }} />
      <text x="250" y="107" textAnchor="middle" className="fill-violet-400 text-[9px] font-mono font-bold">query_knowledge</text>
      <text x="250" y="118" textAnchor="middle" className="fill-gray-500 text-[6px] font-mono">exposed_paths check</text>

      {/* Actors */}
      {actors.map((actor, i) => (
        <motion.g key={actor.name}
          initial={{ opacity: 0, x: actor.x < 200 ? -15 : 15 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 + i * 0.1 }}
        >
          <rect x={actor.x - 40} y={actor.y - 14} width="80" height="28" rx="6"
            fill="#1e293b" stroke={actor.color} strokeWidth="1" />
          <text x={actor.x} y={actor.y + 4} textAnchor="middle"
            className="text-[9px] font-mono font-bold" fill={actor.color}>{actor.name}</text>
        </motion.g>
      ))}

      {/* Crawler → publish → topic */}
      <motion.path d="M100 35 L180 55" stroke="#f59e0b" strokeWidth="1.5"
        markerEnd="url(#amberArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.5 }} />
      <text x="130" y="38" className="fill-amber-400/60 text-[7px] font-mono">publish</text>

      {/* Topic → subscribe → Analyst */}
      <motion.path d="M210 73 L100 105" stroke="#f59e0b" strokeWidth="1.5"
        markerEnd="url(#amberArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.6 }} />
      <text x="140" y="82" className="fill-amber-400/60 text-[7px] font-mono">subscribe</text>

      {/* Writer → query → knowledge check */}
      <motion.path d="M350 110 L320 110" stroke="#8b5cf6" strokeWidth="1.5"
        markerEnd="url(#violetArrow)"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.7 }} />
      <text x="340" y="102" className="fill-violet-400/60 text-[7px] font-mono">query</text>

      {/* Knowledge check → Analyst's store */}
      <motion.path d="M210 110 L100 110" stroke="#8b5cf6" strokeWidth="1"
        strokeDasharray="3 3"
        initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.8 }} />
      <text x="155" y="135" textAnchor="middle" className="fill-gray-600 text-[6px] font-mono">
        /artifacts/ exposed ✓ — /memory/ blocked ✗
      </text>

      {/* Flowing dots on topic */}
      <motion.circle r="2.5" fill="#f59e0b" cy="55"
        animate={{ cx: [185, 315], opacity: [0.8, 0.8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: 'linear', delay: 0.5 }} />
      <motion.circle r="2.5" fill="#f59e0b" cy="55"
        animate={{ cx: [185, 315], opacity: [0.8, 0.8, 0] }}
        transition={{ repeat: Infinity, duration: 2, ease: 'linear', delay: 1.5 }} />

      {/* Overflow badge */}
      <motion.g initial={{ opacity: 0 }} whileInView={{ opacity: 1 }}
        viewport={{ once: true }} transition={{ delay: 0.9 }}>
        <rect x="340" y="42" width="90" height="32" rx="5"
          fill="#1e293b" stroke="#475569" strokeWidth="1" />
        <text x="385" y="55" textAnchor="middle" className="fill-gray-400 text-[7px] font-mono font-bold">Overflow</text>
        <text x="385" y="68" textAnchor="middle" className="fill-gray-600 text-[6px] font-mono">NEWEST | OLDEST | SUMMARY</text>
      </motion.g>

      <defs>
        <marker id="amberArrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#f59e0b" />
        </marker>
        <marker id="violetArrow" viewBox="0 0 6 6" refX="6" refY="3" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0 0 L6 3 L0 6z" fill="#8b5cf6" />
        </marker>
      </defs>
    </svg>
  )
}

function ScenarioShowcase() {
  const scenarios = [
    {
      number: '1',
      title: 'Single Actor, Long Context',
      subtitle: 'Hot memory within one conversation',
      example: 'Research agent running for hours',
      config: 'knowledge_store + compact + assembly',
      flow: ['Agent runs', 'Knowledge tool saves findings', 'Compaction keeps context healthy', 'Agent reads past findings'],
      color: 'rose',
    },
    {
      number: '2',
      title: 'Single Actor, Multi-Session',
      subtitle: 'Episodic memory across conversations',
      example: 'Assistant that remembers past sessions',
      config: 'knowledge_store + aggregate + episodic policy',
      flow: ['Conversation 1 ends', 'Aggregation writes summary', 'Conversation 2 starts', 'Episodic policy injects history'],
      color: 'indigo',
    },
    {
      number: '3',
      title: 'Multi-Actor Knowledge Sharing',
      subtitle: 'Network-wide knowledge flow',
      example: 'Team of agents sharing findings',
      config: 'topics + exposed_paths + query + network tool',
      flow: ['Crawler publishes to #discoveries topic', 'TopicInboxPolicy injects unread messages', 'Analyst queries Crawler\'s /artifacts/ via Hub', 'exposed_paths blocks access to /memory/'],
      color: 'teal',
    },
    {
      number: '4',
      title: 'High-Volume Topic Processing',
      subtitle: 'Overflow handling for busy networks',
      example: 'Monitoring agent on a firehose topic',
      config: 'TopicInboxPolicy + overflow + SUMMARY',
      flow: ['50 messages arrive on #alerts topic', 'TopicInboxPolicy detects overflow', 'SUMMARY strategy LLM-compresses backlog', 'Agent sees one concise summary, not 50 raw msgs'],
      color: 'amber',
    },
  ]

  return (
    <FadeIn className="mt-16">
      <div className="p-8 rounded-2xl bg-gray-900/50 border border-gray-800">
        <h3 className="text-lg font-bold text-white mb-2 text-center">Four Scenarios, One System</h3>
        <p className="text-sm text-gray-500 text-center mb-8 max-w-lg mx-auto">
          The same primitives compose to cover all stateful agent needs
        </p>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          {scenarios.map((s, i) => {
            const colors = {
              rose: { border: 'border-rose-500/20', bg: 'bg-rose-500/5', text: 'text-rose-400', badge: 'bg-rose-500/10' },
              indigo: { border: 'border-indigo-500/20', bg: 'bg-indigo-500/5', text: 'text-indigo-400', badge: 'bg-indigo-500/10' },
              teal: { border: 'border-teal-500/20', bg: 'bg-teal-500/5', text: 'text-teal-400', badge: 'bg-teal-500/10' },
              amber: { border: 'border-amber-500/20', bg: 'bg-amber-500/5', text: 'text-amber-400', badge: 'bg-amber-500/10' },
            }[s.color]!

            return (
              <motion.div
                key={s.title}
                className={`p-5 rounded-xl border ${colors.border} ${colors.bg}`}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 + i * 0.12 }}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className={`w-6 h-6 rounded-md ${colors.badge} flex items-center justify-center text-xs font-mono font-bold ${colors.text}`}>
                    {s.number}
                  </span>
                  <span className={`text-sm font-bold ${colors.text}`}>{s.title}</span>
                </div>
                <p className="text-xs text-gray-500 mb-1">{s.subtitle}</p>
                <p className="text-xs text-gray-400 mb-3 italic">{s.example}</p>

                <div className="mb-3">
                  <span className={`text-[10px] font-mono px-2 py-0.5 rounded ${colors.badge} ${colors.text}`}>
                    {s.config}
                  </span>
                </div>

                <div className="space-y-1.5">
                  {s.flow.map((step, j) => (
                    <div key={step} className="flex items-start gap-2">
                      <span className="text-[10px] font-mono text-gray-600 mt-0.5 w-3 text-right shrink-0">{j + 1}</span>
                      <p className="text-xs text-gray-400">{step}</p>
                    </div>
                  ))}
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </FadeIn>
  )
}

/* ── Main section ─────────────────────────────────────────────────────── */

export function AgentHarness() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="harness">
      <SectionLabel number="06" title="Agent Harness" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Stateful agents.</span>{' '}
        <span className="text-gray-500">Production grade.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-4 leading-relaxed">
        The Agent Harness is the system that enables genuine stateful operations.
        Four concerns — <span className="text-rose-400">persistence</span>,{' '}
        <span className="text-indigo-400">assembly</span>,{' '}
        <span className="text-teal-400">maintenance</span>, and{' '}
        <span className="text-amber-400">network context</span> — compose to cover every
        stateful scenario: single-session, multi-session, and multi-actor.
      </p>
      <p className="text-gray-500 max-w-3xl mb-16">
        Each primitive is independent, protocol-based, and backend-agnostic.
        All additive — zero changes to Agent, Stream, or any Layer 1 code.
      </p>

      {/* Four harness themes */}
      <div className="space-y-8">
        <KnowledgeStoreTheme />
        <AssemblyTheme />
        <MaintenanceTheme />
        <NetworkContextTheme />
      </div>

      {/* Scenario showcase */}
      <ScenarioShowcase />
    </AnimatedSection>
  )
}
