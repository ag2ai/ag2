import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

function ExplosionTimeline() {
  const milestones = [
    { year: '2024', label: 'Copilot era', agents: '~50K agents deployed', color: '#6b7280', width: '15%' },
    { year: '2025', label: 'Agent frameworks explode', agents: '535% framework growth', color: '#f59e0b', width: '40%' },
    { year: '2026', label: 'Coding agents go mainstream', agents: '40% of enterprise apps', color: '#3b82f6', width: '75%' },
    { year: '2028', label: 'Microsoft projection', agents: '1.3 billion agents', color: '#8b5cf6', width: '100%' },
  ]

  return (
    <div className="relative mt-12 mb-8">
      {milestones.map((m, i) => (
        <motion.div
          key={m.year}
          className="flex items-center gap-4 mb-4"
          initial={{ opacity: 0, x: -30 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.15 * i, duration: 0.5 }}
        >
          <span className="text-sm font-mono font-bold text-gray-400 w-12 shrink-0">{m.year}</span>
          <div className="flex-1 relative h-8">
            <motion.div
              className="h-full rounded-lg flex items-center px-3"
              style={{ backgroundColor: m.color + '20', borderLeft: `3px solid ${m.color}` }}
              initial={{ width: 0 }}
              whileInView={{ width: m.width }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 + 0.15 * i, duration: 0.8, ease: 'easeOut' }}
            >
              <span className="text-xs font-mono text-gray-300 whitespace-nowrap">{m.agents}</span>
            </motion.div>
          </div>
          <span className="text-xs text-gray-500 w-44 shrink-0 hidden md:block">{m.label}</span>
        </motion.div>
      ))}
    </div>
  )
}

function IslandVisualization() {
  const islands = [
    { x: 40, y: 35, label: 'Claude Code', color: '#f97316', size: 22 },
    { x: 150, y: 55, label: 'GPT Agent', color: '#10b981', size: 18 },
    { x: 260, y: 30, label: 'LangGraph', color: '#3b82f6', size: 20 },
    { x: 90, y: 110, label: 'Custom Bot', color: '#8b5cf6', size: 16 },
    { x: 200, y: 120, label: 'Bedrock', color: '#06b6d4', size: 19 },
    { x: 310, y: 100, label: 'CrewAI', color: '#ec4899', size: 17 },
    { x: 370, y: 50, label: 'ADK', color: '#f59e0b', size: 18 },
  ]

  return (
    <svg viewBox="0 0 420 160" className="w-full max-w-lg mx-auto">
      {/* Failed connection attempts — dashed lines with X */}
      {[
        { x1: 40, y1: 35, x2: 150, y2: 55 },
        { x1: 150, y1: 55, x2: 260, y2: 30 },
        { x1: 90, y1: 110, x2: 200, y2: 120 },
        { x1: 260, y1: 30, x2: 370, y2: 50 },
        { x1: 40, y1: 35, x2: 90, y2: 110 },
        { x1: 200, y1: 120, x2: 310, y2: 100 },
      ].map((line, i) => (
        <motion.line
          key={i} {...line}
          stroke="#ef4444" strokeWidth="0.8" strokeDasharray="4 6"
          initial={{ pathLength: 0, opacity: 0 }}
          whileInView={{ pathLength: 1, opacity: 0.2 }}
          viewport={{ once: true }}
          transition={{ delay: 0.8 + i * 0.1, duration: 0.5 }}
        />
      ))}

      {/* X marks on failed connections */}
      {[
        { x: 95, y: 45 }, { x: 205, y: 42 }, { x: 145, y: 115 },
        { x: 315, y: 40 }, { x: 65, y: 72 }, { x: 255, y: 110 },
      ].map((pos, i) => (
        <motion.g key={i}
          initial={{ opacity: 0, scale: 0 }}
          whileInView={{ opacity: 0.5, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 1.2 + i * 0.05, type: 'spring' }}
        >
          <line x1={pos.x - 3} y1={pos.y - 3} x2={pos.x + 3} y2={pos.y + 3} stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
          <line x1={pos.x + 3} y1={pos.y - 3} x2={pos.x - 3} y2={pos.y + 3} stroke="#ef4444" strokeWidth="1.5" strokeLinecap="round" />
        </motion.g>
      ))}

      {/* Islands — isolated agents */}
      {islands.map((island, i) => (
        <motion.g key={island.label}
          initial={{ opacity: 0, scale: 0 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 + i * 0.08, type: 'spring', stiffness: 200 }}
        >
          {/* Isolation ring */}
          <circle
            cx={island.x} cy={island.y} r={island.size + 6}
            fill="none" stroke={island.color} strokeWidth="0.5" strokeDasharray="2 3" opacity="0.3"
          />
          {/* Agent node */}
          <circle cx={island.x} cy={island.y} r={island.size} fill={island.color + '15'} stroke={island.color} strokeWidth="1.2" />
          <motion.circle
            cx={island.x} cy={island.y} r="4" fill={island.color}
            animate={{ opacity: [0.4, 0.8, 0.4] }}
            transition={{ repeat: Infinity, duration: 2 + i * 0.3, ease: 'easeInOut' }}
          />
          <text x={island.x} y={island.y + island.size + 14} textAnchor="middle" className="fill-gray-500 text-[7px] font-mono">
            {island.label}
          </text>
        </motion.g>
      ))}
    </svg>
  )
}

function V1vsV2Comparison() {
  return (
    <div className="grid md:grid-cols-2 gap-8">
      {/* Before: Isolated */}
      <FadeIn delay={0.1}>
        <div className="relative p-8 rounded-2xl bg-gray-900/50 border border-gray-800">
          <div className="text-center mb-4">
            <span className="text-xs font-mono text-red-400 uppercase tracking-widest">Today</span>
            <h4 className="text-lg font-semibold text-gray-300 mt-1">Powerful but Isolated</h4>
          </div>
          <IslandVisualization />
          <div className="flex flex-wrap gap-2 justify-center mt-4">
            {['Different frameworks', 'No common protocol', 'No discovery', 'No coordination'].map(tag => (
              <span key={tag} className="text-xs text-red-400/70 bg-red-500/10 px-2 py-1 rounded border border-red-500/10">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </FadeIn>

      {/* After: Networked */}
      <FadeIn delay={0.3}>
        <div className="relative p-8 rounded-2xl bg-gradient-to-br from-blue-950/30 to-violet-950/30 border border-blue-500/20">
          <div className="text-center mb-4">
            <span className="text-xs font-mono text-blue-400 uppercase tracking-widest">With AG2 V2</span>
            <h4 className="text-lg font-semibold text-white mt-1">Connected Agent Network</h4>
          </div>
          <NetworkedVisualization />
          <div className="flex flex-wrap gap-2 justify-center mt-4">
            {['Any framework', 'Transport-agnostic', 'Auto-discovery', 'Observable'].map(tag => (
              <span key={tag} className="text-xs text-blue-300 bg-blue-500/10 px-2 py-1 rounded border border-blue-500/20">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </FadeIn>
    </div>
  )
}

function NetworkedVisualization() {
  const hubX = 210, hubY = 75
  const agents = [
    { cx: 60, cy: 35, label: 'Claude', color: '#f97316' },
    { cx: 120, cy: 120, label: 'GPT', color: '#10b981' },
    { cx: 310, cy: 30, label: 'LangGraph', color: '#3b82f6' },
    { cx: 350, cy: 110, label: 'ADK', color: '#f59e0b' },
    { cx: 60, cy: 120, label: 'Custom', color: '#8b5cf6' },
    { cx: 300, cy: 130, label: 'Bedrock', color: '#06b6d4' },
  ]

  return (
    <svg viewBox="0 0 420 160" className="w-full max-w-lg mx-auto overflow-visible">
      {/* Hub pulse rings */}
      {[30, 45, 62].map((r, i) => (
        <motion.circle
          key={r} cx={hubX} cy={hubY} r={r}
          fill="none" stroke="#3b82f6" strokeWidth="0.5"
          animate={{ opacity: [0, 0.15, 0], r: [r - 4, r, r + 4] }}
          transition={{ repeat: Infinity, duration: 3, delay: i * 0.8, ease: 'easeOut' }}
        />
      ))}

      {/* Hub core */}
      <motion.circle cx={hubX} cy={hubY} r="20" fill="url(#hubGrad2)" stroke="#3b82f6" strokeWidth="1.5"
        initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.3 }} />
      <text x={hubX} y={hubY + 4} textAnchor="middle" className="fill-blue-300 text-[9px] font-mono font-bold">Hub</text>

      {/* Agents with flowing data */}
      {agents.map((agent, i) => (
        <g key={agent.label}>
          <motion.line x1={hubX} y1={hubY} x2={agent.cx} y2={agent.cy}
            stroke={agent.color} strokeWidth="1" opacity="0.3"
            initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }}
            transition={{ delay: 0.4 + i * 0.08 }} />
          <motion.circle r="2" fill={agent.color}
            animate={{ cx: [hubX, agent.cx], cy: [hubY, agent.cy], opacity: [0.8, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 + i * 0.3, delay: i * 0.4, ease: 'easeOut' }} />
          <motion.circle cx={agent.cx} cy={agent.cy} r="14" fill={agent.color + '15'} stroke={agent.color} strokeWidth="1"
            initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 0.5 + i * 0.08 }} />
          <motion.circle cx={agent.cx} cy={agent.cy} r="4" fill={agent.color}
            animate={{ opacity: [0.3, 0.7, 0.3] }} transition={{ repeat: Infinity, duration: 2 + i * 0.3 }} />
          <text x={agent.cx} y={agent.cy + 24} textAnchor="middle" className="fill-gray-400 text-[7px] font-mono">{agent.label}</text>
        </g>
      ))}

      <defs>
        <radialGradient id="hubGrad2" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.05" />
        </radialGradient>
      </defs>
    </svg>
  )
}

export function ParadigmShift() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="paradigm">
      <SectionLabel number="01" title="The Crisis" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Agents are everywhere.</span><br />
        <span className="text-red-400">They just can't talk to each other.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        Coding agents have made building AI agents{' '}
        <span className="text-white font-medium">universally accessible</span>.
        A non-engineer can ship a working agent in an afternoon.
        The result? An explosion of agents — with no way to connect them.
      </p>

      {/* Stats bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-16">
        {[
          { value: 92, display: '92%', label: 'of US developers use AI coding tools daily', color: 'text-blue-400', source: 'Stack Overflow 2026' },
          { value: 535, display: '535%', label: 'growth in AI agent frameworks on GitHub (YoY)', color: 'text-amber-400', source: 'GitHub' },
          { value: 40, display: '40%', label: 'of enterprise apps will have AI agents by end of 2026', color: 'text-violet-400', source: 'Gartner' },
          { value: 50, display: '50%', label: 'of deployed agents operate in isolated silos', color: 'text-red-400', source: 'Salesforce 2026' },
        ].map((stat, i) => (
          <FadeIn key={stat.label} delay={0.1 * i}>
            <div className="p-4 rounded-xl bg-gray-900/50 border border-gray-800 text-center">
              <div className={`text-3xl md:text-4xl font-bold font-mono ${stat.color} mb-2`}>
                {stat.display}
              </div>
              <p className="text-xs text-gray-400 leading-snug mb-2">{stat.label}</p>
              <p className="text-[10px] text-gray-600 font-mono">{stat.source}</p>
            </div>
          </FadeIn>
        ))}
      </div>

      {/* The argument: 3 forces converging */}
      <FadeIn className="mb-20">
        <div className="relative p-8 md:p-12 rounded-2xl bg-gradient-to-br from-gray-900/80 to-gray-900/40 border border-gray-800">
          <h3 className="text-2xl md:text-3xl font-bold text-white mb-8">
            Three forces creating a perfect storm
          </h3>

          <div className="space-y-10">
            {/* Force 1: Universal Access */}
            <div className="flex items-start gap-5">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center shrink-0 mt-1">
                <span className="text-xl font-bold text-blue-400">1</span>
              </div>
              <div>
                <h4 className="text-lg font-bold text-white mb-2">
                  Coding agents democratized agent creation
                </h4>
                <p className="text-gray-400 leading-relaxed mb-3">
                  Claude Code, Cursor, Devin, Codex — AI now writes{' '}
                  <span className="text-white">30% of Microsoft's code</span> and{' '}
                  <span className="text-white">25% of Google's</span>.
                  Devin dropped from $500/mo to $20/mo. "Vibe coding" became a Collins Word of the Year.
                  Building an agent went from a PhD-level task to an afternoon project.
                </p>
                <div className="flex flex-wrap gap-2">
                  {['Claude Code', 'Cursor', 'Devin ($20/mo)', 'Codex', 'Bolt.new', 'Replit Agent'].map(t => (
                    <span key={t} className="text-[10px] font-mono text-blue-400/70 bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/10">{t}</span>
                  ))}
                </div>
              </div>
            </div>

            {/* Force 2: Explosive Growth */}
            <div className="flex items-start gap-5">
              <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center shrink-0 mt-1">
                <span className="text-xl font-bold text-amber-400">2</span>
              </div>
              <div>
                <h4 className="text-lg font-bold text-white mb-2">
                  The numbers are staggering
                </h4>
                <p className="text-gray-400 leading-relaxed mb-3">
                  AI captured{' '}
                  <span className="text-white">61% of all global venture capital</span> in 2025 — $258.7B.
                  Microsoft projects <span className="text-white">1.3 billion agents by 2028</span>.
                  The agent market is growing from $7.8B to $52.6B by 2030 at 46% CAGR.
                  Every company will soon have dozens of agents — Salesforce reports the average is already 12.
                </p>
                <ExplosionTimeline />
              </div>
            </div>

            {/* Force 3: No Infrastructure */}
            <div className="flex items-start gap-5">
              <div className="w-12 h-12 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-center shrink-0 mt-1">
                <span className="text-xl font-bold text-red-400">3</span>
              </div>
              <div>
                <h4 className="text-lg font-bold text-white mb-2">
                  But there's no infrastructure to connect them
                </h4>
                <p className="text-gray-400 leading-relaxed mb-3">
                  Each agent sits behind its own web app or API endpoint.
                  Your CRM agent doesn't know what your data warehouse agent found.
                  <span className="text-red-400"> 86% of IT leaders</span> say agents will add more complexity than value
                  without integration infrastructure. We're building billions of isolated islands.
                </p>
              </div>
            </div>
          </div>
        </div>
      </FadeIn>

      {/* Today vs With AG2 V2 */}
      <V1vsV2Comparison />

      {/* The historical analogy */}
      <FadeIn className="mt-20">
        <div className="relative p-8 md:p-12 rounded-2xl bg-gradient-to-br from-blue-950/20 to-violet-950/20 border border-blue-500/10">
          <h3 className="text-2xl font-bold text-white mb-6 text-center">
            We've seen this exact moment before
          </h3>
          <div className="grid md:grid-cols-2 gap-8">
            <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800">
              <div className="text-xs font-mono text-amber-400 uppercase tracking-widest mb-3">1970s — Computers</div>
              <p className="text-gray-400 leading-relaxed text-sm">
                Every mainframe was powerful but <span className="text-white">isolated</span>.
                Each ran its own OS, its own protocols. ARPANET connected 4 nodes in 1969.
                It took <span className="text-white">TCP/IP (1983)</span> to create a universal language.
                Then <span className="text-white">HTTP (1991)</span> to create applications on top.
                The value wasn't in any single computer — it was in the network.
              </p>
            </div>
            <div className="p-6 rounded-xl bg-blue-500/5 border border-blue-500/20">
              <div className="text-xs font-mono text-blue-400 uppercase tracking-widest mb-3">2026 — AI Agents</div>
              <p className="text-gray-400 leading-relaxed text-sm">
                Every agent is powerful but <span className="text-white">isolated</span>.
                Each runs in its own framework, its own runtime. A2A and MCP define wire formats (the TCP/IP moment).
                But there's <span className="text-blue-400">no infrastructure layer</span> between the wire and the application —
                no routing, no discovery, no state, no observability.{' '}
                <span className="text-white">That's the layer AG2 V2 builds.</span>
              </p>
            </div>
          </div>
          <div className="mt-6 text-center">
            <p className="text-sm text-gray-500 italic">
              NIST launched an AI Agent Standards Initiative in February 2026 citing
              "<span className="text-gray-400">a fragmented ecosystem and stunted adoption</span>"
              without interoperability infrastructure.
            </p>
          </div>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
