import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { AnimatedSection, FadeIn } from '../shared/AnimatedSection'
import { SectionLabel } from '../shared/SectionLabel'
import { NetworkGraph } from './NetworkGraph'
import { EventLog } from './EventLog'
import { PlaybackControls } from './PlaybackControls'
import { usePlayback } from './usePlayback'
import { replays } from '../../data/replays'

const THEME_CFG: Record<string, { color: string; icon: string }> = {
  network:     { color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/20', icon: '~' },
  distributed: { color: 'text-amber-400 bg-amber-500/10 border-amber-500/20', icon: '::' },
  autonomy:    { color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', icon: '*' },
}

const EXAMPLE_GROUPS: { label: string; subtitle: string; color: string; indices: number[] }[] = [
  {
    label: 'Go-to-Market',
    subtitle: 'Customer, content, sales, market intel',
    color: 'text-cyan-500 border-cyan-500/20',
    indices: [0, 2, 8, 10], // Customer Support, Newsroom, GTM Sales, VoC
  },
  {
    label: 'Operations & Safety',
    subtitle: 'Infrastructure, response, governance',
    color: 'text-emerald-500 border-emerald-500/20',
    indices: [1, 3, 4, 7], // SRE Bot, Emergency, Smart Building, Safety Harness
  },
  {
    label: 'Decision Intelligence',
    subtitle: 'Risk, approvals, scenario analysis',
    color: 'text-amber-500 border-amber-500/20',
    indices: [5, 6, 9, 11], // Trading Desk, Loan Pipeline, Hiring Pipeline, Simulation
  },
]

/** Rich descriptions from the ExampleShowcase (merged here). */
const EXAMPLE_DETAILS: Record<string, { desc: string; cardColor: string }> = {
  '01': {
    desc: 'One actor with domain tools handles customer inquiries. The simplest AG2 network demo — shows how Actor extends Agent with zero overhead.',
    cardColor: 'border-emerald-500/20 bg-emerald-500/5',
  },
  '02': {
    desc: 'An agent that operates without human prompts — scheduled health checks every 10 seconds via IntervalWatch, automatic incident investigation, self-healing responses.',
    cardColor: 'border-emerald-500/20 bg-emerald-500/5',
  },
  '03': {
    desc: 'Four agents (Researcher \u2192 Writer \u2192 Editor \u2192 Publisher) collaborate through a Hub with TokenMonitor and LoopDetector observers providing guardrails.',
    cardColor: 'border-cyan-500/20 bg-cyan-500/5',
  },
  '04': {
    desc: 'Always-on 911 dispatch center across 4 servers. Priority triage (CRITICAL vs MINOR), live EMS-hospital transport updates, and hot-adding a fire department at runtime via Hub.connect().',
    cardColor: 'border-amber-500/20 bg-amber-500/5',
  },
  '05': {
    desc: 'HVAC, Energy, Security, and Maintenance agents span two servers (climate + operations). A Network with Scheduler runs autonomous health checks every 10 seconds. Same code runs in-process or distributed across buildings.',
    cardColor: 'border-blue-500/20 bg-blue-500/5',
  },
  '06': {
    desc: 'Custom priority scheme, custom events, routing plugins (RiskGate, RateLimiter), system plugins (ComplianceAudit), and context harnesses. Every extensibility point demonstrated.',
    cardColor: 'border-violet-500/20 bg-violet-500/5',
  },
  '07': {
    desc: 'Three agents (Intake \u2192 Credit Analyst \u2192 Underwriter) process applications through a Hub. A custom ApprovalGate routing plugin pauses for human approval before critical agents receive work.',
    cardColor: 'border-violet-500/20 bg-violet-500/5',
  },
  '08': {
    desc: 'Multiple observers monitor budget, detect loops, enforce content policies, and prevent tool abuse. Signals reshape agent behavior in real-time. FATAL signals halt execution mechanically.',
    cardColor: 'border-emerald-500/20 bg-emerald-500/5',
  },
  '09': {
    desc: 'Five agents (SDR, AE, SE, Marketing, CS) coordinate the full sales cycle. A PipelineGuard plugin enforces valid routing between agents. Scheduler drives autonomous pipeline reviews.',
    cardColor: 'border-cyan-500/20 bg-cyan-500/5',
  },
  '10': {
    desc: 'Fanout topology sends candidates to three parallel screeners (technical, culture, background) simultaneously. A Conditional topology then routes based on results: strong matches fast-track to the hiring manager, others go through standard review.',
    cardColor: 'border-blue-500/20 bg-blue-500/5',
  },
  '11': {
    desc: 'Always-on social listening network. A collector scans X, Reddit, reviews, and news for customer mentions. An AI analyst classifies each mention by category and sentiment, then routes to four specialists in parallel: product-inspector for defects, market-intel for competitive analysis, pr-responder for viral crises, and escalation for safety/legal urgency. Custom observers track volume spikes and sentiment drift. A ContentRouter plugin validates routing.',
    cardColor: 'border-rose-500/20 bg-rose-500/5',
  },
  '12': {
    desc: 'Scale-out network with 11 actors — nearly double any previous example. A coordinator discovers 9 simulation personas (CFO, CMO, CTO, HR, Legal, Customer, Ops, Competitor, Board) via discover_agents, delegates the decision to all of them in parallel, then an analyst synthesizes every perspective into a unified risk/opportunity report with a final recommendation.',
    cardColor: 'border-amber-500/20 bg-amber-500/5',
  },
}

function ReplayViewer() {
  const [exampleIdx, setExampleIdx] = useState(0)
  const [scenarioIdx, setScenarioIdx] = useState(0)

  const replay = replays[exampleIdx]
  const safeScenarioIdx = scenarioIdx < replay.scenarios.length ? scenarioIdx : 0
  const scenario = replay.scenarios[safeScenarioIdx]
  const detail = EXAMPLE_DETAILS[replay.example.id]

  // Reset scenario when example changes
  useEffect(() => {
    setScenarioIdx(0)
  }, [exampleIdx])

  const playback = usePlayback(scenario.events, scenario.totalDurationMs)

  // Reset playback when scenario changes
  useEffect(() => {
    playback.reset()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [exampleIdx, scenarioIdx])

  return (
    <div>
      {/* Grouped example selector */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {EXAMPLE_GROUPS.map(group => (
          <div key={group.label} className={`rounded-xl border border-gray-800 bg-gray-900/20 p-3`}>
            <div className="mb-3">
              <span className={`text-[10px] font-mono font-semibold uppercase tracking-wider ${group.color.split(' ')[0]}`}>
                {group.label}
              </span>
              <span className="text-[10px] font-mono text-gray-600 ml-2">
                {group.subtitle}
              </span>
            </div>
            <div className="flex flex-col gap-1.5">
              {group.indices.map(i => {
                const r = replays[i]
                return (
                  <button
                    key={r.example.id}
                    onClick={() => { setExampleIdx(i); playback.reset() }}
                    className={`px-3 py-2 rounded-lg text-xs font-mono transition-all text-left ${
                      exampleIdx === i
                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                        : 'bg-gray-900/50 text-gray-500 border border-gray-800 hover:border-gray-700 hover:text-gray-400'
                    }`}
                  >
                    {r.example.title}
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Example description card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={exampleIdx}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.2 }}
          className={`p-6 rounded-2xl border mb-6 ${detail?.cardColor ?? 'border-gray-800 bg-gray-900/30'}`}
        >
          <div className="flex items-center gap-3 mb-3">
            <h4 className="text-lg font-bold text-white">{replay.example.title}</h4>
            <span className="text-[10px] font-mono text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
              {replay.example.category}
            </span>
          </div>
          <div className="flex gap-2 mb-3">
            {replay.example.themes.map(theme => {
              const cfg = THEME_CFG[theme]
              return (
                <span key={theme} className={`inline-flex items-center gap-1 text-xs font-mono font-medium px-2 py-0.5 rounded-full border ${cfg.color}`}>
                  <span>{cfg.icon}</span>{theme}
                </span>
              )
            })}
          </div>
          <p className="text-sm text-gray-400 leading-relaxed">{detail?.desc ?? replay.example.description}</p>
        </motion.div>
      </AnimatePresence>

      {/* Scenario tabs */}
      {replay.scenarios.length > 1 && (
        <div className="flex gap-2 mb-4">
          {replay.scenarios.map((s, i) => (
            <button
              key={s.id}
              onClick={() => { setScenarioIdx(i); playback.reset() }}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                scenarioIdx === i
                  ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                  : 'bg-gray-900/50 text-gray-500 border border-gray-800 hover:border-gray-700'
              }`}
            >
              Scenario {s.id}: {s.title}
            </button>
          ))}
        </div>
      )}

      {/* Input message */}
      <div className="mb-4 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
        <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider">Input</span>
        <p className="text-xs text-gray-400 mt-1 leading-relaxed">{scenario.inputMessage}</p>
      </div>

      {/* Main replay area */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-4">
        {/* Network graph - left/main area */}
        <div className="md:col-span-3 p-4 rounded-xl bg-gray-900/30 border border-gray-800">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider">
              Agent Network
              {scenario.servers && (
                <span className="text-cyan-500 ml-2">Distributed</span>
              )}
            </span>
            <div className="flex items-center gap-3 text-[9px] font-mono text-gray-600">
              <span className="flex items-center gap-1">
                <span className="w-3 h-0.5 bg-violet-500 inline-block rounded" /> local
              </span>
              {scenario.servers && (
                <span className="flex items-center gap-1">
                  <span className="w-3 h-0.5 bg-cyan-500 inline-block rounded" style={{ borderBottom: '1px dashed #06b6d4' }} /> HTTP
                </span>
              )}
            </div>
          </div>
          <NetworkGraph
            agents={scenario.agents}
            servers={scenario.servers}
            serverConnections={scenario.serverConnections}
            activeAgent={playback.activeAgent}
            activeDelegations={playback.activeDelegations}
            visibleEvents={playback.visibleEvents}
          />
        </div>

        {/* Event log - right panel */}
        <div className="md:col-span-2 rounded-xl bg-gray-900/30 border border-gray-800 p-3 flex flex-col" style={{ minHeight: '300px', maxHeight: '400px' }}>
          <span className="text-[10px] font-mono text-gray-600 uppercase tracking-wider mb-2">
            Event Stream ({playback.visibleEvents.length}/{scenario.events.length})
          </span>
          <div className="flex-1 overflow-hidden">
            <EventLog events={playback.visibleEvents} />
          </div>
        </div>
      </div>

      {/* Playback controls */}
      <div className="p-3 rounded-xl bg-gray-900/50 border border-gray-800">
        <PlaybackControls
          isPlaying={playback.isPlaying}
          speed={playback.speed}
          currentTime={playback.currentTime}
          totalDuration={scenario.totalDurationMs}
          onTogglePlay={playback.togglePlay}
          onStepForward={playback.stepForward}
          onSpeedChange={playback.setSpeed}
          onSeek={playback.seekTo}
          onReset={playback.reset}
          eventTimestamps={scenario.events.map(e => e.timestamp)}
        />
      </div>
    </div>
  )
}

export function Replay() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="demos">
      <SectionLabel number="08" title="Real-World Demos" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Proven across 12 real-world scenarios.</span>{' '}
        <span className="text-gray-500">Step through every event.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-4 leading-relaxed">
        From single-agent tools to distributed multi-machine networks.
        Pick an example, choose a scenario, and watch agents discover, delegate, and
        coordinate in real time.
      </p>
      <p className="text-gray-500 max-w-3xl mb-12">
        Hit play for the full run, or use step-through mode to inspect every tool call,
        delegation, and cross-server message.
      </p>

      <FadeIn>
        <ReplayViewer />
      </FadeIn>
    </AnimatedSection>
  )
}
