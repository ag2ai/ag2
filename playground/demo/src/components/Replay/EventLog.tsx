import { useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import type { ReplayEvent } from '../../data/replay-types'

const EVENT_CONFIG: Record<string, { icon: string; label: string; color: string; bgColor: string }> = {
  'tool-call':          { icon: '⚡', label: 'TOOL',      color: 'text-amber-400',   bgColor: 'bg-amber-500/10 border-amber-500/20' },
  'tool-result':        { icon: '←',  label: 'RESULT',    color: 'text-gray-500',    bgColor: 'bg-gray-800/50 border-gray-700/30' },
  'delegation-request': { icon: '→',  label: 'DELEGATE',  color: 'text-violet-400',  bgColor: 'bg-violet-500/10 border-violet-500/20' },
  'delegation-result':  { icon: '✓',  label: 'RETURNED',  color: 'text-violet-300',  bgColor: 'bg-violet-500/5 border-violet-500/10' },
  'model-response':     { icon: '●',  label: 'RESPONSE',  color: 'text-emerald-400', bgColor: 'bg-emerald-500/10 border-emerald-500/20' },
  'discover-agents':    { icon: '◎',  label: 'DISCOVER',  color: 'text-cyan-400',    bgColor: 'bg-cyan-500/10 border-cyan-500/20' },
  'hub-connect':        { icon: '⬡',  label: 'CONNECT',   color: 'text-blue-400',    bgColor: 'bg-blue-500/10 border-blue-500/20' },
  'scheduler-trigger':  { icon: '◷',  label: 'SCHEDULE',  color: 'text-cyan-400',    bgColor: 'bg-cyan-500/10 border-cyan-500/20' },
}

function formatTime(ms: number): string {
  const s = Math.floor(ms / 1000)
  const frac = Math.floor((ms % 1000) / 100)
  return `${s}.${frac}s`
}

function getEventSummary(event: ReplayEvent): string {
  switch (event.type) {
    case 'tool-call':
      return `${event.agent}.${event.toolName}(${event.args})`
    case 'tool-result':
      return event.result.length > 100 ? event.result.slice(0, 100) + '…' : event.result
    case 'delegation-request':
      return `${event.source} → ${event.target}${event.channel === 'http' ? ' [HTTP]' : ''}: ${event.taskPreview.slice(0, 80)}…`
    case 'delegation-result':
      return `${event.target} → ${event.source}: ${event.resultPreview.slice(0, 80)}…`
    case 'model-response':
      return event.contentPreview.slice(0, 120) + (event.contentPreview.length > 120 ? '…' : '')
    case 'discover-agents':
      return `${event.agent} discovers [${event.results.join(', ')}] for "${event.capability}"`
    case 'hub-connect':
      return `Connected to ${event.targetServer}: found [${event.agentsDiscovered.join(', ')}]`
    case 'scheduler-trigger':
      return `${event.watch} → ${event.target}: ${event.task}`
    default:
      return ''
  }
}

interface EventLogProps {
  events: ReplayEvent[]
}

export function EventLog({ events }: EventLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [events.length])

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto space-y-1 pr-1">
      <AnimatePresence initial={false}>
        {events.map(event => {
          const cfg = EVENT_CONFIG[event.type] || EVENT_CONFIG['tool-call']
          return (
            <motion.div
              key={event.id}
              initial={{ opacity: 0, x: 10, height: 0 }}
              animate={{ opacity: 1, x: 0, height: 'auto' }}
              transition={{ duration: 0.15 }}
              className={`rounded-lg border px-2.5 py-1.5 ${cfg.bgColor}`}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-[10px] font-mono text-gray-600 w-8 shrink-0">
                  {formatTime(event.timestamp)}
                </span>
                <span className={`text-[9px] font-mono font-bold ${cfg.color} shrink-0`}>
                  {cfg.icon} {cfg.label}
                </span>
                <span className="text-[10px] font-mono text-gray-500 truncate">
                  {event.agent}
                </span>
              </div>
              <p className="text-[11px] text-gray-400 leading-tight pl-10 break-words">
                {getEventSummary(event)}
              </p>
            </motion.div>
          )
        })}
      </AnimatePresence>
      {events.length === 0 && (
        <div className="flex items-center justify-center h-full text-gray-600 text-xs font-mono">
          Press play to start replay
        </div>
      )}
    </div>
  )
}
