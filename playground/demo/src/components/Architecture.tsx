import { motion } from 'framer-motion'
import { AnimatedSection, FadeIn } from './shared/AnimatedSection'
import { SectionLabel } from './shared/SectionLabel'

const layers = [
  {
    number: 5,
    name: 'APPLICATION',
    desc: 'Your code: custom actors, observers, plugins, topologies',
    color: 'from-violet-500 to-purple-500',
    borderColor: 'border-violet-500/30',
    bgColor: 'bg-violet-500/5',
    textColor: 'text-violet-400',
    items: ['Custom Actors', 'Domain Observers', 'Business Plugins'],
  },
  {
    number: 4,
    name: 'COMPOSITION',
    desc: 'Composable routing and wiring patterns',
    color: 'from-pink-500 to-rose-500',
    borderColor: 'border-pink-500/30',
    bgColor: 'bg-pink-500/5',
    textColor: 'text-pink-400',
    items: ['Pipeline', 'Fanout', 'Conditional', 'RouteDecision', 'Network'],
  },
  {
    number: 3,
    name: 'BUILDING BLOCKS',
    desc: 'Independently useful, optionally composable components',
    color: 'from-blue-500 to-cyan-500',
    borderColor: 'border-blue-500/30',
    bgColor: 'bg-blue-500/5',
    textColor: 'text-blue-400',
    items: ['Actor', 'Hub', 'Observer', 'Scheduler', 'Assembler', 'AssemblyPolicies'],
  },
  {
    number: 2,
    name: 'PRIMITIVES',
    desc: 'Low-level, composable, zero cross-dependencies',
    color: 'from-emerald-500 to-teal-500',
    borderColor: 'border-emerald-500/30',
    bgColor: 'bg-emerald-500/5',
    textColor: 'text-emerald-400',
    items: ['Watch', 'Signal', 'Priority', 'Channel', 'Envelope', 'KnowledgeStore', 'CompactStrategy', 'AggregateStrategy'],
  },
  {
    number: 1,
    name: 'FRAMEWORK CORE',
    desc: 'Solid foundation. Zero changes needed.',
    color: 'from-gray-500 to-gray-600',
    borderColor: 'border-gray-600/30',
    bgColor: 'bg-gray-800/30',
    textColor: 'text-gray-400',
    items: ['Agent', 'Stream', 'Events', 'Tools', 'Middleware'],
  },
]

export function Architecture() {
  return (
    <AnimatedSection className="py-32 px-6 max-w-6xl mx-auto" id="architecture">
      <SectionLabel number="04" title="Architecture" />

      <h2 className="text-4xl md:text-5xl font-bold tracking-tight mb-6">
        <span className="text-white">Five layers.</span>{' '}
        <span className="text-gray-500">Each independent.</span>
      </h2>
      <p className="text-xl text-gray-400 max-w-3xl mb-16 leading-relaxed">
        Each layer builds on the previous. Each can be used independently.
        The design principle is <span className="text-white">additive, not invasive</span> —
        zero changes to the core framework.
      </p>

      {/* Layer stack */}
      <div className="max-w-3xl mx-auto space-y-3">
        {layers.map((layer, i) => (
          <motion.div
            key={layer.number}
            initial={{ opacity: 0, x: -40, scale: 0.95 }}
            whileInView={{ opacity: 1, x: 0, scale: 1 }}
            viewport={{ once: true, margin: '-20px' }}
            transition={{
              duration: 0.5,
              delay: (layers.length - 1 - i) * 0.12,
              ease: [0.25, 0.4, 0.25, 1],
            }}
            className={`relative p-5 rounded-xl border ${layer.borderColor} ${layer.bgColor} group hover:scale-[1.02] transition-transform duration-300`}
          >
            <div className="flex items-start gap-4">
              {/* Layer number */}
              <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${layer.color} flex items-center justify-center shrink-0 shadow-lg`}>
                <span className="text-white text-sm font-bold font-mono">{layer.number}</span>
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-3 mb-1">
                  <h3 className={`text-sm font-bold font-mono tracking-wider ${layer.textColor}`}>
                    {layer.name}
                  </h3>
                  {layer.number === 1 && (
                    <span className="text-[10px] text-gray-500 bg-gray-800 px-2 py-0.5 rounded font-mono">
                      EXISTING
                    </span>
                  )}
                  {layer.number >= 2 && layer.number <= 4 && (
                    <span className="text-[10px] text-emerald-500 bg-emerald-500/10 px-2 py-0.5 rounded font-mono">
                      IMPLEMENTED
                    </span>
                  )}
                </div>
                <p className="text-sm text-gray-500 mb-3">{layer.desc}</p>

                <div className="flex flex-wrap gap-2">
                  {layer.items.map(item => (
                    <span
                      key={item}
                      className={`text-xs font-mono px-2 py-1 rounded ${layer.bgColor} ${layer.textColor} border ${layer.borderColor}`}
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Connector arrow */}
            {i < layers.length - 1 && (
              <div className="absolute -bottom-3 left-9 z-10">
                <svg width="12" height="12" viewBox="0 0 12 12" className="text-gray-700">
                  <path d="M6 0v12M3 9l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Key insight callout */}
      <FadeIn className="mt-16 text-center">
        <div className="inline-flex flex-col items-center gap-2 p-6 rounded-2xl bg-gray-900/50 border border-gray-800">
          <p className="text-gray-400 text-sm max-w-lg leading-relaxed">
            An <span className="text-blue-400 font-mono">Actor</span> IS an <span className="text-gray-300 font-mono">Agent</span> with extras.
            A <span className="text-blue-400 font-mono">Channel</span> wraps a <span className="text-gray-300 font-mono">Stream</span>.
            Everything composes — nothing replaces.
          </p>
        </div>
      </FadeIn>
    </AnimatedSection>
  )
}
