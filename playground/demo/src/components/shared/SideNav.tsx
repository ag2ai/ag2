import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const SECTIONS = [
  { id: 'paradigm', label: 'The Crisis', number: '01' },
  { id: 'market', label: 'Landscape', number: '02' },
  { id: 'solution', label: 'Our Answer', number: '03' },
  { id: 'architecture', label: 'Architecture', number: '04' },
  { id: 'concepts', label: 'Deep Dive', number: '05' },
  { id: 'topology', label: 'Topology', number: '06' },
  { id: 'demos', label: 'Demos', number: '07' },
  { id: 'integration', label: 'Integration', number: '08' },
  { id: 'roadmap', label: 'Roadmap', number: '09' },
]

export function SideNav() {
  const [activeId, setActiveId] = useState('')
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the entry with the largest intersection ratio
        const intersecting = entries.filter((e) => e.isIntersecting)
        if (intersecting.length > 0) {
          const best = intersecting.reduce((a, b) =>
            a.intersectionRatio > b.intersectionRatio ? a : b
          )
          setActiveId(best.target.id)
        }
      },
      { rootMargin: '-20% 0px -60% 0px', threshold: [0, 0.25, 0.5, 0.75, 1] }
    )

    // Observe all sections
    for (const s of SECTIONS) {
      const el = document.getElementById(s.id)
      if (el) observer.observe(el)
    }

    // Show nav after scrolling past hero
    const handleScroll = () => {
      setVisible(window.scrollY > window.innerHeight * 0.5)
    }
    handleScroll()
    window.addEventListener('scroll', handleScroll, { passive: true })

    return () => {
      observer.disconnect()
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  const scrollTo = (id: string) => {
    const el = document.getElementById(id)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <AnimatePresence>
      {visible && (
        <motion.nav
          initial={{ opacity: 0, x: -12 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -12 }}
          transition={{ duration: 0.3 }}
          className="fixed left-4 top-1/2 -translate-y-1/2 z-50 hidden lg:flex flex-col gap-1"
        >
          {SECTIONS.map((s) => {
            const isActive = activeId === s.id
            return (
              <button
                key={s.id}
                onClick={() => scrollTo(s.id)}
                className={`group flex items-center gap-2 px-2 py-1.5 rounded-lg transition-all text-left ${
                  isActive
                    ? 'bg-blue-500/10'
                    : 'hover:bg-gray-800/50'
                }`}
                title={s.label}
              >
                <span
                  className={`text-[10px] font-mono font-bold w-5 text-right transition-colors ${
                    isActive ? 'text-blue-400' : 'text-gray-700 group-hover:text-gray-500'
                  }`}
                >
                  {s.number}
                </span>
                <span
                  className={`w-1 h-1 rounded-full transition-all ${
                    isActive
                      ? 'bg-blue-400 scale-150'
                      : 'bg-gray-700 group-hover:bg-gray-500'
                  }`}
                />
                <span
                  className={`text-[11px] font-mono transition-colors whitespace-nowrap ${
                    isActive ? 'text-blue-400' : 'text-gray-600 group-hover:text-gray-400'
                  }`}
                >
                  {s.label}
                </span>
              </button>
            )
          })}
        </motion.nav>
      )}
    </AnimatePresence>
  )
}
