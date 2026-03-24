import { motion } from 'framer-motion'

export function SectionLabel({ number, title }: { number: string; title: string }) {
  return (
    <motion.div
      className="flex items-center gap-3 mb-8"
      initial={{ opacity: 0, x: -20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
    >
      <span className="text-sm font-mono font-bold text-blue-400 bg-blue-500/10 px-3 py-1 rounded-full border border-blue-500/20">
        {number}
      </span>
      <span className="text-sm font-medium uppercase tracking-widest text-gray-500">
        {title}
      </span>
    </motion.div>
  )
}
