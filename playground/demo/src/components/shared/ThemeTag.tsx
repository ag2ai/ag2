const themes = {
  network: { color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/20', icon: '~' },
  distributed: { color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', icon: '::' },
  autonomy: { color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', icon: '*' },
} as const

export function ThemeTag({ theme }: { theme: keyof typeof themes }) {
  const t = themes[theme]
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-mono font-medium px-2 py-0.5 rounded-full border ${t.color} ${t.bg} ${t.border}`}>
      <span>{t.icon}</span>
      {theme}
    </span>
  )
}

export function ThemeLegend() {
  return (
    <div className="flex flex-wrap gap-3 justify-center mb-12">
      <ThemeTag theme="network" />
      <ThemeTag theme="distributed" />
      <ThemeTag theme="autonomy" />
    </div>
  )
}
