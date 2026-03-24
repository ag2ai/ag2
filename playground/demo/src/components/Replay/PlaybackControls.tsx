interface PlaybackControlsProps {
  isPlaying: boolean
  speed: number
  currentTime: number
  totalDuration: number
  onTogglePlay: () => void
  onStepForward: () => void
  onSpeedChange: (speed: number) => void
  onSeek: (time: number) => void
  onReset: () => void
  eventTimestamps: number[]
}

function formatTime(ms: number): string {
  const s = Math.floor(ms / 1000)
  const frac = Math.floor((ms % 1000) / 100)
  return `${s}.${frac}s`
}

export function PlaybackControls({
  isPlaying, speed, currentTime, totalDuration,
  onTogglePlay, onStepForward, onSpeedChange, onSeek, onReset,
  eventTimestamps,
}: PlaybackControlsProps) {
  const progress = totalDuration > 0 ? (currentTime / totalDuration) * 100 : 0

  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const pct = (e.clientX - rect.left) / rect.width
    onSeek(pct * totalDuration)
  }

  return (
    <div className="space-y-2">
      {/* Timeline bar */}
      <div
        className="relative h-6 cursor-pointer group"
        onClick={handleTimelineClick}
      >
        {/* Track */}
        <div className="absolute top-2.5 left-0 right-0 h-1.5 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Event tick marks */}
        {eventTimestamps.map((ts, i) => (
          <div
            key={i}
            className="absolute top-1 w-0.5 h-4 rounded-full bg-gray-600 group-hover:bg-gray-500 transition-colors"
            style={{ left: `${(ts / totalDuration) * 100}%` }}
          />
        ))}

        {/* Playhead */}
        <div
          className="absolute top-1 w-2.5 h-4 rounded-sm bg-white shadow-lg"
          style={{ left: `calc(${progress}% - 5px)` }}
        />
      </div>

      {/* Controls row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          {/* Reset */}
          <button onClick={onReset}
            className="w-7 h-7 rounded-md bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-gray-400 hover:text-white transition-colors text-xs font-mono"
            title="Reset"
          >
            ↺
          </button>

          {/* Play / Pause */}
          <button onClick={onTogglePlay}
            className="w-8 h-7 rounded-md bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 flex items-center justify-center text-blue-400 transition-colors text-sm"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '❚❚' : '▶'}
          </button>

          {/* Step forward */}
          <button onClick={onStepForward}
            className="w-7 h-7 rounded-md bg-gray-800 hover:bg-gray-700 flex items-center justify-center text-gray-400 hover:text-white transition-colors text-xs font-mono"
            title="Step to next event"
          >
            ▸|
          </button>
        </div>

        {/* Time display */}
        <span className="text-[11px] font-mono text-gray-500">
          {formatTime(currentTime)} / {formatTime(totalDuration)}
        </span>

        {/* Speed selector */}
        <div className="flex items-center gap-1">
          {[0.5, 1, 2, 4].map(s => (
            <button
              key={s}
              onClick={() => onSpeedChange(s)}
              className={`px-1.5 py-0.5 rounded text-[10px] font-mono transition-all ${
                speed === s
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                  : 'text-gray-600 hover:text-gray-400'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
