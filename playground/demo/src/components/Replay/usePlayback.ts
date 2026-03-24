import { useState, useRef, useCallback, useEffect } from 'react'
import type { ReplayEvent, DelegationRequestEvent, DelegationResultEvent } from '../../data/replay-types'

export interface PlaybackState {
  isPlaying: boolean
  speed: number
  currentTime: number
  visibleEvents: ReplayEvent[]
  activeAgent: string | null
  activeDelegations: Array<{ source: string; target: string; channel: 'local' | 'http' }>
  play: () => void
  pause: () => void
  togglePlay: () => void
  setSpeed: (s: number) => void
  stepForward: () => void
  seekTo: (time: number) => void
  reset: () => void
}

export function usePlayback(events: ReplayEvent[], totalDurationMs: number): PlaybackState {
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeedState] = useState(1)
  const [currentTime, setCurrentTime] = useState(0)
  const lastFrameRef = useRef<number>(0)
  const rafRef = useRef<number>(0)

  // Animation loop
  const tick = useCallback(() => {
    const now = performance.now()
    const delta = now - lastFrameRef.current
    lastFrameRef.current = now

    setCurrentTime(prev => {
      const next = prev + delta * speed
      if (next >= totalDurationMs) {
        setIsPlaying(false)
        return totalDurationMs
      }
      return next
    })

    rafRef.current = requestAnimationFrame(tick)
  }, [speed, totalDurationMs])

  useEffect(() => {
    if (isPlaying) {
      lastFrameRef.current = performance.now()
      rafRef.current = requestAnimationFrame(tick)
    } else {
      cancelAnimationFrame(rafRef.current)
    }
    return () => cancelAnimationFrame(rafRef.current)
  }, [isPlaying, tick])

  // Derived state
  const visibleEvents = events.filter(e => e.timestamp <= currentTime)

  // Find the most recently active agent
  const lastEvent = visibleEvents.length > 0 ? visibleEvents[visibleEvents.length - 1] : null
  const activeAgent = lastEvent?.agent ?? null

  // Find active (in-flight) delegations
  const activeDelegations: Array<{ source: string; target: string; channel: 'local' | 'http' }> = []
  const delegationRequests = visibleEvents.filter((e): e is DelegationRequestEvent => e.type === 'delegation-request')
  const delegationResults = visibleEvents.filter((e): e is DelegationResultEvent => e.type === 'delegation-result')

  for (const req of delegationRequests) {
    const hasResult = delegationResults.some(
      r => r.source === req.source && r.target === req.target
    )
    if (!hasResult) {
      activeDelegations.push({ source: req.source, target: req.target, channel: req.channel })
    }
  }

  const play = useCallback(() => {
    if (currentTime >= totalDurationMs) {
      setCurrentTime(0)
    }
    setIsPlaying(true)
  }, [currentTime, totalDurationMs])

  const pause = useCallback(() => setIsPlaying(false), [])

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause()
    } else {
      play()
    }
  }, [isPlaying, play, pause])

  const setSpeed = useCallback((s: number) => setSpeedState(s), [])

  const stepForward = useCallback(() => {
    setIsPlaying(false)
    const nextEvent = events.find(e => e.timestamp > currentTime)
    if (nextEvent) {
      setCurrentTime(nextEvent.timestamp)
    } else {
      setCurrentTime(totalDurationMs)
    }
  }, [events, currentTime, totalDurationMs])

  const seekTo = useCallback((time: number) => {
    setCurrentTime(Math.max(0, Math.min(time, totalDurationMs)))
  }, [totalDurationMs])

  const reset = useCallback(() => {
    setIsPlaying(false)
    setCurrentTime(0)
  }, [])

  return {
    isPlaying,
    speed,
    currentTime,
    visibleEvents,
    activeAgent,
    activeDelegations,
    play,
    pause,
    togglePlay,
    setSpeed,
    stepForward,
    seekTo,
    reset,
  }
}
