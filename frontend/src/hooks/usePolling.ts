import { useRef, useCallback, useEffect } from 'react'
import { useAppDispatch } from '../context/AppContext'
import * as api from '../api/client'
import type { UpdateEntry } from '../api/types'

export function usePolling(
  onUpdate: (update: UpdateEntry) => void,
  onComplete: () => void,
  loadMetrics: (runId: number) => Promise<void>,
) {
  const dispatch = useAppDispatch()
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const pollingRunIdRef = useRef<number | null>(null)
  const pollLineRef = useRef(0)

  // Store callbacks in refs so the interval always uses the latest versions
  const onUpdateRef = useRef(onUpdate)
  onUpdateRef.current = onUpdate
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete
  const loadMetricsRef = useRef(loadMetrics)
  loadMetricsRef.current = loadMetrics

  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    pollingRunIdRef.current = null
  }, [])

  const startPolling = useCallback((runId: number, fromLine = 0) => {
    stopPolling()
    pollingRunIdRef.current = runId
    pollLineRef.current = fromLine

    intervalRef.current = setInterval(async () => {
      const currentPollingId = pollingRunIdRef.current
      if (!currentPollingId) return

      try {
        const data = await api.fetchUpdates(currentPollingId, pollLineRef.current)

        data.updates.forEach(update => {
          onUpdateRef.current(update)
        })

        await loadMetricsRef.current(currentPollingId)

        pollLineRef.current = data.next_line
        dispatch({ type: 'SET_POLL_LINE', line: data.next_line })

        const isTerminal = data.run_status === 'completed' || data.run_status === 'error' || data.run_status === 'stopped'
        if (isTerminal || !data.has_more) {
          stopPolling()
          onCompleteRef.current()
          await loadMetricsRef.current(currentPollingId)
        }
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, 2000)
  }, [dispatch, stopPolling])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling()
    }
  }, [stopPolling])

  return { startPolling, stopPolling, pollingRunId: pollingRunIdRef }
}
