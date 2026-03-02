import { useCallback, useRef } from 'react'
import { useAppState, useAppDispatch } from '../context/AppContext'
import * as api from '../api/client'
import type { Run, UpdateEntry } from '../api/types'

export function useRuns() {
  const state = useAppState()
  const dispatch = useAppDispatch()
  const selectedRunIdRef = useRef(state.selectedRunId)
  selectedRunIdRef.current = state.selectedRunId

  const loadRuns = useCallback(async (autoSelectLatest = false) => {
    try {
      const runs = await api.fetchRuns()
      dispatch({ type: 'SET_RUNS', runs })

      const activeData = await api.fetchActiveRunId()
      dispatch({ type: 'SET_ACTIVE_RUN_ID', id: activeData.run_id })

      if (autoSelectLatest && runs.length > 0 && !selectedRunIdRef.current) {
        return runs[0].id // caller should call selectRun
      }
    } catch (error) {
      console.error('Failed to load runs:', error)
    }
    return null
  }, [dispatch])

  const selectRun = useCallback(async (runId: number) => {
    dispatch({ type: 'SET_SELECTED_RUN_ID', id: runId })
    dispatch({ type: 'RESET_MONITOR' })
    dispatch({ type: 'SET_POLL_LINE', line: 0 })

    try {
      const run = await api.fetchRun(runId)
      return run
    } catch (error) {
      console.error('Failed to load run:', error)
      return null
    }
  }, [dispatch])

  const handleDeleteRun = useCallback(async (runId: number): Promise<number | null> => {
    if (!confirm('Are you sure you want to delete this run?')) return null
    try {
      await api.deleteRun(runId)
      dispatch({ type: 'SET_SELECTED_RUN_ID', id: null })
      dispatch({ type: 'RESET_MONITOR' })
      const runs = await api.fetchRuns()
      dispatch({ type: 'SET_RUNS', runs })
      const activeData = await api.fetchActiveRunId()
      dispatch({ type: 'SET_ACTIVE_RUN_ID', id: activeData.run_id })
      // Return the latest run ID so caller can select it
      return runs.length > 0 ? runs[0].id : null
    } catch (error) {
      console.error('Delete error:', error)
      return null
    }
  }, [dispatch])

  const handleStopTraining = useCallback(async (runId: number) => {
    try {
      await api.stopTraining(runId)
      dispatch({ type: 'SET_STATUS_MESSAGE', message: null })
      dispatch({ type: 'UPDATE_RUN_STATUS', runId, status: 'stopped' })
      await loadRuns()
    } catch (error) {
      console.error('Stop error:', error)
    }
  }, [dispatch, loadRuns])

  const handleUpdate = useCallback((update: UpdateEntry) => {
    switch (update.type) {
      case 'log':
        if (update.message) {
          dispatch({ type: 'APPEND_LOG', message: update.message })
        }
        break
      case 'progress': {
        const totalSteps = update.total_steps || 0
        const step = update.step || 0
        const stepDisplay = totalSteps
          ? `${step}/${totalSteps}`
          : String(step)
        dispatch({ type: 'SET_CURRENT_STEP', step: stepDisplay })

        const hasLoss = update.loss != null
        if (hasLoss) {
          dispatch({ type: 'SET_CURRENT_LOSS', loss: update.loss!.toFixed(4) })
        }

        if (update.eta_seconds && update.eta_seconds > 0) {
          dispatch({ type: 'SET_CURRENT_ETA', eta: formatTime(update.eta_seconds) })
        }
        break
      }
      case 'train_end':
        // handled by 'complete' which follows
        break
      case 'status':
        if (update.run_id !== undefined && update.status) {
          dispatch({ type: 'UPDATE_RUN_STATUS', runId: update.run_id, status: update.status })
        }
        break
      case 'complete':
        dispatch({ type: 'APPEND_LOG', message: `Training completed! Model saved to: ${update.output_dir}` })
        dispatch({ type: 'SET_STATUS_MESSAGE', message: null })
        if (update.run_id !== undefined) {
          dispatch({ type: 'UPDATE_RUN_STATUS', runId: update.run_id, status: 'completed' })
        }
        break
      case 'error':
        dispatch({ type: 'APPEND_LOG', message: `Error: ${update.message}` })
        dispatch({ type: 'SET_STATUS_MESSAGE', message: null })
        if (update.run_id !== undefined) {
          dispatch({ type: 'UPDATE_RUN_STATUS', runId: update.run_id, status: 'error' })
        }
        break
    }
  }, [dispatch])

  const loadHistoricalUpdates = useCallback(async (runId: number): Promise<number> => {
    try {
      const data = await api.fetchUpdates(runId, 0)
      data.updates.forEach(update => {
        if (update.type !== 'status') {
          handleUpdate(update)
        }
      })
      dispatch({ type: 'SET_POLL_LINE', line: data.next_line })
      return data.next_line
    } catch (error) {
      console.error('Failed to load historical updates:', error)
      return 0
    }
  }, [dispatch, handleUpdate])

  return {
    runs: state.runs,
    activeRunId: state.activeRunId,
    selectedRunId: state.selectedRunId,
    loadRuns,
    selectRun,
    handleDeleteRun,
    handleStopTraining,
    handleUpdate,
    loadHistoricalUpdates,
  }
}

function formatTime(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}s`
  } else if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.round(seconds % 60)
    return `${mins}m ${secs}s`
  } else {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${mins}m`
  }
}

export function parseRunConfig(run: Run) {
  try {
    return JSON.parse(run.config || '{}')
  } catch {
    return {}
  }
}
