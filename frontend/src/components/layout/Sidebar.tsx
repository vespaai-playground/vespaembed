import { useEffect, useCallback } from 'react'
import { useAppDispatch } from '../../context/AppContext'
import { useRuns, parseRunConfig } from '../../hooks/useRuns'
import { useMetrics } from '../../hooks/useMetrics'
import { usePolling } from '../../hooks/usePolling'
import { useTasks } from '../../hooks/useTasks'
import { RunList } from '../sidebar/RunList'
import { ProjectSummary } from '../sidebar/ProjectSummary'
import type { Run } from '../../api/types'

export function Sidebar() {
  const dispatch = useAppDispatch()
  const { runs, selectedRunId, activeRunId, loadRuns, selectRun, handleDeleteRun, handleStopTraining, handleUpdate, loadHistoricalUpdates } = useRuns()
  const { loadMetrics } = useMetrics()
  const { loadTasks } = useTasks()

  const onComplete = useCallback(() => {
    loadRuns()
  }, [loadRuns])

  const { startPolling, stopPolling, pollingRunId } = usePolling(handleUpdate, onComplete, loadMetrics)

  // Initial load
  useEffect(() => {
    const init = async () => {
      await loadTasks()
      const autoSelectId = await loadRuns(true)
      if (autoSelectId) {
        handleSelectRun(autoSelectId)
      }
    }
    init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Start polling when there's an active run
  useEffect(() => {
    if (activeRunId && !pollingRunId.current) {
      handleSelectRun(activeRunId)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRunId])

  const handleSelectRun = useCallback(async (runId: number) => {
    const run = await selectRun(runId)
    if (!run) return

    const nextLine = await loadHistoricalUpdates(runId)
    await loadMetrics(runId)

    if (run.status === 'running') {
      if (pollingRunId.current !== runId) {
        startPolling(runId, nextLine)
      }
    } else {
      if (run.status === 'completed') {
        dispatch({ type: 'SET_CURRENT_ETA', eta: '0' })
      }
    }
  }, [selectRun, loadHistoricalUpdates, loadMetrics, startPolling, pollingRunId, dispatch])

  const handleStop = useCallback(async () => {
    if (selectedRunId) {
      stopPolling()
      await handleStopTraining(selectedRunId)
    }
  }, [selectedRunId, stopPolling, handleStopTraining])

  const handleDelete = useCallback(async () => {
    if (selectedRunId) {
      stopPolling()
      const nextRunId = await handleDeleteRun(selectedRunId)
      if (nextRunId) {
        handleSelectRun(nextRunId)
      }
    }
  }, [selectedRunId, stopPolling, handleDeleteRun, handleSelectRun])

  const handleArtifacts = useCallback(() => {
    dispatch({ type: 'SET_SHOW_ARTIFACTS_MODAL', show: true })
  }, [dispatch])

  const handleCopy = useCallback(() => {
    if (selectedRunId) {
      const run = runs.find(r => r.id === selectedRunId)
      if (run) {
        const config = parseRunConfig(run)
        dispatch({ type: 'COPY_PROJECT', config })
      }
    }
  }, [selectedRunId, runs, dispatch])

  const selectedRun: Run | null = runs.find(r => r.id === selectedRunId) || null

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-icon">&#9672;</span>
        <span className="brand-text">VespaEmbed</span>
      </div>
      <button
        className="btn-new"
        onClick={() => dispatch({ type: 'SET_SHOW_NEW_PROJECT_MODAL', show: true })}
      >
        + New Project
      </button>
      <RunList
        runs={runs}
        selectedRunId={selectedRunId}
        onSelectRun={handleSelectRun}
      />
      <ProjectSummary
        run={selectedRun}
        onArtifacts={handleArtifacts}
        onCopy={handleCopy}
        onStop={handleStop}
        onDelete={handleDelete}
      />
    </aside>
  )
}
