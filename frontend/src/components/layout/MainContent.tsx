import { useCallback } from 'react'
import { useAppState } from '../../context/AppContext'
import { useRuns } from '../../hooks/useRuns'
import { useMetrics } from '../../hooks/useMetrics'
import { MonitorHeader } from '../monitor/MonitorHeader'
import { StatusBanner } from '../monitor/StatusBanner'
import { MetricsChart } from '../monitor/MetricsChart'
import { LogCard } from '../monitor/LogCard'

export function MainContent() {
  const { currentStep, currentLoss, currentEta, statusMessage, logs, metricsData, currentMetric, activeRunId } = useAppState()
  const { loadRuns } = useRuns()
  const { setCurrentMetric, getAvailableMetrics } = useMetrics()

  const handleRefresh = useCallback(() => {
    loadRuns()
  }, [loadRuns])

  const availableMetrics = getAvailableMetrics()

  return (
    <main className="main">
      <MonitorHeader
        currentStep={currentStep}
        currentLoss={currentLoss}
        currentEta={currentEta}
        onRefresh={handleRefresh}
      />
      <StatusBanner message={statusMessage} />
      <MetricsChart
        metricsData={metricsData}
        currentMetric={currentMetric}
        availableMetrics={availableMetrics}
        onMetricChange={setCurrentMetric}
        isTraining={!!activeRunId || !!statusMessage}
      />
      <LogCard logs={logs} />
    </main>
  )
}
