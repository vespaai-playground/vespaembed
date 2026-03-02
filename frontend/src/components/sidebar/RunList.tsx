import type { Run } from '../../api/types'
import { RunItem } from './RunItem'

interface RunListProps {
  runs: Run[]
  selectedRunId: number | null
  onSelectRun: (runId: number) => void
}

export function RunList({ runs, selectedRunId, onSelectRun }: RunListProps) {
  if (runs.length === 0) {
    return (
      <div className="run-list">
        <div className="run-item placeholder">No training runs yet</div>
      </div>
    )
  }

  return (
    <div className="run-list">
      {runs.map(run => (
        <RunItem
          key={run.id}
          run={run}
          isSelected={run.id === selectedRunId}
          onClick={() => onSelectRun(run.id)}
        />
      ))}
    </div>
  )
}
