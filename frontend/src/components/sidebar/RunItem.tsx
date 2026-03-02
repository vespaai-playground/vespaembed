import type { Run } from '../../api/types'
import { parseRunConfig } from '../../hooks/useRuns'

interface RunItemProps {
  run: Run
  isSelected: boolean
  onClick: () => void
}

export function RunItem({ run, isSelected, onClick }: RunItemProps) {
  const config = parseRunConfig(run)
  const date = new Date(run.created_at).toLocaleDateString()
  const projectName = config.project_name || `Run #${run.id}`

  return (
    <div
      className={`run-item ${isSelected ? 'active' : ''}`}
      onClick={onClick}
    >
      <span className={`run-status-icon ${run.status}`} />
      <div className="run-info">
        <div className="run-id">{projectName}</div>
        <div className="run-date">{date}</div>
      </div>
    </div>
  )
}
