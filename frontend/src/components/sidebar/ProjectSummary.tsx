import type { Run } from '../../api/types'
import { parseRunConfig } from '../../hooks/useRuns'
import { StatusChip } from '../ui/StatusChip'

interface ProjectSummaryProps {
  run: Run | null
  onArtifacts: () => void
  onCopy: () => void
  onStop: () => void
  onDelete: () => void
}

export function ProjectSummary({ run, onArtifacts, onCopy, onStop, onDelete }: ProjectSummaryProps) {
  if (!run) return null

  const config = parseRunConfig(run)

  let dataSource = '--'
  if (config.train_filename) {
    dataSource = config.train_filename.split('/').pop() || '--'
  } else if (config.hf_dataset) {
    dataSource = config.hf_dataset
  }

  const artifactsDisabled = run.status === 'running' || run.status === 'pending'

  return (
    <div className="project-summary" style={{ display: 'block' }}>
      <div className="summary-header">
        <span className="summary-title">{config.project_name || `Run #${run.id}`}</span>
        <StatusChip status={run.status} small />
      </div>
      <div className="summary-grid">
        <SummaryRow label="Task" value={config.task || '--'} />
        <SummaryRow label="Loss" value={config.loss_variant || 'default'} />
        <SummaryRow label="Model" value={config.base_model?.split('/').pop() || '--'} />
        <SummaryRow label="Data" value={dataSource} />
        <SummaryRow label="Epochs" value={config.epochs ?? '--'} />
        <SummaryRow label="Batch" value={config.batch_size ?? '--'} />
        <SummaryRow label="LR" value={config.learning_rate ?? '--'} />
        {config.lora_enabled && (
          <SummaryRow label="LoRA" value={`r=${config.lora_r}, a=${config.lora_alpha}`} />
        )}
        {config.matryoshka_dims && (
          <SummaryRow label="Matryoshka" value={config.matryoshka_dims} />
        )}
      </div>
      {run.status === 'error' && run.error_message && (
        <div className="summary-error">
          {run.error_message}
        </div>
      )}
      <div className="summary-actions">
        <button
          type="button"
          className="btn-ghost"
          disabled={artifactsDisabled}
          onClick={onArtifacts}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Artifacts
        </button>
        <button type="button" className="btn-ghost" onClick={onCopy}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          Copy
        </button>
        {run.status === 'running' ? (
          <button type="button" className="btn-danger" onClick={onStop}>
            <span className="btn-text">Stop</span>
          </button>
        ) : (
          <button type="button" className="btn-ghost" onClick={onDelete}>Delete</button>
        )}
      </div>
    </div>
  )
}

function SummaryRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="summary-row">
      <span className="summary-key">{label}</span>
      <span className="summary-val">{String(value)}</span>
    </div>
  )
}
