interface MonitorHeaderProps {
  currentStep: string
  currentLoss: string
  currentEta: string
  onRefresh: () => void
}

export function MonitorHeader({ currentStep, currentLoss, currentEta, onRefresh }: MonitorHeaderProps) {
  return (
    <div className="monitor-header">
      <div className="header-actions">
        <button className="icon-btn" title="Refresh" onClick={onRefresh}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M23 4v6h-6" />
            <path d="M1 20v-6h6" />
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
          </svg>
        </button>
      </div>
      <div className="metrics-bar">
        <div className="metric">
          <span className="metric-val">{currentStep}</span>
          <span className="metric-label">steps</span>
        </div>
        <div className="metric highlight">
          <span className="metric-val">{currentLoss}</span>
          <span className="metric-label">loss</span>
        </div>
        <div className="metric">
          <span className="metric-val">{currentEta}</span>
          <span className="metric-label">ETA</span>
        </div>
      </div>
    </div>
  )
}
