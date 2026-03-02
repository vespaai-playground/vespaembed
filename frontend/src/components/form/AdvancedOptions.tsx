import { useState, type ReactNode } from 'react'

interface AdvancedOptionsProps {
  children: ReactNode
}

export function AdvancedOptions({ children }: AdvancedOptionsProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="form-section">
      <button
        type="button"
        className="collapsible-header"
        onClick={() => setExpanded(!expanded)}
      >
        <span>Advanced Options</span>
        <span className="collapsible-icon">{expanded ? '▲' : '▼'}</span>
      </button>
      <div className={`collapsible-content ${expanded ? 'expanded' : ''}`}>
        {children}
      </div>
    </div>
  )
}
