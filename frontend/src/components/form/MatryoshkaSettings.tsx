interface MatryoshkaSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  dims: string
  onDimsChange: (dims: string) => void
  visible: boolean
}

export function MatryoshkaSettings({ enabled, onEnabledChange, dims, onDimsChange, visible }: MatryoshkaSettingsProps) {
  if (!visible) return null

  return (
    <div className="advanced-group">
      <div className="checkbox-row">
        <input
          type="checkbox"
          id="matryoshka_enabled"
          checked={enabled}
          onChange={(e) => onEnabledChange(e.target.checked)}
        />
        <label htmlFor="matryoshka_enabled">Enable Matryoshka embeddings</label>
      </div>
      {enabled && (
        <div style={{ marginTop: 'var(--space-xs)' }}>
          <div className="form-field">
            <label>Dimensions</label>
            <input
              type="text"
              value={dims}
              onChange={(e) => onDimsChange(e.target.value)}
              placeholder="768,512,256,128,64"
            />
            <span className="field-hint">Comma-separated (largest to smallest)</span>
          </div>
        </div>
      )}
    </div>
  )
}
