interface HubSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  username: string
  onUsernameChange: (username: string) => void
}

export function HubSettings({ enabled, onEnabledChange, username, onUsernameChange }: HubSettingsProps) {
  return (
    <div className="advanced-group" style={{ borderBottom: 'none' }}>
      <div className="checkbox-row">
        <input
          type="checkbox"
          id="push_to_hub"
          checked={enabled}
          onChange={(e) => onEnabledChange(e.target.checked)}
        />
        <label htmlFor="push_to_hub">Push to HuggingFace Hub</label>
      </div>
      {enabled && (
        <div style={{ marginTop: 'var(--space-xs)' }}>
          <div className="form-field">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => onUsernameChange(e.target.value)}
              placeholder="your-username"
            />
            <span className="field-hint">Requires HF_TOKEN env var</span>
          </div>
        </div>
      )}
    </div>
  )
}
