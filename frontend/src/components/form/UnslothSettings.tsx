interface UnslothSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  saveMethod: string
  onSaveMethodChange: (method: string) => void
}

export function UnslothSettings({ enabled, onEnabledChange, saveMethod, onSaveMethodChange }: UnslothSettingsProps) {
  return (
    <div className="advanced-group">
      <div className="checkbox-row">
        <input
          type="checkbox"
          id="unsloth_enabled"
          checked={enabled}
          onChange={(e) => onEnabledChange(e.target.checked)}
        />
        <label htmlFor="unsloth_enabled">Enable Unsloth (faster training, requires CUDA)</label>
      </div>
      {enabled && (
        <div style={{ marginTop: 'var(--space-xs)' }}>
          <div className="form-field compact">
            <label>Save Method</label>
            <select value={saveMethod} onChange={(e) => onSaveMethodChange(e.target.value)}>
              <option value="merged_16bit">Merged FP16</option>
              <option value="merged_4bit">Merged 4-bit</option>
              <option value="lora">LoRA Only</option>
            </select>
          </div>
        </div>
      )}
    </div>
  )
}
