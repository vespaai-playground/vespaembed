interface LoraSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  loraR: string
  loraAlpha: string
  loraDropout: string
  loraTargetModules: string
  loraTargetPreset: string
  onLoraRChange: (v: string) => void
  onLoraAlphaChange: (v: string) => void
  onLoraDropoutChange: (v: string) => void
  onLoraTargetModulesChange: (v: string) => void
  onLoraTargetPresetChange: (v: string) => void
}

export function LoraSettings(props: LoraSettingsProps) {
  return (
    <div className="advanced-group">
      <div className="checkbox-row">
        <input
          type="checkbox"
          id="lora_enabled"
          checked={props.enabled}
          onChange={(e) => props.onEnabledChange(e.target.checked)}
        />
        <label htmlFor="lora_enabled">Enable LoRA training</label>
      </div>
      {props.enabled && (
        <div style={{ marginTop: 'var(--space-xs)' }}>
          <div className="form-row">
            <div className="form-field compact">
              <label>Rank (r)</label>
              <select value={props.loraR} onChange={(e) => props.onLoraRChange(e.target.value)}>
                <option value="8">8</option>
                <option value="16">16</option>
                <option value="32">32</option>
                <option value="64">64</option>
                <option value="128">128</option>
              </select>
            </div>
            <div className="form-field compact">
              <label>Alpha</label>
              <input type="number" value={props.loraAlpha} onChange={(e) => props.onLoraAlphaChange(e.target.value)} min="1" />
            </div>
            <div className="form-field compact">
              <label>Dropout</label>
              <input type="text" value={props.loraDropout} onChange={(e) => props.onLoraDropoutChange(e.target.value)} />
            </div>
          </div>
          <div className="form-field">
            <label>Target Modules</label>
            <select
              className="target-modules-select"
              value={props.loraTargetPreset}
              onChange={(e) => {
                props.onLoraTargetPresetChange(e.target.value)
                props.onLoraTargetModulesChange(e.target.value)
              }}
            >
              <option value="query, key, value, dense">BERT / MiniLM / MPNet / BGE</option>
              <option value="Wqkv, Wi, Wo">ModernBERT / GTE-ModernBERT</option>
              <option value="q_proj, k_proj, v_proj, o_proj">Llama / Gemma / Qwen</option>
              <option value="q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj">Llama / Gemma / Qwen (extended)</option>
              <option value="">Custom</option>
            </select>
            <input
              type="text"
              value={props.loraTargetModules}
              onChange={(e) => props.onLoraTargetModulesChange(e.target.value)}
              style={{ marginTop: 'var(--space-xs)' }}
              placeholder="Enter comma-separated module names"
            />
          </div>
        </div>
      )}
    </div>
  )
}
