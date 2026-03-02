import { AdvancedOptions } from '../form/AdvancedOptions'
import { LoraSettings } from '../form/LoraSettings'
import { MatryoshkaSettings } from '../form/MatryoshkaSettings'
import { UnslothSettings } from '../form/UnslothSettings'
import { HubSettings } from '../form/HubSettings'
import type { TaskInfo } from '../../api/types'

interface LossInfo {
  lossVariant: string
  onLossVariantChange: (v: string) => void
}

interface TrainingParams {
  epochs: string
  maxSteps: string
  batchSize: string
  learningRate: string
  precision: string
  maxSeqLength: string
  onEpochsChange: (v: string) => void
  onMaxStepsChange: (v: string) => void
  onBatchSizeChange: (v: string) => void
  onLearningRateChange: (v: string) => void
  onPrecisionChange: (v: string) => void
  onMaxSeqLengthChange: (v: string) => void
}

interface AdvancedParams {
  optimizer: string
  scheduler: string
  weightDecay: string
  warmupRatio: string
  gradAccum: string
  loggingSteps: string
  evalSteps: string
  saveSteps: string
  gradientCheckpointing: boolean
  onOptimizerChange: (v: string) => void
  onSchedulerChange: (v: string) => void
  onWeightDecayChange: (v: string) => void
  onWarmupRatioChange: (v: string) => void
  onGradAccumChange: (v: string) => void
  onLoggingStepsChange: (v: string) => void
  onEvalStepsChange: (v: string) => void
  onSaveStepsChange: (v: string) => void
  onGradientCheckpointingChange: (v: boolean) => void
}

interface LoraParams {
  enabled: boolean
  loraR: string
  loraAlpha: string
  loraDropout: string
  loraTargetModules: string
  loraTargetPreset: string
  onEnabledChange: (v: boolean) => void
  onLoraRChange: (v: string) => void
  onLoraAlphaChange: (v: string) => void
  onLoraDropoutChange: (v: string) => void
  onLoraTargetModulesChange: (v: string) => void
  onLoraTargetPresetChange: (v: string) => void
}

interface MatryoshkaParams {
  enabled: boolean
  dims: string
  visible: boolean
  onEnabledChange: (v: boolean) => void
  onDimsChange: (v: string) => void
}

interface UnslothParams {
  enabled: boolean
  saveMethod: string
  onEnabledChange: (v: boolean) => void
  onSaveMethodChange: (v: string) => void
}

interface HubParams {
  enabled: boolean
  username: string
  onEnabledChange: (v: boolean) => void
  onUsernameChange: (v: string) => void
}

interface WizardStep2Props {
  task: TaskInfo | undefined
  loss: LossInfo
  training: TrainingParams
  advanced: AdvancedParams
  lora: LoraParams
  matryoshka: MatryoshkaParams
  unsloth: UnslothParams
  hub: HubParams
  taskSpecificValues: Record<string, string>
  onTaskSpecificChange: (key: string, value: string) => void
}

const LOSS_LABELS: Record<string, string> = {
  'mnr': 'MNR (Multiple Negatives Ranking)',
  'mnr_symmetric': 'MNR Symmetric',
  'gist': 'GIST (Guided In-Sample Triplet)',
  'cached_mnr': 'Cached MNR',
  'cached_gist': 'Cached GIST',
  'cosine': 'Cosine Similarity',
  'cosent': 'CoSENT',
  'angle': 'AnglE',
}

function getLossHint(taskName: string | undefined): string {
  if (!taskName) return ''
  if (['pairs', 'triplets', 'matryoshka'].includes(taskName)) {
    return 'MNR uses in-batch negatives. GIST filters false negatives using the model itself as guide.'
  }
  if (taskName === 'similarity') {
    return 'CoSENT and AnglE often outperform Cosine on STS benchmarks.'
  }
  return ''
}

export function WizardStep2(props: WizardStep2Props) {
  const { task, loss, training, advanced, lora, matryoshka, unsloth, hub } = props

  const showLossField = task && (
    (task.loss_options && task.loss_options.length > 0) || task.name === 'tsdae'
  )

  return (
    <div className="wizard-content">
      {/* Loss Function */}
      {showLossField && (
        <div className="form-section">
          <label className="section-label">Loss Function</label>
          <div className="form-field">
            {task.name === 'tsdae' ? (
              <select disabled>
                <option>Denoising Auto-Encoder Loss</option>
              </select>
            ) : (
              <select
                value={loss.lossVariant}
                onChange={(e) => loss.onLossVariantChange(e.target.value)}
              >
                {task.loss_options.map(l => {
                  const isDefault = l === task.default_loss
                  const label = (LOSS_LABELS[l] || l.toUpperCase()) + (isDefault ? ' (default)' : '')
                  return <option key={l} value={l}>{label}</option>
                })}
              </select>
            )}
            {task.name === 'tsdae' ? (
              <span className="field-hint">TSDAE uses unsupervised denoising to learn embeddings from unlabeled text.</span>
            ) : (
              <span className="field-hint">{getLossHint(task.name)}</span>
            )}
          </div>
        </div>
      )}

      {/* Basic Training Parameters */}
      <div className="form-section">
        <label className="section-label">Training</label>
        <div className="form-row">
          <div className="form-field compact">
            <label>Epochs</label>
            <input type="number" value={training.epochs} onChange={(e) => training.onEpochsChange(e.target.value)} min="1" />
          </div>
          <div className="form-field compact">
            <label>Max Steps</label>
            <input type="number" value={training.maxSteps} onChange={(e) => training.onMaxStepsChange(e.target.value)} placeholder="Optional" min="1" />
          </div>
          <div className="form-field compact">
            <label>Batch Size</label>
            <input type="number" value={training.batchSize} onChange={(e) => training.onBatchSizeChange(e.target.value)} min="1" />
          </div>
          <div className="form-field compact">
            <label>Learning Rate</label>
            <input type="text" value={training.learningRate} onChange={(e) => training.onLearningRateChange(e.target.value)} />
          </div>
        </div>
        <div className="form-row">
          <div className="form-field compact">
            <label>Precision</label>
            <select value={training.precision} onChange={(e) => training.onPrecisionChange(e.target.value)}>
              <option value="fp32">FP32</option>
              <option value="fp16">FP16</option>
              <option value="bf16">BF16</option>
            </select>
          </div>
          <div className="form-field compact">
            <label>Max Seq Length</label>
            <input type="number" value={training.maxSeqLength} onChange={(e) => training.onMaxSeqLengthChange(e.target.value)} placeholder="auto" min="1" />
          </div>
        </div>
      </div>

      {/* Task-specific parameters */}
      {task && task.task_specific_params && Object.keys(task.task_specific_params).length > 0 && (
        <div className="form-section">
          <label className="section-label">Task Settings</label>
          {Object.entries(task.task_specific_params).map(([key, config]) => (
            <div className="form-field" key={key}>
              <label>{config.label}</label>
              <input
                type={config.type}
                value={props.taskSpecificValues[key] || config.default || ''}
                onChange={(e) => props.onTaskSpecificChange(key, e.target.value)}
              />
              {config.description && <span className="field-hint">{config.description}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Advanced Options */}
      <AdvancedOptions>
        <LoraSettings
          enabled={lora.enabled}
          onEnabledChange={lora.onEnabledChange}
          loraR={lora.loraR}
          loraAlpha={lora.loraAlpha}
          loraDropout={lora.loraDropout}
          loraTargetModules={lora.loraTargetModules}
          loraTargetPreset={lora.loraTargetPreset}
          onLoraRChange={lora.onLoraRChange}
          onLoraAlphaChange={lora.onLoraAlphaChange}
          onLoraDropoutChange={lora.onLoraDropoutChange}
          onLoraTargetModulesChange={lora.onLoraTargetModulesChange}
          onLoraTargetPresetChange={lora.onLoraTargetPresetChange}
        />

        <MatryoshkaSettings
          enabled={matryoshka.enabled}
          onEnabledChange={matryoshka.onEnabledChange}
          dims={matryoshka.dims}
          onDimsChange={matryoshka.onDimsChange}
          visible={matryoshka.visible}
        />

        {/* Optimizer & Scheduler */}
        <div className="advanced-group">
          <span className="group-label">Optimizer &amp; Scheduler</span>
          <div className="form-row">
            <div className="form-field compact">
              <label>Optimizer</label>
              <select value={advanced.optimizer} onChange={(e) => advanced.onOptimizerChange(e.target.value)}>
                <option value="adamw_torch">AdamW</option>
                <option value="adamw_torch_fused">AdamW Fused</option>
                <option value="adamw_8bit">AdamW 8-bit</option>
                <option value="adafactor">Adafactor</option>
                <option value="sgd">SGD</option>
              </select>
            </div>
            <div className="form-field compact">
              <label>Scheduler</label>
              <select value={advanced.scheduler} onChange={(e) => advanced.onSchedulerChange(e.target.value)}>
                <option value="linear">Linear</option>
                <option value="cosine">Cosine</option>
                <option value="cosine_with_restarts">Cosine w/ Restarts</option>
                <option value="constant">Constant</option>
                <option value="polynomial">Polynomial</option>
              </select>
            </div>
            <div className="form-field compact">
              <label>Weight Decay</label>
              <input type="text" value={advanced.weightDecay} onChange={(e) => advanced.onWeightDecayChange(e.target.value)} />
            </div>
            <div className="form-field compact">
              <label>Warmup Ratio</label>
              <input type="text" value={advanced.warmupRatio} onChange={(e) => advanced.onWarmupRatioChange(e.target.value)} />
            </div>
          </div>
          <div className="form-row">
            <div className="form-field compact">
              <label>Grad Accum</label>
              <input type="number" value={advanced.gradAccum} onChange={(e) => advanced.onGradAccumChange(e.target.value)} min="1" />
            </div>
            <div className="form-field compact">
              <label>Logging Steps</label>
              <input type="number" value={advanced.loggingSteps} onChange={(e) => advanced.onLoggingStepsChange(e.target.value)} min="0" step="any" />
            </div>
            <div className="form-field compact">
              <label>Eval Steps</label>
              <input type="number" value={advanced.evalSteps} onChange={(e) => advanced.onEvalStepsChange(e.target.value)} min="0" step="any" />
            </div>
            <div className="form-field compact">
              <label>Save Steps</label>
              <input type="number" value={advanced.saveSteps} onChange={(e) => advanced.onSaveStepsChange(e.target.value)} min="0" step="any" />
            </div>
          </div>
          <div className="checkbox-row">
            <input
              type="checkbox"
              id="gradient_checkpointing"
              checked={advanced.gradientCheckpointing}
              onChange={(e) => advanced.onGradientCheckpointingChange(e.target.checked)}
            />
            <label htmlFor="gradient_checkpointing">Gradient Checkpointing (saves VRAM)</label>
          </div>
        </div>

        <UnslothSettings
          enabled={unsloth.enabled}
          onEnabledChange={unsloth.onEnabledChange}
          saveMethod={unsloth.saveMethod}
          onSaveMethodChange={unsloth.onSaveMethodChange}
        />

        <HubSettings
          enabled={hub.enabled}
          onEnabledChange={hub.onEnabledChange}
          username={hub.username}
          onUsernameChange={hub.onUsernameChange}
        />
      </AdvancedOptions>
    </div>
  )
}
