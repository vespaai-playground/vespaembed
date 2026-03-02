import { TaskSelector } from '../form/TaskSelector'
import { DataSourceTabs } from '../form/DataSourceTabs'
import { FileUpload } from '../form/FileUpload'
import type { TaskInfo } from '../../api/types'

interface WizardStep1Props {
  projectName: string
  onProjectNameChange: (name: string) => void
  tasksData: TaskInfo[]
  selectedTask: string
  onTaskChange: (taskName: string) => void
  baseModel: string
  onBaseModelChange: (model: string) => void
  dataSource: 'file' | 'huggingface'
  onDataSourceChange: (ds: 'file' | 'huggingface') => void
  // File upload
  trainFilepath: string
  trainFileInfo: string
  onTrainUploaded: (filepath: string, info: string) => void
  evalFilepath: string
  evalFileInfo: string
  onEvalUploaded: (filepath: string, info: string) => void
  autoSplitFile: boolean
  onAutoSplitFileChange: (v: boolean) => void
  evalSplitPct: string
  onEvalSplitPctChange: (v: string) => void
  // HuggingFace
  hfDataset: string
  onHfDatasetChange: (v: string) => void
  hfSubset: string
  onHfSubsetChange: (v: string) => void
  hfTrainSplit: string
  onHfTrainSplitChange: (v: string) => void
  hfEvalSplit: string
  onHfEvalSplitChange: (v: string) => void
  hfAutoSplit: boolean
  onHfAutoSplitChange: (v: boolean) => void
  hfEvalSplitPct: string
  onHfEvalSplitPctChange: (v: string) => void
}

export function WizardStep1(props: WizardStep1Props) {
  return (
    <div className="wizard-content">
      {/* Project Name */}
      <div className="form-section">
        <label className="section-label">Project Name</label>
        <input
          type="text"
          value={props.projectName}
          onChange={(e) => props.onProjectNameChange(e.target.value)}
          required
          pattern="[a-zA-Z0-9][a-zA-Z0-9-]*"
        />
        <span className="field-hint">Saved to ~/.vespaembed/projects/</span>
      </div>

      {/* Task & Model */}
      <TaskSelector
        tasksData={props.tasksData}
        selectedTask={props.selectedTask}
        onTaskChange={props.onTaskChange}
      />
      <div className="form-section" style={{ marginTop: 'var(--space-sm)' }}>
        <div className="form-field">
          <label>Base Model</label>
          <input
            type="text"
            value={props.baseModel}
            onChange={(e) => props.onBaseModelChange(e.target.value)}
            required
          />
        </div>
      </div>

      {/* Data Source */}
      <DataSourceTabs
        currentTab={props.dataSource}
        onTabChange={props.onDataSourceChange}
        fileContent={
          <>
            <div className="upload-grid">
              <FileUpload
                label="Training Data"
                required
                fileType="train"
                filepath={props.trainFilepath}
                fileInfo={props.trainFileInfo}
                onUploaded={props.onTrainUploaded}
              />
              <FileUpload
                label="Eval Data"
                fileType="eval"
                filepath={props.evalFilepath}
                fileInfo={props.evalFileInfo}
                onUploaded={props.onEvalUploaded}
                disabled={props.autoSplitFile}
              />
            </div>
            <div className="autosplit-option">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={props.autoSplitFile}
                  onChange={(e) => props.onAutoSplitFileChange(e.target.checked)}
                  disabled={!!props.evalFilepath}
                />
                <span className="toggle-slider" />
              </label>
              <label className="toggle-label">
                Auto-split for evaluation
                <span className="hint">Split a portion of training data for validation</span>
              </label>
              <div className="autosplit-pct-input">
                <input
                  type="number"
                  value={props.evalSplitPct}
                  onChange={(e) => props.onEvalSplitPctChange(e.target.value)}
                  min="0.1"
                  max="50"
                  step="0.1"
                  disabled={!props.autoSplitFile}
                />
                <span>%</span>
              </div>
            </div>
          </>
        }
        hfContent={
          <>
            <div className="form-row">
              <div className="form-field">
                <label>Dataset <span className="required">*</span></label>
                <input
                  type="text"
                  value={props.hfDataset}
                  onChange={(e) => props.onHfDatasetChange(e.target.value)}
                  placeholder="sentence-transformers/all-nli"
                />
              </div>
              <div className="form-field">
                <label>Subset</label>
                <input
                  type="text"
                  value={props.hfSubset}
                  onChange={(e) => props.onHfSubsetChange(e.target.value)}
                  placeholder="triplet"
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-field">
                <label>Train Split</label>
                <input
                  type="text"
                  value={props.hfTrainSplit}
                  onChange={(e) => props.onHfTrainSplitChange(e.target.value)}
                />
              </div>
              <div className="form-field">
                <label>Eval Split</label>
                <input
                  type="text"
                  value={props.hfEvalSplit}
                  onChange={(e) => props.onHfEvalSplitChange(e.target.value)}
                  placeholder="validation"
                  disabled={props.hfAutoSplit}
                />
              </div>
            </div>
            <div className="autosplit-option">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={props.hfAutoSplit}
                  onChange={(e) => props.onHfAutoSplitChange(e.target.checked)}
                  disabled={!!props.hfEvalSplit}
                />
                <span className="toggle-slider" />
              </label>
              <label className="toggle-label">
                Auto-split for evaluation
                <span className="hint">Use when dataset has no eval/validation split</span>
              </label>
              <div className="autosplit-pct-input">
                <input
                  type="number"
                  value={props.hfEvalSplitPct}
                  onChange={(e) => props.onHfEvalSplitPctChange(e.target.value)}
                  min="0.1"
                  max="50"
                  step="0.1"
                  disabled={!props.hfAutoSplit}
                />
                <span>%</span>
              </div>
            </div>
          </>
        }
      />
    </div>
  )
}
