import { useState, useCallback, useEffect, useMemo } from 'react'
import { useAppState, useAppDispatch } from '../../context/AppContext'
import { useRuns } from '../../hooks/useRuns'
import { Modal, ModalHeader } from '../ui/Modal'
import { WizardStep1 } from './WizardStep1'
import { WizardStep2 } from './WizardStep2'
import * as api from '../../api/client'
import type { TrainRequest, TaskInfo } from '../../api/types'

function generateProjectName(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
  let name = ''
  for (let i = 0; i < 8; i++) {
    name += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return name
}

function getDefaults(task?: TaskInfo) {
  const h = task?.hyperparameters as Record<string, unknown> | undefined
  return {
    epochs: String(h?.epochs ?? 3),
    batchSize: String(h?.batch_size ?? 32),
    learningRate: String(h?.learning_rate ?? 2e-5),
    warmupRatio: String(h?.warmup_ratio ?? 0.1),
    weightDecay: String(h?.weight_decay ?? 0.01),
    evalSteps: String(h?.eval_steps ?? 0.25),
    saveSteps: String(h?.save_steps ?? 0.5),
    loggingSteps: String(h?.logging_steps ?? 0.02),
    gradAccum: String(h?.gradient_accumulation_steps ?? 1),
    optimizer: String(h?.optimizer ?? 'adamw_torch'),
    scheduler: String(h?.scheduler ?? 'linear'),
    precision: h?.bf16 ? 'bf16' : h?.fp16 ? 'fp16' : 'fp32',
    lossVariant: task?.default_loss || '',
  }
}

export function NewProjectModal() {
  const { showNewProjectModal, tasksData, copyFromConfig } = useAppState()
  const dispatch = useAppDispatch()
  const { loadRuns } = useRuns()

  // Wizard step
  const [step, setStep] = useState(1)

  // Step 1 state
  const [projectName, setProjectName] = useState('')
  const [selectedTask, setSelectedTask] = useState('')
  const [baseModel, setBaseModel] = useState('sentence-transformers/all-MiniLM-L6-v2')
  const [dataSource, setDataSource] = useState<'file' | 'huggingface'>('file')
  const [trainFilepath, setTrainFilepath] = useState('')
  const [trainFileInfo, setTrainFileInfo] = useState('CSV or JSONL')
  const [evalFilepath, setEvalFilepath] = useState('')
  const [evalFileInfo, setEvalFileInfo] = useState('CSV or JSONL')
  const [autoSplitFile, setAutoSplitFile] = useState(false)
  const [evalSplitPct, setEvalSplitPct] = useState('10')
  const [hfDataset, setHfDataset] = useState('')
  const [hfSubset, setHfSubset] = useState('')
  const [hfTrainSplit, setHfTrainSplit] = useState('train')
  const [hfEvalSplit, setHfEvalSplit] = useState('')
  const [hfAutoSplit, setHfAutoSplit] = useState(false)
  const [hfEvalSplitPct, setHfEvalSplitPct] = useState('10')

  // Step 2 state
  const [epochs, setEpochs] = useState('3')
  const [maxSteps, setMaxSteps] = useState('')
  const [batchSize, setBatchSize] = useState('32')
  const [learningRate, setLearningRate] = useState('2e-5')
  const [precision, setPrecision] = useState('fp32')
  const [maxSeqLength, setMaxSeqLength] = useState('')
  const [lossVariant, setLossVariant] = useState('')

  // Advanced
  const [optimizer, setOptimizer] = useState('adamw_torch')
  const [scheduler, setScheduler] = useState('linear')
  const [weightDecay, setWeightDecay] = useState('0.01')
  const [warmupRatio, setWarmupRatio] = useState('0.1')
  const [gradAccum, setGradAccum] = useState('1')
  const [loggingSteps, setLoggingSteps] = useState('0.02')
  const [evalSteps, setEvalSteps] = useState('0.25')
  const [saveSteps, setSaveSteps] = useState('0.5')
  const [gradientCheckpointing, setGradientCheckpointing] = useState(false)

  // LoRA
  const [loraEnabled, setLoraEnabled] = useState(false)
  const [loraR, setLoraR] = useState('64')
  const [loraAlpha, setLoraAlpha] = useState('128')
  const [loraDropout, setLoraDropout] = useState('0.1')
  const [loraTargetModules, setLoraTargetModules] = useState('query, key, value, dense')
  const [loraTargetPreset, setLoraTargetPreset] = useState('query, key, value, dense')

  // Matryoshka
  const [matryoshkaEnabled, setMatryoshkaEnabled] = useState(false)
  const [matryoshkaDims, setMatryoshkaDims] = useState('768,512,256,128,64')

  // Unsloth
  const [unslothEnabled, setUnslothEnabled] = useState(false)
  const [unslothSaveMethod, setUnslothSaveMethod] = useState('merged_16bit')

  // Hub
  const [hubEnabled, setHubEnabled] = useState(false)
  const [hfUsername, setHfUsername] = useState('')

  // Task-specific
  const [taskSpecificValues, setTaskSpecificValues] = useState<Record<string, string>>({})

  // Submitting
  const [submitting, setSubmitting] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)

  const currentTask = tasksData.find(t => t.name === selectedTask)

  // Full form reset when modal opens
  useEffect(() => {
    if (showNewProjectModal) {
      const c = copyFromConfig as Record<string, unknown> | null
      setStep(1)
      setSubmitting(false)
      setFormError(null)

      // Always generate a new project name (with "copy" suffix if copying)
      setProjectName(c?.project_name
        ? `${String(c.project_name)}-copy`
        : generateProjectName()
      )

      // Data source — when copying, file paths won't exist on disk, so reset file uploads
      // but preserve HuggingFace dataset info
      setDataSource(c?.hf_dataset ? 'huggingface' : 'file')
      setTrainFilepath('')
      setTrainFileInfo('CSV or JSONL')
      setEvalFilepath('')
      setEvalFileInfo('CSV or JSONL')
      setAutoSplitFile(false)
      setEvalSplitPct(c?.eval_split_pct ? String(c.eval_split_pct) : '10')
      setHfDataset(c?.hf_dataset ? String(c.hf_dataset) : '')
      setHfSubset(c?.hf_subset ? String(c.hf_subset) : '')
      setHfTrainSplit(c?.hf_train_split ? String(c.hf_train_split) : 'train')
      setHfEvalSplit(c?.hf_eval_split ? String(c.hf_eval_split) : '')
      setHfAutoSplit(false)
      setHfEvalSplitPct(c?.eval_split_pct ? String(c.eval_split_pct) : '10')

      // Model & task
      setBaseModel(c?.base_model ? String(c.base_model) : 'sentence-transformers/all-MiniLM-L6-v2')

      // Training params — use copied config or task defaults
      if (c) {
        setSelectedTask(String(c.task || ''))
        setEpochs(String(c.epochs ?? 3))
        setMaxSteps(c.max_steps ? String(c.max_steps) : '')
        setBatchSize(String(c.batch_size ?? 32))
        setLearningRate(String(c.learning_rate ?? 2e-5))
        setPrecision(c.bf16 ? 'bf16' : c.fp16 ? 'fp16' : 'fp32')
        setMaxSeqLength(c.max_seq_length ? String(c.max_seq_length) : '')
        setLossVariant(c.loss_variant ? String(c.loss_variant) : '')
        setOptimizer(String(c.optimizer ?? 'adamw_torch'))
        setScheduler(String(c.scheduler ?? 'linear'))
        setWeightDecay(String(c.weight_decay ?? 0.01))
        setWarmupRatio(String(c.warmup_ratio ?? 0.1))
        setGradAccum(String(c.gradient_accumulation_steps ?? 1))
        setLoggingSteps(String(c.logging_steps ?? 0.02))
        setEvalSteps(String(c.eval_steps ?? 0.25))
        setSaveSteps(String(c.save_steps ?? 0.5))
        setGradientCheckpointing(!!c.gradient_checkpointing)
        setLoraEnabled(!!c.lora_enabled)
        setLoraR(String(c.lora_r ?? 64))
        setLoraAlpha(String(c.lora_alpha ?? 128))
        setLoraDropout(String(c.lora_dropout ?? 0.1))
        setLoraTargetModules(c.lora_target_modules ? String(c.lora_target_modules) : 'query, key, value, dense')
        setLoraTargetPreset(c.lora_target_modules ? String(c.lora_target_modules) : 'query, key, value, dense')
        setMatryoshkaEnabled(!!c.matryoshka_dims)
        setMatryoshkaDims(c.matryoshka_dims ? String(c.matryoshka_dims) : '768,512,256,128,64')
        setUnslothEnabled(!!c.unsloth_enabled)
        setUnslothSaveMethod(c.unsloth_save_method ? String(c.unsloth_save_method) : 'merged_16bit')
        setHubEnabled(!!c.push_to_hub)
        setHfUsername(c.hf_username ? String(c.hf_username) : '')
      } else {
        setMaxSteps('')
        setMaxSeqLength('')
        setGradientCheckpointing(false)
        setLoraEnabled(false)
        setLoraR('64')
        setLoraAlpha('128')
        setLoraDropout('0.1')
        setLoraTargetModules('query, key, value, dense')
        setLoraTargetPreset('query, key, value, dense')
        setMatryoshkaEnabled(false)
        setMatryoshkaDims('768,512,256,128,64')
        setUnslothEnabled(false)
        setUnslothSaveMethod('merged_16bit')
        setHubEnabled(false)
        setHfUsername('')

        if (tasksData.length > 0) {
          const taskName = tasksData[0].name
          setSelectedTask(taskName)
          const d = getDefaults(tasksData[0])
          setEpochs(d.epochs)
          setBatchSize(d.batchSize)
          setLearningRate(d.learningRate)
          setWarmupRatio(d.warmupRatio)
          setWeightDecay(d.weightDecay)
          setEvalSteps(d.evalSteps)
          setSaveSteps(d.saveSteps)
          setLoggingSteps(d.loggingSteps)
          setGradAccum(d.gradAccum)
          setOptimizer(d.optimizer)
          setScheduler(d.scheduler)
          setPrecision(d.precision)
          setLossVariant(d.lossVariant)
        }
      }
      setTaskSpecificValues({})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [showNewProjectModal])

  // Update defaults when task changes
  const handleTaskChange = useCallback((taskName: string) => {
    setSelectedTask(taskName)
    const task = tasksData.find(t => t.name === taskName)
    if (!task) return
    const d = getDefaults(task)
    setEpochs(d.epochs)
    setBatchSize(d.batchSize)
    setLearningRate(d.learningRate)
    setWarmupRatio(d.warmupRatio)
    setWeightDecay(d.weightDecay)
    setEvalSteps(d.evalSteps)
    setSaveSteps(d.saveSteps)
    setLoggingSteps(d.loggingSteps)
    setGradAccum(d.gradAccum)
    setOptimizer(d.optimizer)
    setScheduler(d.scheduler)
    setPrecision(d.precision)
    setLossVariant(d.lossVariant)
    if (task.name === 'tsdae') {
      setMatryoshkaEnabled(false)
    }
    setTaskSpecificValues({})
  }, [tasksData])

  const close = useCallback(() => {
    dispatch({ type: 'SET_SHOW_NEW_PROJECT_MODAL', show: false })
  }, [dispatch])

  const validateStep1 = useCallback((): boolean => {
    if (!projectName.trim()) {
      setFormError('Please enter a project name')
      return false
    }
    if (!/^[a-zA-Z0-9][a-zA-Z0-9-]*$/.test(projectName.trim())) {
      setFormError('Project name must start with alphanumeric and contain only alphanumeric characters and hyphens')
      return false
    }
    if (dataSource === 'file' && !trainFilepath) {
      setFormError('Please upload training data')
      return false
    }
    if (dataSource === 'huggingface' && !hfDataset.trim()) {
      setFormError('Please enter a HuggingFace dataset name')
      return false
    }
    setFormError(null)
    return true
  }, [projectName, dataSource, trainFilepath, hfDataset])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setFormError(null)

    // Validate steps/ratio fields
    const stepsFields = [
      { value: loggingSteps, name: 'Logging' },
      { value: evalSteps, name: 'Eval' },
      { value: saveSteps, name: 'Save' },
    ]
    for (const field of stepsFields) {
      const v = parseFloat(field.value)
      if (isNaN(v) || v <= 0) {
        setFormError(`${field.name} steps/ratio must be a positive number`)
        return
      }
      if (v >= 1 && v !== Math.floor(v)) {
        setFormError(`${field.name}: Values >= 1 must be whole numbers. Use decimals < 1 for ratios.`)
        return
      }
    }

    if (hubEnabled && !hfUsername.trim()) {
      setFormError('Please enter your HuggingFace username')
      return
    }

    const formData: TrainRequest = {
      project_name: projectName.trim(),
      task: selectedTask,
      base_model: baseModel,
      loss_variant: lossVariant || null,
      epochs: parseInt(epochs),
      max_steps: maxSteps ? parseInt(maxSteps) : null,
      batch_size: parseInt(batchSize),
      learning_rate: parseFloat(learningRate),
      warmup_ratio: parseFloat(warmupRatio),
      weight_decay: parseFloat(weightDecay),
      gradient_accumulation_steps: parseInt(gradAccum),
      logging_steps: parseFloat(loggingSteps),
      eval_steps: parseFloat(evalSteps),
      save_steps: parseFloat(saveSteps),
      fp16: precision === 'fp16',
      bf16: precision === 'bf16',
      optimizer,
      scheduler,
      lora_enabled: loraEnabled,
      lora_r: parseInt(loraR),
      lora_alpha: parseInt(loraAlpha),
      lora_dropout: parseFloat(loraDropout),
      lora_target_modules: loraTargetModules.trim(),
      max_seq_length: maxSeqLength ? parseInt(maxSeqLength) : null,
      gradient_checkpointing: gradientCheckpointing,
      unsloth_enabled: unslothEnabled,
      unsloth_save_method: unslothSaveMethod,
      matryoshka_dims: matryoshkaEnabled ? matryoshkaDims.trim() : null,
      push_to_hub: hubEnabled,
      hf_username: hfUsername.trim() || null,
    }

    // Add data source
    if (dataSource === 'file') {
      formData.train_filename = trainFilepath
      formData.eval_filename = evalFilepath || null
      if (autoSplitFile && !evalFilepath) {
        formData.eval_split_pct = parseFloat(evalSplitPct)
      }
    } else {
      formData.hf_dataset = hfDataset.trim()
      formData.hf_subset = hfSubset.trim() || null
      formData.hf_train_split = hfTrainSplit.trim() || 'train'
      formData.hf_eval_split = hfEvalSplit.trim() || null
      if (hfAutoSplit && !hfEvalSplit.trim()) {
        formData.eval_split_pct = parseFloat(hfEvalSplitPct)
      }
    }

    // Add task-specific params
    for (const [key, value] of Object.entries(taskSpecificValues)) {
      if (value) formData[key] = value
    }

    setSubmitting(true)
    try {
      await api.startTraining(formData)
      close()
      dispatch({ type: 'RESET_MONITOR' })
      dispatch({ type: 'SET_STATUS_MESSAGE', message: 'Initializing training...' })
      await loadRuns()
    } catch (error) {
      setFormError(`Training failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setSubmitting(false)
    }
  }

  const onTrainUploaded = useCallback((fp: string, info: string) => {
    setTrainFilepath(fp); setTrainFileInfo(info)
  }, [])
  const onEvalUploaded = useCallback((fp: string, info: string) => {
    setEvalFilepath(fp); setEvalFileInfo(info)
  }, [])

  const lossProps = useMemo(() => ({
    lossVariant, onLossVariantChange: setLossVariant,
  }), [lossVariant])

  const trainingProps = useMemo(() => ({
    epochs, maxSteps, batchSize, learningRate, precision, maxSeqLength,
    onEpochsChange: setEpochs, onMaxStepsChange: setMaxSteps,
    onBatchSizeChange: setBatchSize, onLearningRateChange: setLearningRate,
    onPrecisionChange: setPrecision, onMaxSeqLengthChange: setMaxSeqLength,
  }), [epochs, maxSteps, batchSize, learningRate, precision, maxSeqLength])

  const advancedProps = useMemo(() => ({
    optimizer, scheduler, weightDecay, warmupRatio, gradAccum,
    loggingSteps, evalSteps, saveSteps, gradientCheckpointing,
    onOptimizerChange: setOptimizer, onSchedulerChange: setScheduler,
    onWeightDecayChange: setWeightDecay, onWarmupRatioChange: setWarmupRatio,
    onGradAccumChange: setGradAccum, onLoggingStepsChange: setLoggingSteps,
    onEvalStepsChange: setEvalSteps, onSaveStepsChange: setSaveSteps,
    onGradientCheckpointingChange: setGradientCheckpointing,
  }), [optimizer, scheduler, weightDecay, warmupRatio, gradAccum, loggingSteps, evalSteps, saveSteps, gradientCheckpointing])

  const loraProps = useMemo(() => ({
    enabled: loraEnabled, loraR, loraAlpha, loraDropout,
    loraTargetModules, loraTargetPreset,
    onEnabledChange: setLoraEnabled, onLoraRChange: setLoraR,
    onLoraAlphaChange: setLoraAlpha, onLoraDropoutChange: setLoraDropout,
    onLoraTargetModulesChange: setLoraTargetModules,
    onLoraTargetPresetChange: setLoraTargetPreset,
  }), [loraEnabled, loraR, loraAlpha, loraDropout, loraTargetModules, loraTargetPreset])

  const matryoshkaProps = useMemo(() => ({
    enabled: matryoshkaEnabled, dims: matryoshkaDims,
    visible: currentTask?.name !== 'tsdae',
    onEnabledChange: setMatryoshkaEnabled, onDimsChange: setMatryoshkaDims,
  }), [matryoshkaEnabled, matryoshkaDims, currentTask?.name])

  const unslothProps = useMemo(() => ({
    enabled: unslothEnabled, saveMethod: unslothSaveMethod,
    onEnabledChange: setUnslothEnabled, onSaveMethodChange: setUnslothSaveMethod,
  }), [unslothEnabled, unslothSaveMethod])

  const hubProps = useMemo(() => ({
    enabled: hubEnabled, username: hfUsername,
    onEnabledChange: setHubEnabled, onUsernameChange: setHfUsername,
  }), [hubEnabled, hfUsername])

  const onTaskSpecificChange = useCallback((key: string, value: string) => {
    setTaskSpecificValues(prev => ({ ...prev, [key]: value }))
  }, [])

  return (
    <Modal show={showNewProjectModal} onClose={close} className="modal-wizard" closeOnBackdrop={false}>
      <ModalHeader title="New Project" onClose={close} />

      {/* Step Indicators */}
      <div className="wizard-steps">
        <div className={`wizard-step ${step === 1 ? 'active' : step > 1 ? 'completed' : ''}`}>
          <span className="step-number">1</span>
          <span className="step-label">Setup</span>
        </div>
        <div className={`wizard-step ${step === 2 ? 'active' : ''}`}>
          <span className="step-number">2</span>
          <span className="step-label">Training</span>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        {step === 1 && (
          <WizardStep1
            projectName={projectName}
            onProjectNameChange={setProjectName}
            tasksData={tasksData}
            selectedTask={selectedTask}
            onTaskChange={handleTaskChange}
            baseModel={baseModel}
            onBaseModelChange={setBaseModel}
            dataSource={dataSource}
            onDataSourceChange={setDataSource}
            trainFilepath={trainFilepath}
            trainFileInfo={trainFileInfo}
            onTrainUploaded={onTrainUploaded}
            evalFilepath={evalFilepath}
            evalFileInfo={evalFileInfo}
            onEvalUploaded={onEvalUploaded}
            autoSplitFile={autoSplitFile}
            onAutoSplitFileChange={setAutoSplitFile}
            evalSplitPct={evalSplitPct}
            onEvalSplitPctChange={setEvalSplitPct}
            hfDataset={hfDataset}
            onHfDatasetChange={setHfDataset}
            hfSubset={hfSubset}
            onHfSubsetChange={setHfSubset}
            hfTrainSplit={hfTrainSplit}
            onHfTrainSplitChange={setHfTrainSplit}
            hfEvalSplit={hfEvalSplit}
            onHfEvalSplitChange={setHfEvalSplit}
            hfAutoSplit={hfAutoSplit}
            onHfAutoSplitChange={setHfAutoSplit}
            hfEvalSplitPct={hfEvalSplitPct}
            onHfEvalSplitPctChange={setHfEvalSplitPct}
          />
        )}
        {step === 2 && (
          <WizardStep2
            task={currentTask}
            loss={lossProps}
            training={trainingProps}
            advanced={advancedProps}
            lora={loraProps}
            matryoshka={matryoshkaProps}
            unsloth={unslothProps}
            hub={hubProps}
            taskSpecificValues={taskSpecificValues}
            onTaskSpecificChange={onTaskSpecificChange}
          />
        )}

        {formError && (
          <div className="form-error-banner">{formError}</div>
        )}
        <div className="modal-footer wizard-footer">
          {step > 1 && (
            <button type="button" className="btn-secondary" onClick={() => setStep(step - 1)}>
              Back
            </button>
          )}
          <div className="footer-spacer" />
          {step < 2 && (
            <button
              type="button"
              className="btn-primary"
              onClick={() => { if (validateStep1()) setStep(2) }}
            >
              Next
            </button>
          )}
          {step === 2 && (
            <button type="submit" className="btn-primary" disabled={submitting}>
              {submitting ? 'Starting...' : 'Start Training'}
            </button>
          )}
        </div>
      </form>
    </Modal>
  )
}
