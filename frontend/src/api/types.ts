export interface Run {
  id: number
  project_name: string
  status: string
  config: string // JSON string
  output_dir: string | null
  pid: number | null
  error_message: string | null
  created_at: string
  updated_at: string
}

export interface RunConfig {
  project_name: string
  task: string
  base_model: string
  loss_variant: string | null
  epochs: number
  max_steps: number | null
  batch_size: number
  learning_rate: number
  warmup_ratio: number
  weight_decay: number
  gradient_accumulation_steps: number
  logging_steps: number
  eval_steps: number
  save_steps: number
  fp16: boolean
  bf16: boolean
  optimizer: string
  scheduler: string
  lora_enabled: boolean
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string
  max_seq_length: number | null
  gradient_checkpointing: boolean
  unsloth_enabled: boolean
  unsloth_save_method: string
  matryoshka_dims: string | null
  push_to_hub: boolean
  hf_username: string | null
  train_filename: string | null
  eval_filename: string | null
  eval_split_pct: number | null
  hf_dataset: string | null
  hf_subset: string | null
  hf_train_split: string
  hf_eval_split: string | null
  output_dir: string
}

export interface TrainRequest {
  project_name: string
  task: string
  base_model: string
  loss_variant?: string | null
  epochs: number
  max_steps?: number | null
  batch_size: number
  learning_rate: number
  warmup_ratio: number
  weight_decay: number
  gradient_accumulation_steps: number
  logging_steps: number
  eval_steps: number
  save_steps: number
  fp16: boolean
  bf16: boolean
  optimizer: string
  scheduler: string
  lora_enabled: boolean
  lora_r: number
  lora_alpha: number
  lora_dropout: number
  lora_target_modules: string
  max_seq_length?: number | null
  gradient_checkpointing: boolean
  unsloth_enabled: boolean
  unsloth_save_method: string
  matryoshka_dims?: string | null
  push_to_hub: boolean
  hf_username?: string | null
  train_filename?: string | null
  eval_filename?: string | null
  eval_split_pct?: number | null
  hf_dataset?: string | null
  hf_subset?: string | null
  hf_train_split?: string
  hf_eval_split?: string | null
  [key: string]: unknown // task-specific params
}

export interface UploadResponse {
  filename: string
  filepath: string
  columns: string[]
  preview: Record<string, unknown>[]
  row_count: number
}

export interface UpdateEntry {
  type: 'log' | 'progress' | 'status' | 'complete' | 'error' | 'train_end'
  message?: string
  step?: number
  total_steps?: number
  loss?: number | null
  eta_seconds?: number
  run_id?: number
  status?: string
  output_dir?: string
  epoch?: number
  total_epochs?: number
  elapsed_seconds?: number
}

export interface UpdatesResponse {
  updates: UpdateEntry[]
  next_line: number
  has_more: boolean
  run_status: string
}

export interface MetricsResponse {
  metrics: Record<string, { step: number; value: number | null }[]>
  error?: string
}

export interface Artifact {
  name: string
  label: string
  category: string
  path: string
  size: number
  is_directory: boolean
}

export interface ArtifactsResponse {
  artifacts: Artifact[]
  output_dir?: string
}

export interface TaskParam {
  label: string
  type: string
  default?: string
  description?: string
}

export interface TaskInfo {
  name: string
  description: string
  expected_columns: string[]
  sample_data: Record<string, unknown>[]
  hyperparameters: Record<string, unknown>
  task_specific_params: Record<string, TaskParam>
  loss_options: string[]
  default_loss: string
}
