import type {
  Run,
  TrainRequest,
  UploadResponse,
  UpdatesResponse,
  MetricsResponse,
  ArtifactsResponse,
  TaskInfo,
} from './types'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export function fetchRuns(): Promise<Run[]> {
  return request('/runs')
}

export function fetchRun(runId: number): Promise<Run> {
  return request(`/runs/${runId}`)
}

export function deleteRun(runId: number): Promise<{ message: string }> {
  return request(`/runs/${runId}`, { method: 'DELETE' })
}

export function startTraining(config: TrainRequest): Promise<{ message: string; run_id: number; output_dir: string }> {
  return request('/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
}

export function stopTraining(runId: number): Promise<{ message: string }> {
  return request('/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId }),
  })
}

export async function uploadFile(file: File, fileType: 'train' | 'eval'): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('file_type', fileType)
  return request('/upload', { method: 'POST', body: formData })
}

export function fetchUpdates(runId: number, sinceLine: number): Promise<UpdatesResponse> {
  return request(`/runs/${runId}/updates?since_line=${sinceLine}`)
}

export function fetchMetrics(runId: number): Promise<MetricsResponse> {
  return request(`/runs/${runId}/metrics`)
}

export function fetchActiveRunId(): Promise<{ run_id: number | null }> {
  return request('/active_run_id')
}

export function fetchTasks(): Promise<TaskInfo[]> {
  return request('/api/tasks')
}

export function fetchTask(taskName: string): Promise<TaskInfo> {
  return request(`/api/tasks/${taskName}`)
}

export function fetchArtifacts(runId: number): Promise<ArtifactsResponse> {
  return request(`/runs/${runId}/artifacts`)
}

export function fetchVersion(): Promise<{ version: string }> {
  return request('/api/version')
}

export function getArtifactDownloadUrl(runId: number, artifactName: string): string {
  return `/runs/${runId}/artifacts/${encodeURIComponent(artifactName)}/download`
}
