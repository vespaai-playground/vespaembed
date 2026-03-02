import { createContext, useContext, useReducer, type ReactNode, type Dispatch } from 'react'
import type { Run, TaskInfo } from '../api/types'

export interface MetricPoint {
  step: number
  value: number | null
}

export interface AppState {
  runs: Run[]
  activeRunId: number | null
  selectedRunId: number | null
  metricsData: Record<string, MetricPoint[]>
  currentMetric: string
  logs: string[]
  pollLine: number
  currentStep: string
  currentLoss: string
  currentEta: string
  tasksData: TaskInfo[]
  statusMessage: string | null
  showNewProjectModal: boolean
  showArtifactsModal: boolean
  copyFromConfig: Record<string, unknown> | null
  version: string
}

export type AppAction =
  | { type: 'SET_RUNS'; runs: Run[] }
  | { type: 'SET_ACTIVE_RUN_ID'; id: number | null }
  | { type: 'SET_SELECTED_RUN_ID'; id: number | null }
  | { type: 'SET_METRICS_DATA'; data: Record<string, MetricPoint[]> }
  | { type: 'SET_CURRENT_METRIC'; metric: string }
  | { type: 'APPEND_LOG'; message: string }
  | { type: 'CLEAR_LOGS' }
  | { type: 'SET_POLL_LINE'; line: number }
  | { type: 'SET_CURRENT_STEP'; step: string }
  | { type: 'SET_CURRENT_LOSS'; loss: string }
  | { type: 'SET_CURRENT_ETA'; eta: string }
  | { type: 'SET_TASKS_DATA'; tasks: TaskInfo[] }
  | { type: 'SET_STATUS_MESSAGE'; message: string | null }
  | { type: 'SET_SHOW_NEW_PROJECT_MODAL'; show: boolean }
  | { type: 'SET_SHOW_ARTIFACTS_MODAL'; show: boolean }
  | { type: 'COPY_PROJECT'; config: Record<string, unknown> }
  | { type: 'SET_VERSION'; version: string }
  | { type: 'RESET_MONITOR' }
  | { type: 'UPDATE_RUN_STATUS'; runId: number; status: string }

const initialState: AppState = {
  runs: [],
  activeRunId: null,
  selectedRunId: null,
  metricsData: {},
  currentMetric: 'loss',
  logs: [],
  pollLine: 0,
  currentStep: '0',
  currentLoss: '--',
  currentEta: '--',
  tasksData: [],
  statusMessage: null,
  showNewProjectModal: false,
  showArtifactsModal: false,
  copyFromConfig: null,
  version: 'dev',
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_RUNS':
      return { ...state, runs: action.runs }
    case 'SET_ACTIVE_RUN_ID':
      return { ...state, activeRunId: action.id }
    case 'SET_SELECTED_RUN_ID':
      return { ...state, selectedRunId: action.id }
    case 'SET_METRICS_DATA':
      return { ...state, metricsData: action.data }
    case 'SET_CURRENT_METRIC':
      return { ...state, currentMetric: action.metric }
    case 'APPEND_LOG': {
      const MAX_LOGS = 5000
      const newLogs = [...state.logs, action.message]
      return { ...state, logs: newLogs.length > MAX_LOGS ? newLogs.slice(-MAX_LOGS) : newLogs }
    }
    case 'CLEAR_LOGS':
      return { ...state, logs: [] }
    case 'SET_POLL_LINE':
      return { ...state, pollLine: action.line }
    case 'SET_CURRENT_STEP':
      return { ...state, currentStep: action.step }
    case 'SET_CURRENT_LOSS':
      return { ...state, currentLoss: action.loss }
    case 'SET_CURRENT_ETA':
      return { ...state, currentEta: action.eta }
    case 'SET_TASKS_DATA':
      return { ...state, tasksData: action.tasks }
    case 'SET_STATUS_MESSAGE':
      return { ...state, statusMessage: action.message }
    case 'SET_SHOW_NEW_PROJECT_MODAL':
      return { ...state, showNewProjectModal: action.show, copyFromConfig: action.show ? state.copyFromConfig : null }
    case 'SET_SHOW_ARTIFACTS_MODAL':
      return { ...state, showArtifactsModal: action.show }
    case 'COPY_PROJECT':
      return { ...state, copyFromConfig: action.config, showNewProjectModal: true }
    case 'SET_VERSION':
      return { ...state, version: action.version }
    case 'RESET_MONITOR':
      return {
        ...state,
        metricsData: {},
        logs: [],
        currentStep: '0',
        currentLoss: '--',
        currentEta: '--',
        statusMessage: null,
      }
    case 'UPDATE_RUN_STATUS': {
      const updatedRuns = state.runs.map(r =>
        r.id === action.runId ? { ...r, status: action.status } : r
      )
      return { ...state, runs: updatedRuns }
    }
    default:
      return state
  }
}

interface AppContextValue {
  state: AppState
  dispatch: Dispatch<AppAction>
}

const AppContext = createContext<AppContextValue | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState)

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  )
}

export function useAppState(): AppState {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useAppState must be used within AppProvider')
  return ctx.state
}

export function useAppDispatch(): Dispatch<AppAction> {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useAppDispatch must be used within AppProvider')
  return ctx.dispatch
}
