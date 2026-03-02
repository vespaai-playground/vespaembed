import { useCallback } from 'react'
import { useAppState, useAppDispatch } from '../context/AppContext'
import * as api from '../api/client'

export function useMetrics() {
  const state = useAppState()
  const dispatch = useAppDispatch()

  const loadMetrics = useCallback(async (runId: number) => {
    try {
      const data = await api.fetchMetrics(runId)
      dispatch({ type: 'SET_METRICS_DATA', data: data.metrics || {} })
    } catch (error) {
      console.error('Failed to load metrics:', error)
    }
  }, [dispatch])

  const setCurrentMetric = useCallback((metric: string) => {
    dispatch({ type: 'SET_CURRENT_METRIC', metric })
  }, [dispatch])

  const getAvailableMetrics = useCallback((): string[] => {
    const available = Object.keys(state.metricsData).filter(key => {
      const keyLower = key.toLowerCase()
      if (keyLower === 'epoch' || keyLower.includes('epoch')) return false
      const data = state.metricsData[key]
      if (!data || data.length < 2) return false
      return data.filter(d => d.value !== null).length >= 2
    })

    available.sort((a, b) => {
      const aLower = a.toLowerCase()
      const bLower = b.toLowerCase()
      if (aLower === 'loss') return -1
      if (bLower === 'loss') return 1
      if (aLower === 'eval_loss') return -1
      if (bLower === 'eval_loss') return 1
      const aIsUtility = aLower.includes('flos') || aLower.includes('runtime') || aLower.includes('per_second')
      const bIsUtility = bLower.includes('flos') || bLower.includes('runtime') || bLower.includes('per_second')
      if (aIsUtility && !bIsUtility) return 1
      if (bIsUtility && !aIsUtility) return -1
      return a.localeCompare(b)
    })

    return available
  }, [state.metricsData])

  return {
    metricsData: state.metricsData,
    currentMetric: state.currentMetric,
    loadMetrics,
    setCurrentMetric,
    getAvailableMetrics,
  }
}
