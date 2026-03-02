import { useCallback } from 'react'
import { useAppState, useAppDispatch } from '../context/AppContext'
import * as api from '../api/client'

export function useTasks() {
  const state = useAppState()
  const dispatch = useAppDispatch()

  const loadTasks = useCallback(async () => {
    try {
      const tasks = await api.fetchTasks()
      dispatch({ type: 'SET_TASKS_DATA', tasks })
    } catch (error) {
      console.error('Failed to load tasks:', error)
    }
  }, [dispatch])

  return {
    tasksData: state.tasksData,
    loadTasks,
  }
}
