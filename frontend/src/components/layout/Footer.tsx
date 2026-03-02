import { useEffect } from 'react'
import { useAppState, useAppDispatch } from '../../context/AppContext'
import * as api from '../../api/client'

export function Footer() {
  const { version } = useAppState()
  const dispatch = useAppDispatch()

  useEffect(() => {
    api.fetchVersion()
      .then(data => dispatch({ type: 'SET_VERSION', version: data.version }))
      .catch(() => {})
  }, [dispatch])

  return (
    <footer className="app-footer">
      <span>vespaembed v{version}</span>
    </footer>
  )
}
