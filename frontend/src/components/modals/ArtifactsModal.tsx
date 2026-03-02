import { useState, useEffect, useCallback, useRef } from 'react'
import { useAppState, useAppDispatch } from '../../context/AppContext'
import { Modal, ModalHeader } from '../ui/Modal'
import * as api from '../../api/client'
import type { Artifact } from '../../api/types'

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
}

export function ArtifactsModal() {
  const { showArtifactsModal, selectedRunId } = useAppState()
  const dispatch = useAppDispatch()
  const [artifacts, setArtifacts] = useState<Artifact[]>([])
  const [outputDir, setOutputDir] = useState('')
  const [copiedPath, setCopiedPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const copyTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const close = useCallback(() => {
    dispatch({ type: 'SET_SHOW_ARTIFACTS_MODAL', show: false })
  }, [dispatch])

  useEffect(() => {
    if (showArtifactsModal && selectedRunId) {
      setLoading(true)
      api.fetchArtifacts(selectedRunId).then(data => {
        setArtifacts(data.artifacts)
        setOutputDir(data.output_dir || '')
      }).catch(err => {
        console.error('Failed to load artifacts:', err)
        setArtifacts([])
      }).finally(() => setLoading(false))
    }
  }, [showArtifactsModal, selectedRunId])

  // Cleanup copy timeout on unmount
  useEffect(() => {
    return () => {
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current)
    }
  }, [])

  const copyPath = useCallback((path: string) => {
    navigator.clipboard.writeText(path).then(() => {
      setCopiedPath(path)
      if (copyTimeoutRef.current) clearTimeout(copyTimeoutRef.current)
      copyTimeoutRef.current = setTimeout(() => setCopiedPath(null), 1000)
    }).catch(err => console.error('Failed to copy:', err))
  }, [])

  return (
    <Modal show={showArtifactsModal} onClose={close}>
      <ModalHeader title="Artifacts" onClose={close} />
      <div className="modal-body">
        <div className="artifacts-list">
          {loading ? (
            <div className="artifacts-empty">Loading artifacts...</div>
          ) : artifacts.length === 0 ? (
            <div className="artifacts-empty">No artifacts available</div>
          ) : (
            artifacts.map(artifact => (
              <div key={artifact.name} className="artifact-item">
                <div className="artifact-info">
                  <span className="artifact-name">{artifact.label}</span>
                  <div className="artifact-meta">
                    <span className="artifact-category">{artifact.category}</span>
                    <span>{formatFileSize(artifact.size)}</span>
                  </div>
                </div>
                <button
                  className="artifact-download"
                  onClick={() => copyPath(artifact.path)}
                >
                  {copiedPath === artifact.path ? 'Copied!' : 'Copy Path'}
                </button>
              </div>
            ))
          )}
        </div>
        {outputDir && (
          <div className="artifacts-footer">
            <span className="artifacts-path">{outputDir}</span>
          </div>
        )}
      </div>
    </Modal>
  )
}
