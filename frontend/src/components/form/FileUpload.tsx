import { useRef, useCallback, useState, type DragEvent } from 'react'
import * as api from '../../api/client'

interface FileUploadProps {
  label: string
  required?: boolean
  fileType: 'train' | 'eval'
  filepath: string
  fileInfo: string
  onUploaded: (filepath: string, info: string) => void
  disabled?: boolean
}

export function FileUpload({ label, required, fileType, filepath, fileInfo, onUploaded, disabled }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const isUploaded = !!filepath

  const handleFile = useCallback(async (file: File) => {
    if (uploading) return
    setUploading(true)
    setError(null)
    try {
      const data = await api.uploadFile(file, fileType)
      onUploaded(data.filepath, `${file.name} (${data.row_count} rows)`)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
      // Reset input so re-selecting the same file triggers onChange
      if (inputRef.current) inputRef.current.value = ''
    }
  }, [fileType, onUploaded, uploading])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [handleFile])

  return (
    <div className="upload-section">
      <label className="upload-label">
        {label} {required && <span className="required">*</span>}
        {!required && <span className="optional">(optional)</span>}
      </label>
      <div
        className={`upload-box ${isUploaded ? 'uploaded' : ''} ${uploading ? 'uploading' : ''}`}
        onClick={() => !disabled && !uploading && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        style={dragOver ? { borderColor: 'var(--heather-light)' } : undefined}
      >
        <input
          type="file"
          ref={inputRef}
          accept=".csv,.jsonl"
          hidden
          disabled={disabled}
          onChange={(e) => {
            if (e.target.files?.length) handleFile(e.target.files[0])
          }}
        />
        <span className="upload-icon">{uploading ? '...' : isUploaded ? '✓' : '↑'}</span>
        <span className="upload-text">{uploading ? 'Uploading...' : fileInfo}</span>
      </div>
      {error && <div className="upload-error">{error}</div>}
    </div>
  )
}
