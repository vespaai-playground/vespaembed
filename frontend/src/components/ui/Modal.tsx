import { useEffect, useCallback, type ReactNode } from 'react'

interface ModalProps {
  show: boolean
  onClose: () => void
  children: ReactNode
  className?: string
  closeOnBackdrop?: boolean
}

export function Modal({ show, onClose, children, className = '', closeOnBackdrop = true }: ModalProps) {
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    if (show) {
      document.addEventListener('keydown', handleKeyDown)
      return () => document.removeEventListener('keydown', handleKeyDown)
    }
  }, [show, handleKeyDown])

  if (!show) return null

  return (
    <div
      className="modal"
      style={{ display: 'flex' }}
      onClick={(e) => { if (closeOnBackdrop && e.target === e.currentTarget) onClose() }}
    >
      <div className={`modal-box ${className}`}>
        {children}
      </div>
    </div>
  )
}

interface ModalHeaderProps {
  title: string
  onClose: () => void
}

export function ModalHeader({ title, onClose }: ModalHeaderProps) {
  return (
    <div className="modal-header">
      <span>{title}</span>
      <button className="close-modal" onClick={onClose}>&times;</button>
    </div>
  )
}
