import { useEffect, useRef } from 'react'

interface LogCardProps {
  logs: string[]
}

export function LogCard({ logs }: LogCardProps) {
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [logs])

  return (
    <div className="log-card">
      <div className="card-header">
        Training Logs
        <span className="badge">{logs.length}</span>
      </div>
      <div className="log-content" ref={contentRef}>
        {logs.length > 0 ? logs.join('\n') : 'Logs will appear here...'}
      </div>
    </div>
  )
}
