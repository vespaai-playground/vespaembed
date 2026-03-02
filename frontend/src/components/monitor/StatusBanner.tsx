interface StatusBannerProps {
  message: string | null
}

export function StatusBanner({ message }: StatusBannerProps) {
  if (!message) return null

  return (
    <div className="status-banner" style={{ display: 'flex' }}>
      <span className="status-spinner" />
      <span className="status-message">{message}</span>
    </div>
  )
}
