interface StatusChipProps {
  status: string
  small?: boolean
}

export function StatusChip({ status, small }: StatusChipProps) {
  return (
    <span className={`status-chip${small ? ' small' : ''} ${status}`}>
      {status}
    </span>
  )
}
