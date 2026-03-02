import type { TaskInfo } from '../../api/types'

interface TaskSelectorProps {
  tasksData: TaskInfo[]
  selectedTask: string
  onTaskChange: (taskName: string) => void
}

export function TaskSelector({ tasksData, selectedTask, onTaskChange }: TaskSelectorProps) {
  const task = tasksData.find(t => t.name === selectedTask)

  return (
    <div className="form-section">
      <label className="section-label">Task &amp; Model</label>
      <div className="form-field">
        <label>Task</label>
        <select
          id="task"
          value={selectedTask}
          onChange={(e) => onTaskChange(e.target.value)}
          required
        >
          {tasksData.map(t => (
            <option key={t.name} value={t.name}>{t.name.toUpperCase()}</option>
          ))}
        </select>
        {task && <span className="field-hint">{task.description}</span>}
      </div>
      {task && task.expected_columns && (
        <div className="required-columns">
          <span className="required-columns-label">Required columns:</span>
          <span className="required-columns-list">
            {task.expected_columns.map(col => (
              <span key={col} className="required-column-tag">{col}</span>
            ))}
          </span>
        </div>
      )}
      {task && task.sample_data && task.sample_data.length > 0 && (
        <div className="sample-data-box">
          <div className="sample-data-header">Example data:</div>
          <div className="sample-data-content">
            {task.expected_columns.map(col => {
              const value = task.sample_data[0][col]
              const displayValue = typeof value === 'string'
                ? (value.length > 50 ? value.substring(0, 50) + '...' : value)
                : String(value)
              return (
                <span key={col} className="sample-data-row">
                  <span className="sample-data-key">{col}:</span>{' '}
                  <span className="sample-data-value">"{displayValue}"</span>
                </span>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
