import type { ReactNode } from 'react'

interface DataSourceTabsProps {
  currentTab: 'file' | 'huggingface'
  onTabChange: (tab: 'file' | 'huggingface') => void
  fileContent: ReactNode
  hfContent: ReactNode
}

export function DataSourceTabs({ currentTab, onTabChange, fileContent, hfContent }: DataSourceTabsProps) {
  return (
    <div className="form-section">
      <label className="section-label">Data Source</label>
      <div className="tabs">
        <button
          type="button"
          className={`tab ${currentTab === 'file' ? 'active' : ''}`}
          onClick={() => onTabChange('file')}
        >
          File Upload
        </button>
        <button
          type="button"
          className={`tab ${currentTab === 'huggingface' ? 'active' : ''}`}
          onClick={() => onTabChange('huggingface')}
        >
          HuggingFace Dataset
        </button>
      </div>
      <div className="tab-container">
        <div className={`tab-content ${currentTab === 'file' ? 'active' : ''}`}>
          {fileContent}
        </div>
        <div className={`tab-content ${currentTab === 'huggingface' ? 'active' : ''}`}>
          {hfContent}
        </div>
      </div>
    </div>
  )
}
