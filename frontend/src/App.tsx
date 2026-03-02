import { AppProvider } from './context/AppContext'
import { Sidebar } from './components/layout/Sidebar'
import { MainContent } from './components/layout/MainContent'
import { Footer } from './components/layout/Footer'
import { NewProjectModal } from './components/modals/NewProjectModal'
import { ArtifactsModal } from './components/modals/ArtifactsModal'

export default function App() {
  return (
    <AppProvider>
      <div className="app">
        <Sidebar />
        <MainContent />
        <Footer />
      </div>
      <NewProjectModal />
      <ArtifactsModal />
    </AppProvider>
  )
}
