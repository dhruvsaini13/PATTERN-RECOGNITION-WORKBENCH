import { useEffect, useState } from 'react'
import { api } from './api'
import DataPage from './pages/DataPage'
import ClassificationPage from './pages/ClassificationPage'
import ClusteringPage from './pages/ClusteringPage'
import StatsPage from './pages/StatsPage'

const TABS = [
  { id: 'data', label: '1 · Data' },
  { id: 'stats', label: '2 · Statistics' },
  { id: 'classification', label: '3 · Classification' },
  { id: 'clustering', label: '4 · Clustering' },
]

export default function App() {
  const [tab, setTab] = useState('data')
  const [visited, setVisited] = useState({ data: true })
  const [dataset, setDataset] = useState(null) // the dataset every page works on
  const [backendUp, setBackendUp] = useState(false)

  function openTab(id) {
    setTab(id)
    setVisited((v) => ({ ...v, [id]: true }))
  }

  useEffect(() => {
    const check = () => api.health().then(() => setBackendUp(true)).catch(() => setBackendUp(false))
    check()
    const timer = setInterval(check, 10000)
    return () => clearInterval(timer)
  }, [])

  return (
    <>
      <header className="app-header">
        <h1>Pattern Recognition Workbench</h1>
        <span className={backendUp ? 'status ok' : 'status'}>
          {backendUp ? 'Backend connected' : 'Backend offline: start uvicorn'}
        </span>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={t.id === tab ? 'tab active' : 'tab'} onClick={() => openTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {/* A page is created the first time you open its tab, then kept (just hidden) so your work isn't lost. */}
        <div hidden={tab !== 'data'}>
          <DataPage dataset={dataset} setDataset={setDataset} />
        </div>
        {visited.stats && (
          <div hidden={tab !== 'stats'}>
            <StatsPage dataset={dataset} />
          </div>
        )}
        {visited.classification && (
          <div hidden={tab !== 'classification'}>
            <ClassificationPage key={dataset?.dataset_id} dataset={dataset} />
          </div>
        )}
        {visited.clustering && (
          <div hidden={tab !== 'clustering'}>
            <ClusteringPage key={dataset?.dataset_id} dataset={dataset} />
          </div>
        )}
      </main>
    </>
  )
}
