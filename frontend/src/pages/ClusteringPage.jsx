import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import Chart from '../components/Chart'
import ClusterView from '../components/ClusterView'
import DataTable from '../components/DataTable'
import ErrorBox from '../components/ErrorBox'
import PipelineCard from '../components/PipelineCard'
import { adaptDefaults, cleanParams, findModule, initialValues, newId } from '../params'

const SWEEPABLE = ['kmeans', 'hierarchical', 'gmm']

function makePipeline(catalog, dataset, extractorNames, clustererName) {
  const steps = extractorNames.map((name) => ({
    id: newId(),
    name,
    values: initialValues(adaptDefaults(name, findModule(catalog.feature_extractors, name).params, dataset)),
  }))
  const defaults = adaptDefaults(clustererName, findModule(catalog.clusterers, clustererName).params, dataset)
  return { id: newId(), label: '', steps, clusterer: { name: clustererName, values: initialValues(defaults) } }
}

// Form state -> request body. Every pipeline gets a unique name.
function toConfigs(pipelines, catalog) {
  const taken = new Set()
  return pipelines.map((p) => {
    const steps = p.steps.map((s) => ({ name: s.name, params: cleanParams(findModule(catalog.feature_extractors, s.name).params, s.values) }))
    const clusterer = { name: p.clusterer.name, params: cleanParams(findModule(catalog.clusterers, p.clusterer.name).params, p.clusterer.values) }
    let label = p.label.trim() || [...steps.map((s) => s.name), clusterer.name].join('+')
    const base = label
    for (let n = 2; taken.has(label); n++) label = `${base} (${n})`
    taken.add(label)
    return { label, steps, clusterer }
  })
}

export default function ClusteringPage({ dataset }) {
  const [catalog, setCatalog] = useState(null)
  const [pipelines, setPipelines] = useState([])
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)
  const [usedConfigs, setUsedConfigs] = useState([])
  const [selected, setSelected] = useState('')
  const [view, setView] = useState(null)
  const [viewLoading, setViewLoading] = useState(false)

  const [sweepClusterer, setSweepClusterer] = useState('kmeans')
  const [kMin, setKMin] = useState(2)
  const [kMax, setKMax] = useState(8)
  const [sweep, setSweep] = useState(null)
  const [sweepBusy, setSweepBusy] = useState(false)
  const [sweepError, setSweepError] = useState('')

  const latestView = useRef(0) // ignore answers that arrive after a newer request

  // App gives this page a new key per dataset, so state starts fresh whenever the dataset changes.
  useEffect(() => {
    if (!dataset) return
    api
      .modules()
      .then((c) => {
        setCatalog(c)
        setPipelines([makePipeline(c, dataset, ['standard_scaler'], 'kmeans')])
      })
      .catch((e) => setError(e.message))
  }, [dataset])

  if (!dataset) return <div className="notice">Load a dataset on the Data tab first.</div>

  const update = (id, next) => setPipelines((list) => list.map((p) => (p.id === id ? next : p)))

  async function showClusters(label, configs) {
    const cfg = configs.find((c) => c.label === label)
    const ticket = ++latestView.current
    setViewLoading(true)
    setError('')
    try {
      const r = await api.clusterRun({ dataset_id: dataset.dataset_id, steps: cfg.steps, clusterer: cfg.clusterer })
      if (ticket === latestView.current) setView(r)
    } catch (e) {
      if (ticket === latestView.current) setError(e.message)
    } finally {
      if (ticket === latestView.current) setViewLoading(false)
    }
  }

  async function run() {
    setError('')
    setResults(null)
    setView(null)
    setRunning(true)
    try {
      const configs = toConfigs(pipelines, catalog)
      const res = await api.clusterCompare({ dataset_id: dataset.dataset_id, configs })
      setUsedConfigs(configs)
      setResults(res)
      setSelected(res.summary[0].label)
      await showClusters(res.summary[0].label, configs)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  async function runSweep() {
    setSweepError('')
    setSweep(null)
    setSweepBusy(true)
    try {
      const steps = toConfigs(pipelines, catalog)[0].steps // preprocessing of Pipeline 1
      setSweep(await api.clusterSweep({ dataset_id: dataset.dataset_id, clusterer: sweepClusterer, k_min: kMin, k_max: kMax, steps }))
    } catch (e) {
      setSweepError(e.message)
    } finally {
      setSweepBusy(false)
    }
  }

  const failed = results ? results.summary.filter((r) => r.n_clusters < 2).map((r) => r.label) : []
  const scored = sweep ? sweep.sweep.filter((r) => r.silhouette !== null) : []
  const bestK = scored.length ? scored.reduce((a, b) => (b.silhouette > a.silhouette ? b : a)).k : null
  const ks = sweep ? sweep.sweep.map((r) => r.k) : []
  const line = (title, key, colour) => (
    <Chart
      height={280}
      data={[{ type: 'scatter', mode: 'lines+markers', x: ks, y: sweep.sweep.map((r) => r[key]), line: { color: colour } }]}
      layout={{ title: { text: title }, xaxis: { title: { text: 'k (number of clusters)' }, dtick: 1 } }}
    />
  )

  return (
    <>
      {!catalog && !error && <div className="notice">Loading modules…</div>}
      <ErrorBox message={error} />

      {catalog && (
        <>
          <div className="card">
            <h2>Build clustering pipelines</h2>
            <p>
              Clustering never sees the class labels.{' '}
              {dataset.has_labels
                ? `This dataset has real classes, so the results are also compared with them afterwards. A good first number of clusters is the number of real classes (${dataset.n_classes}).`
                : 'This dataset has no labels, so only the shape-based scores apply. Use the k-sweep below to choose the number of clusters.'}
            </p>
            <div className="row" style={{ marginBottom: 14 }}>
              <button className="btn secondary" onClick={() => setPipelines((l) => [...l, makePipeline(catalog, dataset, ['standard_scaler'], 'kmeans')])}>+ Add pipeline</button>
              <button className="btn secondary" onClick={() => setPipelines(catalog.clusterers.map((c) => makePipeline(catalog, dataset, ['standard_scaler'], c.name)))}>
                Quick start: compare every clusterer
              </button>
            </div>

            {pipelines.map((p, i) => (
              <PipelineCard
                key={p.id}
                pipeline={p}
                index={i}
                catalog={catalog}
                dataset={dataset}
                canRemove={pipelines.length > 1}
                modelKey="clusterer"
                modelOptions={catalog.clusterers}
                modelTitle="Clustering algorithm"
                onChange={(next) => update(p.id, next)}
                onRemove={() => setPipelines((l) => l.filter((x) => x.id !== p.id))}
              />
            ))}

            <button className="btn" onClick={run} disabled={running || pipelines.length === 0}>
              {running ? 'Clustering…' : `Run clustering (${pipelines.length} pipeline${pipelines.length === 1 ? '' : 's'})`}
            </button>
          </div>

          {results && (
            <>
              <div className="card">
                <h2>Comparison</h2>
                {failed.length > 0 && (
                  <div className="notice">
                    {failed.join(', ')}: found fewer than 2 clusters (every point labelled noise, or everything in one cluster), so the quality scores are blank. For DBSCAN, try a different <code>eps</code>, or add a PCA step before it so distances are measured in fewer dimensions.
                  </div>
                )}
                {results.summary.length > 1 && (
                  <Chart
                    height={Math.max(260, results.summary.length * 60 + 100)}
                    data={[
                      { type: 'bar', orientation: 'h', name: 'Silhouette', y: results.summary.map((r) => r.label), x: results.summary.map((r) => r.silhouette), marker: { color: '#3b5bdb' } },
                      ...(dataset.has_labels
                        ? [{ type: 'bar', orientation: 'h', name: 'Adjusted Rand', y: results.summary.map((r) => r.label), x: results.summary.map((r) => r.adjusted_rand ?? null), marker: { color: '#e8590c' } }]
                        : []),
                    ]}
                    layout={{ barmode: 'group', yaxis: { autorange: 'reversed' }, legend: { orientation: 'h', y: 1.15 }, margin: { l: 200, r: 20, t: 40, b: 40 } }}
                  />
                )}
                <DataTable
                  columns={[
                    { key: 'label', label: 'Pipeline' }, { key: 'n_clusters', label: 'Clusters' }, { key: 'n_noise', label: 'Noise' },
                    { key: 'silhouette', label: 'Silhouette' }, { key: 'davies_bouldin', label: 'Davies-Bouldin' }, { key: 'calinski_harabasz', label: 'Calinski-Harabasz' },
                    { key: 'adjusted_rand', label: 'Adj. Rand' }, { key: 'nmi', label: 'NMI' }, { key: 'v_measure', label: 'V-measure' }, { key: 'seconds', label: 'Time (s)' },
                  ]}
                  rows={results.summary}
                />
              </div>

              <div className="card">
                <div className="row" style={{ justifyContent: 'space-between' }}>
                  <h2>Look at the clusters</h2>
                  <label className="field">
                    Pipeline
                    <select
                      value={selected}
                      onChange={(e) => {
                        setSelected(e.target.value)
                        showClusters(e.target.value, usedConfigs)
                      }}
                    >
                      {results.summary.map((r) => <option key={r.label} value={r.label}>{r.label}</option>)}
                    </select>
                  </label>
                </div>
                {viewLoading && <div className="notice">Drawing…</div>}
                {view && <ClusterView view={view} totalSamples={dataset.n_samples} />}
              </div>
            </>
          )}

          <div className="card">
            <h2>How many clusters? (k-sweep)</h2>
            <p>Tries a range of cluster counts using the preprocessing steps of Pipeline 1. Look for an elbow in inertia, a peak in silhouette and a dip in Davies-Bouldin.</p>
            <ErrorBox message={sweepError} />
            <div className="row">
              <label className="field">
                Algorithm
                <select value={sweepClusterer} onChange={(e) => setSweepClusterer(e.target.value)}>
                  {SWEEPABLE.map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </label>
              <label className="field">
                From k
                <input type="number" min={2} max={30} value={kMin} onChange={(e) => setKMin(Number(e.target.value))} />
              </label>
              <label className="field">
                To k
                <input type="number" min={2} max={30} value={kMax} onChange={(e) => setKMax(Number(e.target.value))} />
              </label>
              <button className="btn" onClick={runSweep} disabled={sweepBusy} style={{ alignSelf: 'flex-end' }}>
                {sweepBusy ? 'Sweeping…' : 'Run k-sweep'}
              </button>
            </div>

            {sweep && (
              <div style={{ marginTop: 16 }}>
                {bestK !== null && (
                  <div className="notice">
                    Highest silhouette at k = {bestK}. Treat that as a suggestion rather than a rule, and check the other two charts too.
                  </div>
                )}
                <div className="charts">
                  {sweep.sweep.some((r) => r.inertia !== null) && line('Inertia (look for the elbow)', 'inertia', '#3b5bdb')}
                  {line('Silhouette (higher is better)', 'silhouette', '#2b8a3e')}
                  {line('Davies-Bouldin (lower is better)', 'davies_bouldin', '#c92a2a')}
                </div>
                <DataTable
                  columns={[{ key: 'k', label: 'k' }, { key: 'inertia', label: 'Inertia' }, { key: 'silhouette', label: 'Silhouette' },
                            { key: 'davies_bouldin', label: 'Davies-Bouldin' }, { key: 'calinski_harabasz', label: 'Calinski-Harabasz' }]}
                  rows={sweep.sweep}
                />
              </div>
            )}
          </div>
        </>
      )}
    </>
  )
}
