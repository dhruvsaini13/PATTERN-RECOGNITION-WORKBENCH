import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import ErrorBox from '../components/ErrorBox'
import ModelPanel from '../components/ModelPanel'
import PipelineCard from '../components/PipelineCard'
import ResultsPanel from '../components/ResultsPanel'
import { adaptDefaults, cleanParams, findModule, initialValues, newId } from '../params'

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))

function makePipeline(catalog, dataset, extractorNames, classifierName) {
  const steps = extractorNames.map((name) => ({
    id: newId(),
    name,
    values: initialValues(adaptDefaults(name, findModule(catalog.feature_extractors, name).params, dataset)),
  }))
  return { id: newId(), label: '', steps, classifier: { name: classifierName, values: initialValues(findModule(catalog.classifiers, classifierName).params) } }
}

// Turn the form state into the request body the API expects. Every pipeline gets a unique name.
function toConfigs(pipelines, catalog) {
  const taken = new Set()
  return pipelines.map((p) => {
    const steps = p.steps.map((s) => ({
      name: s.name,
      params: cleanParams(adaptDefaults(s.name, findModule(catalog.feature_extractors, s.name).params, null), s.values),
    }))
    const classifier = { name: p.classifier.name, params: cleanParams(findModule(catalog.classifiers, p.classifier.name).params, p.classifier.values) }
    let label = p.label.trim() || [...steps.map((s) => s.name), classifier.name].join('+')
    const base = label
    for (let n = 2; taken.has(label); n++) label = `${base} (${n})`
    taken.add(label)
    return { label, steps, classifier }
  })
}

export default function ClassificationPage({ dataset }) {
  const [catalog, setCatalog] = useState(null)
  const [pipelines, setPipelines] = useState([])
  const [testSize, setTestSize] = useState(0.25)
  const [cvSplits, setCvSplits] = useState(5)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)
  const [selected, setSelected] = useState('')
  const [usedConfigs, setUsedConfigs] = useState([])
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  // App gives this page a new key per dataset, so state starts fresh whenever the dataset changes.
  useEffect(() => {
    if (!dataset) return
    api
      .modules()
      .then((c) => {
        setCatalog(c)
        setPipelines([makePipeline(c, dataset, ['standard_scaler'], 'svm')])
      })
      .catch((e) => setError(e.message))
  }, [dataset])

  if (!dataset) return <div className="notice">Load a dataset on the Data tab first.</div>
  if (!dataset.has_labels) {
    return <div className="notice">This dataset has no labels, so it can’t be used for classification. Upload it again with a target column, or use the Clustering tab.</div>
  }

  const update = (id, next) => setPipelines((list) => list.map((p) => (p.id === id ? next : p)))

  async function run() {
    setError('')
    setResults(null)
    setRunning(true)
    try {
      const configs = toConfigs(pipelines, catalog)
      const { job_id } = await api.submitCompareJob({ dataset_id: dataset.dataset_id, configs, test_size: testSize, cv_splits: cvSplits })
      for (;;) {
        const job = await api.job(job_id)
        if (!alive.current) return
        if (job.status === 'done') {
          setUsedConfigs(configs)
          setResults(job.result)
          setSelected(job.result.summary[0].label)
          break
        }
        if (job.status === 'error') throw new Error(job.error)
        await sleep(700)
      }
    } catch (e) {
      if (alive.current) setError(e.message)
    } finally {
      if (alive.current) setRunning(false)
    }
  }

  const selectedConfig = usedConfigs.find((c) => c.label === selected)

  return (
    <>
      {!catalog && !error && <div className="notice">Loading modules…</div>}
      <ErrorBox message={error} />

      {catalog && (
        <>
          <div className="card">
            <h2>Build pipelines</h2>
            <p>
              Each pipeline is a chain of preprocessing steps followed by a classifier. Add as many as you like and compare them on {dataset.name} ({dataset.n_samples} samples, {dataset.n_features} features, {dataset.n_classes} classes).
            </p>
            <div className="row" style={{ marginBottom: 14 }}>
              <button className="btn secondary" onClick={() => setPipelines((l) => [...l, makePipeline(catalog, dataset, ['standard_scaler'], 'svm')])}>+ Add pipeline</button>
              <button
                className="btn secondary"
                onClick={() => setPipelines(catalog.classifiers.map((c) => makePipeline(catalog, dataset, ['standard_scaler'], c.name)))}
              >
                Quick start: compare every classifier
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
                onChange={(next) => update(p.id, next)}
                onRemove={() => setPipelines((l) => l.filter((x) => x.id !== p.id))}
              />
            ))}

            <div className="row" style={{ marginTop: 8 }}>
              <label className="field">
                Test set size
                <select value={testSize} onChange={(e) => setTestSize(Number(e.target.value))}>
                  {[0.2, 0.25, 0.3, 0.4].map((t) => <option key={t} value={t}>{Math.round(t * 100)}%</option>)}
                </select>
              </label>
              <label className="field">
                Cross-validation folds
                <select value={cvSplits} onChange={(e) => setCvSplits(Number(e.target.value))}>
                  {[3, 5, 10].map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
              <button className="btn" onClick={run} disabled={running || pipelines.length === 0} style={{ alignSelf: 'flex-end' }}>
                {running ? 'Running… (this can take a few seconds)' : `Run comparison (${pipelines.length} pipeline${pipelines.length === 1 ? '' : 's'})`}
              </button>
            </div>
          </div>

          {results && (
            <>
              <ResultsPanel results={results} selected={selected} setSelected={setSelected} />
              {selectedConfig && <ModelPanel key={selected} dataset={dataset} config={selectedConfig} testSize={testSize} />}
            </>
          )}
        </>
      )}
    </>
  )
}
