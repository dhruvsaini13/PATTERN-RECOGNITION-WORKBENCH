import { useState } from 'react'
import { api } from '../api'
import Chart from './Chart'
import ErrorBox from './ErrorBox'

// Train the selected pipeline as a reusable model, look at its ROC curve, and ask it for predictions.
export default function ModelPanel({ dataset, config, testSize }) {
  const [model, setModel] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [roc, setRoc] = useState(null)
  const [text, setText] = useState('')
  const [prediction, setPrediction] = useState(null)

  async function train() {
    setBusy(true)
    setError('')
    setRoc(null)
    setPrediction(null)
    try {
      setModel(await api.train({ dataset_id: dataset.dataset_id, steps: config.steps, classifier: config.classifier, test_size: testSize }))
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  async function showRoc() {
    setError('')
    try {
      const r = await api.modelPlot(model.model_id, 'roc')
      setRoc(`data:${r.mime};base64,${r.image_base64}`)
    } catch (e) {
      setError(e.message)
    }
  }

  async function fillSample() {
    setError('')
    try {
      const preview = await api.preview(dataset.dataset_id, 1)
      const row = preview.rows[0]
      setText(model.feature_names.map((f) => row[f]).join(', '))
    } catch (e) {
      setError(e.message)
    }
  }

  async function predict() {
    setError('')
    setPrediction(null)
    const numbers = text.split(/[\s,;]+/).filter(Boolean).map(Number)
    if (numbers.length !== model.n_features || numbers.some(Number.isNaN)) {
      setError(`Enter exactly ${model.n_features} numbers separated by commas (one per feature).`)
      return
    }
    try {
      const r = await api.predict(model.model_id, [numbers])
      setPrediction({ label: r.labels[0], probabilities: r.probabilities ? r.probabilities[0] : null })
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="card">
      <h2>Train and predict</h2>
      <p>Train the pipeline currently selected above as a model, then try it on a new sample.</p>
      <ErrorBox message={error} />
      <button className="btn" onClick={train} disabled={busy}>{busy ? 'Training…' : 'Train this pipeline'}</button>

      {model && (
        <div style={{ marginTop: 16 }}>
          <div className="notice">
            Model <code>{model.model_id}</code> trained: test accuracy {model.metrics.accuracy.toFixed(3)}, macro F1 {model.metrics.f1_macro.toFixed(3)}.
          </div>

          <div className="row" style={{ marginBottom: 12 }}>
            <button className="btn secondary" onClick={showRoc}>Show ROC curve</button>
          </div>
          {roc && <img className="plot-img" src={roc} alt="ROC curve" style={{ marginBottom: 14 }} />}

          <h3>Predict a sample</h3>
          <p className="muted">Needs {model.n_features} numbers, in this feature order: {model.feature_names.join(', ')}</p>
          <textarea rows={3} value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. 13.2, 1.78, 2.14, …" style={{ width: '100%', padding: 8, fontFamily: 'monospace' }} />
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn secondary" onClick={fillSample}>Use the dataset’s first row</button>
            <button className="btn" onClick={predict} disabled={!text.trim()}>Predict</button>
          </div>

          {prediction && (
            <div style={{ marginTop: 14 }}>
              <div className="stat" style={{ display: 'inline-block' }}>
                <div className="label">Predicted class</div>
                <div className="value">{prediction.label}</div>
              </div>
              {prediction.probabilities && (
                <Chart
                  height={280}
                  data={[{ type: 'bar', x: model.class_names, y: prediction.probabilities, marker: { color: '#3b5bdb' } }]}
                  layout={{ title: { text: 'Class probabilities' }, yaxis: { range: [0, 1] } }}
                />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
