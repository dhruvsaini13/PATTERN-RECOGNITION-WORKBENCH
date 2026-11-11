import { useEffect, useState } from 'react'
import { api } from '../api'
import DataTable from '../components/DataTable'
import ErrorBox from '../components/ErrorBox'

export default function DataPage({ dataset, setDataset }) {
  const [builtin, setBuiltin] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [preview, setPreview] = useState(null)

  // CSV upload form
  const [file, setFile] = useState(null)
  const [target, setTarget] = useState('')
  const [dropCols, setDropCols] = useState('')
  const [missing, setMissing] = useState('median')
  const [encode, setEncode] = useState(true)

  useEffect(() => {
    api.builtinNames().then((r) => setBuiltin(r.datasets)).catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!dataset) return
    api.preview(dataset.dataset_id, 10).then(setPreview).catch((e) => setError(e.message))
  }, [dataset])

  async function run(action) {
    setBusy(true)
    setError('')
    try {
      setDataset(await action())
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  function uploadFile() {
    const form = new FormData()
    form.append('file', file)
    if (target.trim()) form.append('target_column', target.trim())
    form.append('drop_columns', dropCols)
    form.append('missing', missing)
    form.append('encode_categorical', String(encode))
    run(() => api.uploadCsv(form))
  }

  return (
    <>
      <ErrorBox message={error} />

      <div className="card">
        <h2>Built-in datasets</h2>
        <p>Click one to load it into the workbench.</p>
        <div className="row">
          {builtin.map((name) => (
            <button key={name} className="btn secondary" disabled={busy} onClick={() => run(() => api.loadBuiltin(name))}>
              {name}
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <h2>Upload your own CSV</h2>
        <p>Leave “target column” empty for unlabeled data (you can still do statistics and clustering).</p>
        <div className="row">
          <label className="field">
            CSV file
            <input type="file" accept=".csv,text/csv" onChange={(e) => setFile(e.target.files[0] || null)} />
          </label>
          <label className="field">
            Target column
            <input type="text" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. label" />
          </label>
          <label className="field">
            Columns to drop (comma-separated)
            <input type="text" value={dropCols} onChange={(e) => setDropCols(e.target.value)} placeholder="e.g. id, name" />
          </label>
          <label className="field">
            Missing values
            <select value={missing} onChange={(e) => setMissing(e.target.value)}>
              <option value="median">fill with median</option>
              <option value="mean">fill with mean</option>
              <option value="zero">fill with zero</option>
              <option value="drop">drop those rows</option>
            </select>
          </label>
          <label className="field">
            Text columns
            <select value={String(encode)} onChange={(e) => setEncode(e.target.value === 'true')}>
              <option value="true">one-hot encode</option>
              <option value="false">drop them</option>
            </select>
          </label>
          <button className="btn" disabled={!file || busy} onClick={uploadFile}>
            {busy ? 'Working…' : 'Upload'}
          </button>
        </div>
      </div>

      {dataset && (
        <div className="card">
          <h2>Active dataset: {dataset.name}</h2>
          <div className="grid">
            <div className="stat"><div className="label">Samples</div><div className="value">{dataset.n_samples}</div></div>
            <div className="stat"><div className="label">Features</div><div className="value">{dataset.n_features}</div></div>
            <div className="stat"><div className="label">Classes</div><div className="value">{dataset.has_labels ? dataset.n_classes : '—'}</div></div>
            <div className="stat"><div className="label">Labels</div><div className="value">{dataset.has_labels ? 'yes' : 'no'}</div></div>
          </div>
          <h3 style={{ marginTop: 18 }}>First rows</h3>
          {preview && (
            <DataTable columns={preview.columns.map((c) => ({ key: c, label: c }))} rows={preview.rows} />
          )}
        </div>
      )}
    </>
  )
}
