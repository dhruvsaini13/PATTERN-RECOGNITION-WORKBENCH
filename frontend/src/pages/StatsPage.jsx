import { useEffect, useState } from 'react'
import { api } from '../api'
import Chart from '../components/Chart'
import DataTable from '../components/DataTable'
import ErrorBox from '../components/ErrorBox'

// Charts drawn by the backend (matplotlib) and shown as images, loaded on demand.
function ServerPlot({ datasetId, kind, title }) {
  const [image, setImage] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  async function load() {
    setLoading(true)
    setError('')
    try {
      const r = await api.datasetPlot(datasetId, kind)
      setImage(`data:${r.mime};base64,${r.image_base64}`)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <div className="row">
        <h3 style={{ margin: 0 }}>{title}</h3>
        <button className="btn secondary" onClick={load} disabled={loading}>
          {loading ? 'Drawing…' : image ? 'Redraw' : 'Show'}
        </button>
      </div>
      <ErrorBox message={error} />
      {image && <img className="plot-img" src={image} alt={title} style={{ marginTop: 10 }} />}
    </div>
  )
}

export default function StatsPage({ dataset }) {
  const [threshold, setThreshold] = useState(0.9)
  const [result, setResult] = useState({ key: null, datasetId: null, report: null, error: '' })

  const datasetId = dataset?.dataset_id
  const key = `${datasetId}|${threshold}`

  useEffect(() => {
    if (!datasetId) return
    let cancelled = false
    api
      .analysis(datasetId, threshold)
      .then((r) => !cancelled && setResult({ key, datasetId, report: r, error: '' }))
      .catch((e) => !cancelled && setResult({ key, datasetId, report: null, error: e.message }))
    return () => {
      cancelled = true
    }
  }, [datasetId, threshold, key])

  const report = result.datasetId === datasetId ? result.report : null // never show another dataset's numbers
  const error = result.key === key ? result.error : ''
  const loading = Boolean(datasetId) && result.key !== key

  if (!dataset) return <div className="notice">Load a dataset on the Data tab first.</div>

  const ranking = report?.feature_ranking
  const topRanked = ranking ? ranking.slice(0, 15) : []

  return (
    <>
      <ErrorBox message={error} />
      {loading && !report && <div className="notice">Analysing…</div>}

      {report && (
        <>
          <div className="card">
            <h2>Overview: {report.dataset.name}</h2>
            <div className="grid">
              <div className="stat"><div className="label">Samples</div><div className="value">{report.dataset.n_samples}</div></div>
              <div className="stat"><div className="label">Features</div><div className="value">{report.dataset.n_features}</div></div>
              {report.class_balance && (
                <>
                  <div className="stat"><div className="label">Classes</div><div className="value">{report.dataset.n_classes}</div></div>
                  <div className="stat"><div className="label">Imbalance ratio</div><div className="value">{report.class_balance.imbalance_ratio}</div></div>
                </>
              )}
            </div>
          </div>

          {report.class_balance && (
            <div className="card">
              <h2>Class balance</h2>
              <p>A ratio near 1 means classes are evenly represented. Large ratios call for care when reading accuracy.</p>
              <Chart
                height={300}
                data={[{ type: 'bar', x: report.class_balance.classes, y: report.class_balance.counts,
                         text: report.class_balance.percent.map((p) => `${p}%`), marker: { color: '#3b5bdb' } }]}
                layout={{ yaxis: { title: { text: 'samples' } } }}
              />
            </div>
          )}

          {ranking && (
            <div className="card">
              <h2>Feature ranking</h2>
              <p>Ranked by three methods (ANOVA F-score, mutual information, random-forest importance); lower mean rank is better.</p>
              <Chart
                height={Math.max(300, topRanked.length * 24 + 80)}
                data={[{ type: 'bar', orientation: 'h', x: [...topRanked].reverse().map((r) => r.rf_importance),
                         y: [...topRanked].reverse().map((r) => r.feature), marker: { color: '#3b5bdb' } }]}
                layout={{ title: { text: 'Top features by random-forest importance' }, margin: { l: 180, r: 20, t: 45, b: 45 } }}
              />
              <DataTable
                columns={[
                  { key: 'feature', label: 'Feature' }, { key: 'f_score', label: 'F-score' },
                  { key: 'mutual_info', label: 'Mutual info' }, { key: 'rf_importance', label: 'RF importance' },
                  { key: 'mean_rank', label: 'Mean rank' },
                ]}
                rows={ranking}
              />
            </div>
          )}

          <div className="card">
            <h2>Summary statistics</h2>
            <DataTable
              columns={[
                { key: 'feature', label: 'Feature' }, { key: 'mean', label: 'Mean' }, { key: 'std', label: 'Std' },
                { key: 'min', label: 'Min' }, { key: '50%', label: 'Median' }, { key: 'max', label: 'Max' },
                { key: 'skew', label: 'Skew' }, { key: 'kurtosis', label: 'Kurtosis' }, { key: 'missing', label: 'Missing' },
              ]}
              rows={report.summary}
            />
          </div>

          <div className="card">
            <h2>Outliers (IQR method)</h2>
            <Chart
              height={320}
              data={[{ type: 'bar', x: report.outliers_iqr.map((o) => o.feature), y: report.outliers_iqr.map((o) => o.n_outliers),
                       marker: { color: '#e8590c' } }]}
              layout={{ yaxis: { title: { text: 'outlying samples' } }, xaxis: { tickangle: -45 } }}
            />
          </div>

          <div className="card">
            <h2>Normality tests</h2>
            <p>Shapiro-Wilk (or D’Agostino-Pearson for large datasets). “yes” means the feature looks normally distributed at the 5% level.</p>
            <DataTable
              columns={[
                { key: 'feature', label: 'Feature' }, { key: 'test', label: 'Test' },
                { key: 'statistic', label: 'Statistic' }, { key: 'p_value', label: 'p-value' }, { key: 'is_normal', label: 'Normal?' },
              ]}
              rows={report.normality}
            />
          </div>

          <div className="card">
            <div className="row" style={{ justifyContent: 'space-between' }}>
              <h2>Highly correlated feature pairs</h2>
              <label className="field">
                Minimum |correlation|
                <select value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}>
                  {[0.5, 0.7, 0.8, 0.9, 0.95].map((t) => <option key={t} value={t}>{t}</option>)}
                </select>
              </label>
            </div>
            <DataTable
              columns={[{ key: 'feature_a', label: 'Feature A' }, { key: 'feature_b', label: 'Feature B' }, { key: 'correlation', label: 'Correlation' }]}
              rows={report.high_correlations}
            />
          </div>

          <div className="card">
            <h2>More plots</h2>
            <ServerPlot key={`${datasetId}-histograms`} datasetId={datasetId} kind="histograms" title="Feature distributions" />
            <ServerPlot key={`${datasetId}-correlation`} datasetId={datasetId} kind="correlation" title="Correlation heatmap" />
            <ServerPlot key={`${datasetId}-boxplots`} datasetId={datasetId} kind="boxplots" title="Boxplots" />
            <ServerPlot key={`${datasetId}-pca`} datasetId={datasetId} kind="pca" title="2D PCA projection" />
          </div>
        </>
      )}
    </>
  )
}
