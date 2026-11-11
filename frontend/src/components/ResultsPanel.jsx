import { useState } from 'react'
import Chart from './Chart'
import DataTable from './DataTable'

export default function ResultsPanel({ results, selected, setSelected }) {
  const [normalize, setNormalize] = useState(false)
  const { summary, metrics } = results
  const m = metrics[selected] ?? metrics[summary[0].label]
  const names = m.class_names

  const cm = normalize
    ? m.confusion_matrix.map((row) => {
        const total = row.reduce((a, b) => a + b, 0) || 1
        return row.map((v) => v / total)
      })
    : m.confusion_matrix

  const lowest = Math.min(...summary.map((r) => r.cv_f1_macro_mean))

  return (
    <>
      <div className="card">
        <h2>Model comparison</h2>
        <p>Cross-validation uses only the training split; the test score comes from data no model saw during selection. Models are sorted by cross-validated F1.</p>
        <Chart
          height={Math.max(300, summary.length * 70 + 100)}
          data={[
            { type: 'bar', orientation: 'h', name: 'CV F1 (mean ± std)', y: summary.map((r) => r.label), x: summary.map((r) => r.cv_f1_macro_mean),
              error_x: { type: 'data', array: summary.map((r) => r.cv_f1_macro_std) }, marker: { color: '#3b5bdb' } },
            { type: 'bar', orientation: 'h', name: 'Test F1', y: summary.map((r) => r.label), x: summary.map((r) => r.test_f1_macro),
              marker: { color: '#e8590c' } },
          ]}
          layout={{ barmode: 'group', yaxis: { autorange: 'reversed' }, xaxis: { range: [Math.max(0, lowest - 0.15), 1.02], title: { text: 'macro F1' } },
                    legend: { orientation: 'h', y: 1.12 }, margin: { l: 200, r: 20, t: 40, b: 50 } }}
        />
        <DataTable
          columns={[
            { key: 'label', label: 'Pipeline' }, { key: 'cv_accuracy_mean', label: 'CV accuracy' }, { key: 'cv_f1_macro_mean', label: 'CV F1' },
            { key: 'cv_f1_macro_std', label: 'CV F1 std' }, { key: 'test_accuracy', label: 'Test accuracy' }, { key: 'test_f1_macro', label: 'Test F1' },
            { key: 'test_roc_auc', label: 'Test ROC AUC' }, { key: 'fit_seconds', label: 'Fit time (s)' },
          ]}
          rows={summary}
        />
      </div>

      <div className="card">
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h2>Inspect a pipeline</h2>
          <label className="field">
            Pipeline
            <select value={selected} onChange={(e) => setSelected(e.target.value)}>
              {summary.map((r) => <option key={r.label} value={r.label}>{r.label}</option>)}
            </select>
          </label>
        </div>

        <div className="grid">
          <div className="stat"><div className="label">Accuracy</div><div className="value">{m.accuracy.toFixed(3)}</div></div>
          <div className="stat"><div className="label">Balanced accuracy</div><div className="value">{m.balanced_accuracy.toFixed(3)}</div></div>
          <div className="stat"><div className="label">Macro F1</div><div className="value">{m.f1_macro.toFixed(3)}</div></div>
          <div className="stat"><div className="label">MCC</div><div className="value">{m.mcc.toFixed(3)}</div></div>
          <div className="stat"><div className="label">ROC AUC</div><div className="value">{m.roc_auc === null ? '—' : m.roc_auc.toFixed(3)}</div></div>
        </div>

        <div className="row" style={{ justifyContent: 'space-between', marginTop: 18 }}>
          <h3>Confusion matrix (test set)</h3>
          <label className="row muted">
            <input type="checkbox" checked={normalize} onChange={(e) => setNormalize(e.target.checked)} /> show as fractions of each true class
          </label>
        </div>
        <Chart
          height={Math.max(320, names.length * 45 + 140)}
          data={[{ type: 'heatmap', z: cm, x: names, y: names, colorscale: 'Blues', showscale: false,
                   texttemplate: normalize ? '%{z:.2f}' : '%{z}', hovertemplate: 'true %{y}<br>predicted %{x}<br>%{z}<extra></extra>' }]}
          layout={{ xaxis: { title: { text: 'predicted' }, type: 'category' }, yaxis: { title: { text: 'true' }, autorange: 'reversed', type: 'category' } }}
        />

        <h3 style={{ marginTop: 10 }}>Per-class results</h3>
        <DataTable
          columns={[{ key: 'class', label: 'Class' }, { key: 'precision', label: 'Precision' }, { key: 'recall', label: 'Recall' },
                    { key: 'f1', label: 'F1' }, { key: 'support', label: 'Support' }]}
          rows={m.per_class}
        />
      </div>
    </>
  )
}
