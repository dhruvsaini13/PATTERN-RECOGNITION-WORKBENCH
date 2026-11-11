import Chart from './Chart'

const PALETTE = ['#3b5bdb', '#e8590c', '#2b8a3e', '#ae3ec9', '#1098ad', '#f08c00', '#c2255c', '#5c940d', '#6741d9', '#e03131']

// One scatter trace per group (cluster or true class). Noise (-1) is drawn as grey crosses.
function groupTraces(points, keyOf, nameOf) {
  const groups = new Map()
  for (const p of points) {
    const k = keyOf(p)
    if (!groups.has(k)) groups.set(k, [])
    groups.get(k).push(p)
  }
  const keys = [...groups.keys()].sort((a, b) => (a === -1 ? -1 : b === -1 ? 1 : a < b ? -1 : a > b ? 1 : 0))
  let colour = 0
  return keys.map((k) => {
    const pts = groups.get(k)
    const noise = k === -1
    return {
      type: 'scatter', mode: 'markers', name: nameOf(k),
      x: pts.map((p) => p.x), y: pts.map((p) => p.y),
      marker: noise ? { color: '#adb5bd', symbol: 'x', size: 7 } : { color: PALETTE[colour++ % PALETTE.length], size: 7, opacity: 0.8 },
    }
  })
}

const fmt = (v) => (v === null || v === undefined ? '—' : Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3))

export default function ClusterView({ view, totalSamples }) {
  const m = view.metrics
  const hasTruth = view.points.length > 0 && view.points[0].truth !== null
  const sizeKeys = Object.keys(view.cluster_sizes)
  const layout2d = { xaxis: { title: { text: 'dimension 1' } }, yaxis: { title: { text: 'dimension 2' } }, legend: { orientation: 'h', y: -0.25 } }

  return (
    <>
      <div className="grid">
        <div className="stat"><div className="label">Clusters found</div><div className="value">{m.n_clusters}</div></div>
        <div className="stat"><div className="label">Noise points</div><div className="value">{m.n_noise}</div></div>
        <div className="stat"><div className="label">Silhouette ↑</div><div className="value">{fmt(m.silhouette)}</div></div>
        <div className="stat"><div className="label">Davies-Bouldin ↓</div><div className="value">{fmt(m.davies_bouldin)}</div></div>
        <div className="stat"><div className="label">Calinski-Harabasz ↑</div><div className="value">{fmt(m.calinski_harabasz)}</div></div>
        {'adjusted_rand' in m && (
          <>
            <div className="stat"><div className="label">Adjusted Rand ↑</div><div className="value">{fmt(m.adjusted_rand)}</div></div>
            <div className="stat"><div className="label">NMI ↑</div><div className="value">{fmt(m.nmi)}</div></div>
            <div className="stat"><div className="label">V-measure ↑</div><div className="value">{fmt(m.v_measure)}</div></div>
          </>
        )}
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        Silhouette, Davies-Bouldin and Calinski-Harabasz judge the clusters by their shape alone. Adjusted Rand, NMI and V-measure compare them with the dataset’s real classes (1 = perfect match, about 0 = no better than chance).
      </p>

      <div className={hasTruth ? 'charts' : ''}>
        <Chart
          height={380}
          data={groupTraces(view.points, (p) => p.label, (k) => (k === -1 ? 'noise' : `cluster ${k}`))}
          layout={{ ...layout2d, title: { text: 'Clusters found' } }}
        />
        {hasTruth && (
          <Chart
            height={380}
            data={groupTraces(view.points, (p) => p.truth, (k) => String(k))}
            layout={{ ...layout2d, title: { text: 'Real classes (for comparison)' } }}
          />
        )}
      </div>
      {view.n_points_shown < totalSamples && <p className="muted">Showing a random sample of {view.n_points_shown} of {totalSamples} points.</p>}
      <p className="muted">When there are more than 2 features, the picture is a 2D PCA projection of the features that were clustered.</p>

      <Chart
        height={260}
        data={[{ type: 'bar', x: sizeKeys.map((k) => (k === '-1' ? 'noise' : `cluster ${k}`)), y: sizeKeys.map((k) => view.cluster_sizes[k]), marker: { color: '#3b5bdb' } }]}
        layout={{ title: { text: 'Cluster sizes' }, yaxis: { title: { text: 'samples' } } }}
      />
    </>
  )
}
