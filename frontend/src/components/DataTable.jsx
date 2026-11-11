function formatValue(v) {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return <span className={v ? 'badge' : 'badge no'}>{v ? 'yes' : 'no'}</span>
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return v
    if (v !== 0 && Math.abs(v) < 0.001) return v.toExponential(2)
    return Number(v.toFixed(4))
  }
  return String(v)
}

// columns: [{ key: 'mean', label: 'Mean' }, ...]   rows: array of objects
export default function DataTable({ columns, rows, maxHeight = 360 }) {
  if (!rows || rows.length === 0) return <p className="muted">Nothing to show.</p>
  return (
    <div className="table-wrap" style={{ maxHeight }}>
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key}>{c.label ?? c.key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map((c) => (
                <td key={c.key}>{formatValue(row[c.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
