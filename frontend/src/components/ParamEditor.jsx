import { CHOICES } from '../params'

// Draws one input per parameter, based on the module's default values.
export default function ParamEditor({ defaults, values, onChange, scoreFuncs = [] }) {
  const keys = Object.keys(defaults)
  if (keys.length === 0) return <span className="muted">no parameters</span>

  return (
    <div className="row">
      {keys.map((key) => {
        const def = defaults[key]
        const value = values[key] ?? ''
        let input
        if (typeof def === 'boolean') {
          input = <input type="checkbox" checked={Boolean(value)} onChange={(e) => onChange(key, e.target.checked)} />
        } else if (key === 'score_func' || CHOICES[key]) {
          const options = key === 'score_func' ? scoreFuncs : CHOICES[key]
          input = (
            <select value={value} onChange={(e) => onChange(key, e.target.value)}>
              {options.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          )
        } else if (typeof def === 'number') {
          input = <input type="number" step={Number.isInteger(def) ? 1 : 'any'} value={value} onChange={(e) => onChange(key, e.target.value)} />
        } else {
          input = <input type="text" placeholder={def === null ? 'none' : ''} value={value} onChange={(e) => onChange(key, e.target.value)} />
        }
        return (
          <label key={key} className="field small">
            {key}
            {input}
          </label>
        )
      })}
    </div>
  )
}
