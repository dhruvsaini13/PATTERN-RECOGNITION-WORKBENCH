import { adaptDefaults, findModule, initialValues, newId } from '../params'
import ParamEditor from './ParamEditor'

// One pipeline = ordered preprocessing steps + one classifier.
// modelKey / modelOptions / modelTitle let the same card serve classification ('classifier') and clustering ('clusterer').
export default function PipelineCard({
  pipeline, index, catalog, dataset, onChange, onRemove, canRemove,
  modelKey = 'classifier', modelOptions = catalog.classifiers, modelTitle = 'Classifier',
}) {
  const setStepValue = (stepId, key, value) =>
    onChange({
      ...pipeline,
      steps: pipeline.steps.map((s) => (s.id === stepId ? { ...s, values: { ...s.values, [key]: value } } : s)),
    })

  function addStep(name) {
    if (!name) return
    const defaults = adaptDefaults(name, findModule(catalog.feature_extractors, name).params, dataset)
    onChange({ ...pipeline, steps: [...pipeline.steps, { id: newId(), name, values: initialValues(defaults) }] })
  }

  function moveStep(i, direction) {
    const steps = [...pipeline.steps]
    const j = i + direction
    if (j < 0 || j >= steps.length) return
    ;[steps[i], steps[j]] = [steps[j], steps[i]]
    onChange({ ...pipeline, steps })
  }

  function setModel(name) {
    const defaults = adaptDefaults(name, findModule(modelOptions, name).params, dataset)
    onChange({ ...pipeline, [modelKey]: { name, values: initialValues(defaults) } })
  }

  const model = pipeline[modelKey]
  const modelDefaults = findModule(modelOptions, model.name).params

  return (
    <div className="pipeline">
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <h3>Pipeline {index + 1}</h3>
        {canRemove && <button className="btn secondary" onClick={onRemove}>Remove</button>}
      </div>

      <label className="field" style={{ maxWidth: 320, marginBottom: 12 }}>
        Name (optional)
        <input type="text" value={pipeline.label} placeholder="auto-named from its steps" onChange={(e) => onChange({ ...pipeline, label: e.target.value })} />
      </label>

      <div className="muted" style={{ marginBottom: 6 }}>Preprocessing steps (run top to bottom)</div>
      {pipeline.steps.length === 0 && <p className="muted">No preprocessing: the raw features go straight to the model.</p>}
      {pipeline.steps.map((step, i) => (
        <div key={step.id} className="step">
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <strong>{i + 1}. {step.name}</strong>
            <div className="row">
              <button className="btn secondary" disabled={i === 0} onClick={() => moveStep(i, -1)} title="Move up">↑</button>
              <button className="btn secondary" disabled={i === pipeline.steps.length - 1} onClick={() => moveStep(i, 1)} title="Move down">↓</button>
              <button className="btn secondary" onClick={() => onChange({ ...pipeline, steps: pipeline.steps.filter((s) => s.id !== step.id) })}>Remove</button>
            </div>
          </div>
          <ParamEditor
            defaults={findModule(catalog.feature_extractors, step.name).params}
            values={step.values}
            scoreFuncs={catalog.score_funcs}
            onChange={(key, value) => setStepValue(step.id, key, value)}
          />
        </div>
      ))}
      <select value="" onChange={(e) => addStep(e.target.value)} style={{ marginTop: 6 }}>
        <option value="">+ Add a preprocessing step…</option>
        {catalog.feature_extractors.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
      </select>

      <div className="muted" style={{ margin: '16px 0 6px' }}>{modelTitle}</div>
      <div className="step">
        <label className="field" style={{ maxWidth: 260, marginBottom: 10 }}>
          Algorithm
          <select value={model.name} onChange={(e) => setModel(e.target.value)}>
            {modelOptions.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
          </select>
        </label>
        <ParamEditor
          defaults={modelDefaults}
          values={model.values}
          onChange={(key, value) =>
            onChange({ ...pipeline, [modelKey]: { ...model, values: { ...model.values, [key]: value } } })
          }
        />
      </div>
    </div>
  )
}
