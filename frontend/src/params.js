// Helpers for turning a module's default parameters into form values and back into API parameters.

let counter = 0
export const newId = () => ++counter

// Parameters with a fixed set of choices get a dropdown.
export const CHOICES = {
  kernel: ['rbf', 'linear', 'poly', 'sigmoid'],
  linkage: ['ward', 'complete', 'average', 'single'],
}

export function findModule(list, name) {
  return list.find((m) => m.name === name)
}

// Keep default parameters valid for the loaded dataset (e.g. LDA can't ask for more components than classes - 1).
export function adaptDefaults(name, defaults, dataset) {
  const out = { ...defaults }
  const nFeatures = dataset?.n_features ?? 1
  const nClasses = dataset?.n_classes ?? 2
  if (name === 'pca' && 'n_components' in out) out.n_components = Math.max(1, Math.min(out.n_components, nFeatures))
  if (name === 'lda' && 'n_components' in out) out.n_components = Math.max(1, Math.min(out.n_components, nClasses - 1, nFeatures))
  if (name === 'select_k_best' && 'k' in out) out.k = Math.max(1, Math.min(out.k, nFeatures))
  // Clustering: start from the number of real classes when the dataset has labels.
  const guessK = dataset?.has_labels ? nClasses : 3
  const maxK = Math.max(2, (dataset?.n_samples ?? 100) - 1)
  if ((name === 'kmeans' || name === 'hierarchical') && 'n_clusters' in out) out.n_clusters = Math.max(2, Math.min(guessK, maxK))
  if (name === 'gmm' && 'n_components' in out) out.n_components = Math.max(2, Math.min(guessK, maxK))
  return out
}

export function initialValues(defaults) {
  const values = {}
  for (const [key, def] of Object.entries(defaults)) values[key] = def === null ? '' : def
  return values
}

// Form values -> parameters for the API. Blank fields are left out so the backend default is used.
export function cleanParams(defaults, values) {
  const out = {}
  for (const [key, def] of Object.entries(defaults)) {
    const v = values[key]
    if (v === undefined || v === '') continue
    if (typeof def === 'number') out[key] = Number(v)
    else if (typeof def === 'boolean') out[key] = Boolean(v)
    else if (def === null) out[key] = /^-?\d+(\.\d+)?$/.test(String(v).trim()) ? Number(v) : String(v) // e.g. max_depth
    else out[key] = v
  }
  return out
}
