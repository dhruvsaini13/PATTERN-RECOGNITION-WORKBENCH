// Every call to the FastAPI backend goes through this file.
const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function formatError(detail) {
  if (!detail) return null
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) {
    // FastAPI's validation errors: [{ loc: [...], msg: '...' }, ...]
    return detail.map((d) => `${(d.loc || []).slice(1).join(' → ')}: ${d.msg}`).join('\n')
  }
  return JSON.stringify(detail)
}

async function request(path, options = {}) {
  let response
  try {
    response = await fetch(BASE_URL + path, options)
  } catch {
    throw new Error(`Cannot reach the backend at ${BASE_URL}. Is uvicorn running?`)
  }
  if (response.status === 204) return null
  let body = null
  try {
    body = await response.json()
  } catch {
    /* response had no JSON body */
  }
  if (!response.ok) {
    throw new Error(formatError(body?.detail) || `Request failed (HTTP ${response.status})`)
  }
  return body
}

const postJson = (path, payload) =>
  request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

export const api = {
  health: () => request('/health'),
  modules: () => request('/modules'),

  // datasets
  builtinNames: () => request('/datasets/builtin'),
  loadBuiltin: (name) => request(`/datasets/builtin/${name}`, { method: 'POST' }),
  uploadCsv: (formData) => request('/datasets/upload', { method: 'POST', body: formData }), // browser sets the multipart header
  preview: (id, rows = 10) => request(`/datasets/${id}/preview?rows=${rows}`),
  analysis: (id, corrThreshold = 0.9) => request(`/datasets/${id}/analysis?corr_threshold=${corrThreshold}`),
  datasetPlot: (id, kind) => request(`/datasets/${id}/plots/${kind}`),

  // classification
  compare: (payload) => postJson('/classification/compare', payload),
  train: (payload) => postJson('/classification/train', payload),
  predict: (modelId, rows) => postJson(`/models/${modelId}/predict`, { rows }),
  modelPlot: (modelId, kind) => request(`/models/${modelId}/plots/${kind}`),

  // clustering
  clusterRun: (payload) => postJson('/clustering/run', payload),
  clusterCompare: (payload) => postJson('/clustering/compare', payload),
  clusterSweep: (payload) => postJson('/clustering/sweep', payload),

  // background jobs
  submitCompareJob: (payload) => postJson('/jobs/classification/compare', payload),
  job: (jobId) => request(`/jobs/${jobId}`),
}
