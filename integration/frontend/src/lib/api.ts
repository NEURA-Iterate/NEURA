import type { Job, Overview } from './types'

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`
    try { const b = await r.json(); msg = b.detail ?? b.error ?? msg } catch { /* not JSON */ }
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg))
  }
  return r.json() as Promise<T>
}

export const getOverview = () => fetch('/api/overview').then((r) => json<Overview>(r))
export const getDemoSamples = () =>
  fetch('/api/demo-samples').then((r) => json<{ image_id: string; batch: string }[]>(r))
export const getJob = (id: string) => fetch(`/api/jobs/${id}`).then((r) => json<Job>(r))

export function classify(files: { bse: File; etd: File; inlens: File }, sampleId?: string) {
  const fd = new FormData()
  fd.append('bse', files.bse)
  fd.append('etd', files.etd)
  fd.append('inlens', files.inlens)
  if (sampleId) fd.append('sample_id', sampleId)
  return fetch('/api/classify', { method: 'POST', body: fd }).then((r) => json<{ job_id: string }>(r))
}

export const classifyDemo = (image_id: string) =>
  fetch('/api/classify-demo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_id }),
  }).then((r) => json<{ job_id: string }>(r))
