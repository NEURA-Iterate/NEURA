<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { classify, classifyDemo, getDemoSamples, getJob } from './api'
  import type { Job, Overview } from './types'
  import Result from './Result.svelte'

  let { overview }: { overview: Overview } = $props()
  let bse = $state<File | null>(null)
  let etd = $state<File | null>(null)
  let inlens = $state<File | null>(null)
  let sampleId = $state('')
  let demo = $state<{ image_id: string; batch: string }[]>([])
  let demoId = $state('')
  let job = $state<Job | null>(null)
  let error = $state<string | null>(null)
  let timer: ReturnType<typeof setInterval> | null = null

  onMount(async () => {
    const id = new URLSearchParams(location.search).get('job')
    if (id) poll(id)
    try { demo = await getDemoSamples() } catch { demo = [] }
  })
  onDestroy(() => timer && clearInterval(timer))

  const pick = (e: Event) => (e.currentTarget as HTMLInputElement).files?.[0] ?? null

  function onMulti(e: Event) {
    for (const f of Array.from((e.currentTarget as HTMLInputElement).files ?? [])) {
      const n = f.name.toLowerCase()
      if (n.includes('_bse')) bse = f
      else if (n.includes('_inlens')) inlens = f
      else if (n.includes('_etd') || n.includes('_se')) etd = f
    }
    const name = bse?.name.replace(/_BSE\.tiff?$/i, '')
    if (name && !sampleId) sampleId = name
  }

  function poll(id: string) {
    timer && clearInterval(timer)
    timer = setInterval(async () => {
      try {
        job = await getJob(id)
        const url = new URL(location.href)
        if (url.searchParams.get('job') !== id) { url.searchParams.set('job', id); history.replaceState(null, '', url) }
        if (job.status === 'done' || job.status === 'error') { clearInterval(timer!); timer = null }
      } catch (e) { error = (e as Error).message; clearInterval(timer!); timer = null }
    }, 1500)
  }
  async function run(start: () => Promise<{ job_id: string }>) {
    error = null
    job = { status: 'queued', step: 'uploading', progress: 0 }
    try { poll((await start()).job_id) } catch (e) { error = (e as Error).message; job = null }
  }
  const busy = $derived(job?.status === 'queued' || job?.status === 'running')
</script>

<div class="grid">
  <section class="card">
    <h2>Classify a sample</h2>
    <p class="muted">Upload the three detector TIFFs of one location. They must be co-registered (same size) and at the training magnification.</p>
    <div class="upload">
      <label class="multi">Select all three at once <input type="file" multiple accept=".tif,.tiff" onchange={onMulti} disabled={busy} /></label>
      <label>BSE <input type="file" accept=".tif,.tiff" onchange={(e) => (bse = pick(e))} disabled={busy} /><span>{bse?.name ?? 'required'}</span></label>
      <label>ETD or SE <input type="file" accept=".tif,.tiff" onchange={(e) => (etd = pick(e))} disabled={busy} /><span>{etd?.name ?? 'required (cracks)'}</span></label>
      <label>Inlens <input type="file" accept=".tif,.tiff" onchange={(e) => (inlens = pick(e))} disabled={busy} /><span>{inlens?.name ?? 'required'}</span></label>
      <label>Sample name <input type="text" bind:value={sampleId} placeholder="optional" disabled={busy} /></label>
      <button class="primary" disabled={busy || !bse || !etd || !inlens}
        onclick={() => run(() => classify({ bse: bse!, etd: etd!, inlens: inlens! }, sampleId || undefined))}>Classify</button>
    </div>
    {#if demo.length}
      <div class="demo">
        <span class="muted">Or re-run a labelled training sample (sanity check; it was in the training set):</span>
        <select bind:value={demoId} disabled={busy}>
          <option value="">choose…</option>
          {#each demo as d}<option value={d.image_id}>{d.image_id} ({d.batch.replace('_', ' ')})</option>{/each}
        </select>
        <button disabled={busy || !demoId} onclick={() => run(() => classifyDemo(demoId))}>Run</button>
      </div>
    {/if}
    {#if error}<div class="warn" style="margin-top:10px">{error}</div>{/if}
    {#if job && job.status !== 'done'}
      <div class="progress">
        {#if job.status === 'error'}
          <div class="warn">Failed: {job.error}</div>
        {:else}
          <div class="pbar"><div style="width:{Math.max(job.progress, 0.03) * 100}%"></div></div>
          <span class="muted">{job.status === 'queued' ? 'Queued' : job.step}… (≈1–3 min on CPU)</span>
        {/if}
      </div>
    {/if}
  </section>

  {#if job?.status === 'done' && job.result}
    {#if job.created_at}
      <p class="muted saved">Saved run from {new Date(job.created_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}. Find all past runs in the History tab.</p>
    {/if}
    <Result result={job.result} {overview} />
  {/if}
</div>

<style>
  .upload { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px 16px; align-items: end; }
  .upload label { display: grid; gap: 3px; font-size: 0.85rem; font-weight: 600; }
  .upload label span { font-weight: 400; color: #666; font-size: 0.78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .upload .multi { grid-column: 1 / -1; font-weight: 400; }
  .demo { margin-top: 14px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .progress { margin-top: 14px; display: grid; gap: 4px; }
  .saved { margin: -4px 0 0; font-size: 0.82rem; }
  .pbar { height: 8px; background: #eee; border-radius: 4px; overflow: hidden; }
  .pbar div { height: 100%; background: #1f5fbf; transition: width 0.5s; }
</style>
