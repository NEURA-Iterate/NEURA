<script lang="ts">
  import { onMount, untrack } from 'svelte'
  import { classify, classifyDemo, getDemoSamples, getJob } from './api'
  import type { Job, Overview } from './types'
  import Result from './Result.svelte'
  import FileDrop from './FileDrop.svelte'

  let {
    overview,
    jobId,
    onjob,
    onchange,
  }: {
    overview: Overview
    jobId: string | null
    onjob: (id: string | null) => void
    onchange: () => void
  } = $props()
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
    try { demo = await getDemoSamples() } catch { demo = [] }
  })
  $effect(() => {
    const id = jobId
    untrack(() => (id ? poll(id) : (job = null)))
    return stop
  })

  type Slot = 'bse' | 'etd' | 'inlens'
  const slots: { key: Slot; label: string; hint: string }[] = [
    { key: 'bse', label: 'BSE', hint: 'Backscatter' },
    { key: 'etd', label: 'ETD or SE', hint: 'Secondary electrons · cracks' },
    { key: 'inlens', label: 'Inlens', hint: 'In-lens detector' },
  ]
  let unmatched = $state<string[]>([])
  let nameEdited = false

  const getSlot = (k: Slot) => (k === 'bse' ? bse : k === 'etd' ? etd : inlens)
  function setSlot(k: Slot, f: File | null) {
    if (k === 'bse') bse = f
    else if (k === 'etd') etd = f
    else inlens = f
    if (k === 'bse' && f && !nameEdited) sampleId = f.name.replace(/_BSE\.tiff?$/i, '').replace(/\.tiff?$/i, '')
  }
  function detect(name: string): Slot | null {
    const n = name.toLowerCase()
    if (n.includes('_bse')) return 'bse'
    if (n.includes('_inlens')) return 'inlens'
    if (n.includes('_etd') || n.includes('_se')) return 'etd'
    return null
  }
  function onMulti(files: File[]) {
    const missed: string[] = []
    for (const f of files) {
      const k = detect(f.name)
      if (k) setSlot(k, f)
      else missed.push(f.name)
    }
    unmatched = missed
  }
  function clearAll() {
    bse = etd = inlens = null
    sampleId = ''
    nameEdited = false
    unmatched = []
  }
  const size = (f: File) => (f.size > 1e6 ? `${(f.size / 1e6).toFixed(1)} MB` : `${Math.ceil(f.size / 1e3)} kB`)
  const count = $derived([bse, etd, inlens].filter(Boolean).length)

  function stop() {
    if (timer) clearInterval(timer)
    timer = null
  }
  function poll(id: string) {
    stop()
    const tick = async () => {
      try {
        const next = await getJob(id)
        if (id !== jobId) return
        job = next
        if (next.status === 'done' || next.status === 'error') { stop(); onchange() }
      } catch (e) { error = (e as Error).message; stop(); job = null; onjob(null) }
    }
    tick()
    timer = setInterval(tick, 1500)
  }
  async function run(start: () => Promise<{ job_id: string }>) {
    error = null
    job = { status: 'queued', step: 'uploading', progress: 0 }
    try {
      const { job_id } = await start()
      onjob(job_id)
      onchange()
    } catch (e) { error = (e as Error).message; job = null }
  }
  const when = (iso: string) =>
    new Date(iso).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
  const busy = $derived(job?.status === 'queued' || job?.status === 'running')
</script>

<div class="grid">
{#if !job}
  <section class="card">
    <h2 class="title">Classify a sample</h2>
    <FileDrop multiple onfiles={onMulti} disabled={busy}>
      <svg class="icon" viewBox="0 0 24 24" width="34" height="34" aria-hidden="true"><path d="M12 16V4m0 0-4.5 4.5M12 4l4.5 4.5M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <div class="drop-title">Drop the three TIFFs here</div>
      <div class="muted">or <span class="link">browse</span> · files are matched by <code>_BSE</code>, <code>_ETD</code>/<code>_SE</code>, <code>_Inlens</code></div>
    </FileDrop>
    {#if unmatched.length}
      <div class="warn" style="margin-top:10px">Couldn't tell the detector for {unmatched.join(', ')}. Drop it onto the matching slot below.</div>
    {/if}

    <div class="slots">
      {#each slots as s (s.key)}
        {@const f = getSlot(s.key)}
        <FileDrop compact filled={!!f} disabled={busy} onfiles={(fs) => { setSlot(s.key, fs[0]); unmatched = [] }}>
          <div class="slot">
            <div class="slot-head">
              <span class="slot-label">{s.label}</span>
              {#if f}
                <button class="remove" title="Remove" aria-label="Remove {s.label}" disabled={busy}
                  onclick={(e) => { e.stopPropagation(); setSlot(s.key, null) }}>×</button>
              {:else}
                <span class="req">required</span>
              {/if}
            </div>
            {#if f}
              <div class="fname" title={f.name}>{f.name}</div>
              <div class="muted small">{size(f)}</div>
            {:else}
              <div class="muted small">{s.hint}</div>
              <div class="muted small">Drop or click</div>
            {/if}
          </div>
        </FileDrop>
      {/each}
    </div>

    <div class="actions">
      <label class="name">
        <span>Sample name</span>
        <input type="text" bind:value={sampleId} oninput={() => (nameEdited = true)} placeholder="optional" disabled={busy} />
      </label>
      {#if count}<button class="ghost" onclick={clearAll} disabled={busy}>Clear</button>{/if}
      <button class="primary big" disabled={busy || !bse || !etd || !inlens}
        onclick={() => run(() => classify({ bse: bse!, etd: etd!, inlens: inlens! }, sampleId || undefined))}>
        {busy ? 'Classifying…' : `Classify${count < 3 ? ` (${count}/3)` : ''}`}
      </button>
    </div>
    {#if demo.length}
      <div class="demo">
        <span class="demo-label">Re-run</span>
        <select bind:value={demoId} disabled={busy}>
          <option value="">choose…</option>
          {#each demo as d}<option value={d.image_id}>{d.image_id} ({d.batch.replace('_', ' ')})</option>{/each}
        </select>
        <button disabled={busy || !demoId} onclick={() => run(() => classifyDemo(demoId))}>Run</button>
      </div>
    {/if}
    {#if error}<div class="warn" style="margin-top:10px">{error}</div>{/if}
  </section>
{:else}
  <div class="backbar">
    <button class="back" onclick={() => onjob(null)} aria-label="Back to input">
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M15 18l-6-6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      New sample
    </button>
    {#if job.created_at}<span class="muted saved">Saved run · {when(job.created_at)}</span>{/if}
  </div>
  {#if job.status === 'done' && job.result}
    <Result result={job.result} {overview} />
  {:else}
    <section class="card progress">
      {#if job.status === 'error'}
        <div class="warn">Failed: {job.error}</div>
      {:else}
        <h2>{job.status === 'queued' ? 'Queued' : 'Classifying'}…</h2>
        <div class="pbar"><div style="width:{Math.max(job.progress, 0.03) * 100}%"></div></div>
        <span class="muted">{job.step} · usually 1–3 min on CPU. You can leave this page; the run is saved to history.</span>
      {/if}
    </section>
  {/if}
{/if}
</div>

<style>
  .title { margin-bottom: 16px; }
  .icon { color: #1f5fbf; }
  .drop-title { font-size: 1.05rem; font-weight: 600; color: #1d1d1f; }
  .link { color: #1f5fbf; font-weight: 600; }
  code { font-size: 0.8rem; background: #eef0f3; padding: 1px 5px; border-radius: 6px; }
  .slots { margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
  .slot { display: grid; gap: 2px; min-width: 0; }
  .slot-head { display: flex; justify-content: space-between; align-items: center; min-height: 28px; }
  .slot-label { font-weight: 600; font-size: 0.92rem; color: #1d1d1f; }
  .req { font-size: 0.72rem; color: #8a8f98; background: #eef0f3; padding: 2px 8px; border-radius: 999px; }
  .remove { width: 28px; height: 28px; padding: 0; border-radius: 999px; font-size: 1.1rem; line-height: 1; color: #666; border-color: transparent; background: transparent; }
  .remove:hover { background: #e8ebef; }
  .fname { font-size: 0.85rem; color: #1d6b33; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .small { font-size: 0.78rem; }
  .actions { margin-top: 18px; display: flex; gap: 12px; align-items: end; flex-wrap: wrap; }
  .name { display: grid; gap: 6px; flex: 1 1 260px; font-size: 0.85rem; font-weight: 600; }
  .name input { font-weight: 400; }
  .big { min-width: 180px; }
  .demo-label { font-size: 0.85rem; font-weight: 600; }
  .demo { margin-top: 22px; padding-top: 18px; border-top: 1px solid #eef0f3; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .backbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
  .back { display: inline-flex; align-items: center; gap: 6px; padding-left: 12px; }
  .progress { display: grid; gap: 10px; }
  .saved { font-size: 0.82rem; }
  .pbar { height: 8px; background: #eef0f3; border-radius: 999px; overflow: hidden; }
  .pbar div { height: 100%; background: #1f5fbf; border-radius: 999px; transition: width 0.5s; }
</style>
