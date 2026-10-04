<script lang="ts">
  import { getHistory } from './api'
  import type { HistoryItem } from './types'
  import { batchColor } from './format'

  type View = 'classify' | 'history' | 'overview'
  let {
    open,
    view,
    jobId,
    version,
    onnavigate,
    onopen,
    ontoggle,
  }: {
    open: boolean
    view: View
    jobId: string | null
    version: number
    onnavigate: (v: View) => void
    onopen: (id: string) => void
    ontoggle: () => void
  } = $props()

  let items = $state<HistoryItem[] | null>(null)
  let error = $state(false)

  async function load() {
    try { items = await getHistory(); error = false } catch { error = true }
  }
  $effect(() => {
    version
    load()
  })
  $effect(() => {
    if (!items?.some((it) => it.status === 'queued' || it.status === 'running')) return
    const t = setInterval(load, 4000)
    return () => clearInterval(t)
  })

  const when = (iso: string) => {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso
    const today = new Date().toDateString() === d.toDateString()
    return today
      ? d.toLocaleTimeString(undefined, { timeStyle: 'short' })
      : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
  }
</script>

<aside class:open aria-hidden={!open}>
  <div class="top">
    <div class="brand"><strong>NEURA</strong> <span>SEM classifier</span></div>
    <button class="icon" onclick={ontoggle} aria-label="Collapse menu" title="Collapse menu" tabindex={open ? 0 : -1}>
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M15 18l-6-6 6-6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
    </button>
  </div>

  <nav>
    <button class="item" class:active={view === 'classify' && !jobId} onclick={() => onnavigate('classify')} tabindex={open ? 0 : -1}>
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      New sample
    </button>
    <button class="item" class:active={view === 'overview'} onclick={() => onnavigate('overview')} tabindex={open ? 0 : -1}>
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      Classifier overview
    </button>
  </nav>

  <div class="section">
    <span>History</span>
    <button class="link" class:active={view === 'history'} onclick={() => onnavigate('history')} tabindex={open ? 0 : -1}>View all</button>
  </div>
  <ul class="history">
    {#if error}
      <li class="muted small pad">Couldn't load history.</li>
    {:else if !items}
      <li class="muted small pad">Loading…</li>
    {:else if !items.length}
      <li class="muted small pad">No runs yet.</li>
    {:else}
      {#each items as it (it.job_id)}
        <li>
          <button class="run" class:active={jobId === it.job_id && view === 'classify'} onclick={() => onopen(it.job_id)} tabindex={open ? 0 : -1}>
            <span class="dot" style="background:{it.status === 'done' && it.predicted ? batchColor(it.predicted) : it.status === 'error' ? '#d92d20' : '#c4c8cf'}"></span>
            <span class="name" title={it.sample_id}>{it.sample_id}</span>
            <span class="meta">
              {#if it.status === 'done' && it.predicted}{it.predicted.replace('Batch_', 'B')}
              {:else if it.status === 'error'}failed
              {:else}running…{/if}
              · {when(it.created_at)}
            </span>
          </button>
        </li>
      {/each}
    {/if}
  </ul>
</aside>

<style>
  aside {
    position: sticky; top: 0; height: 100vh; flex: 0 0 auto;
    width: 0; overflow: hidden; background: #fff; border-right: 1px solid transparent;
    display: flex; flex-direction: column; transition: width 0.2s ease, border-color 0.2s;
  }
  aside.open { width: 280px; border-right-color: #eceef1; }
  aside > * { min-width: 280px; }
  .top { display: flex; align-items: center; justify-content: space-between; padding: 16px 14px 12px 20px; }
  .brand { font-size: 1.05rem; white-space: nowrap; }
  .brand span { color: #666; font-size: 0.9rem; margin-left: 4px; }
  .icon { min-height: 36px; width: 36px; padding: 0; display: grid; place-items: center; border-color: transparent; border-radius: 10px; color: #555; }
  nav { display: grid; gap: 2px; padding: 4px 12px 12px; }
  .item, .run { display: flex; align-items: center; gap: 10px; width: 100%; text-align: left; border: 0; background: transparent; border-radius: 12px; color: #333; }
  .item { min-height: 44px; padding: 10px 12px; }
  .item:hover:not(:disabled), .run:hover:not(:disabled) { background: #f3f5f8; }
  .item.active, .run.active { background: #eef3fc; color: #1f5fbf; }
  .section { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px 6px 24px; border-top: 1px solid #f0f1f4; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: #8a8f98; }
  .link { min-height: 28px; padding: 2px 8px; border: 0; background: transparent; color: #1f5fbf; font-size: 0.78rem; text-transform: none; letter-spacing: 0; border-radius: 8px; }
  .link.active { background: #eef3fc; }
  .history { list-style: none; margin: 0; padding: 0 12px 20px; overflow-y: auto; flex: 1; }
  .run { display: grid; grid-template-columns: 10px 1fr; column-gap: 10px; row-gap: 1px; min-height: 0; padding: 8px 12px; }
  .dot { width: 9px; height: 9px; border-radius: 999px; grid-row: span 2; align-self: center; }
  .name { font-size: 0.88rem; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .meta { font-size: 0.75rem; color: #8a8f98; }
  .small { font-size: 0.8rem; }
  .pad { padding: 8px 12px; }
  @media (max-width: 900px) {
    aside { position: fixed; left: 0; z-index: 30; box-shadow: none; }
    aside.open { box-shadow: 0 10px 40px rgba(16, 24, 40, 0.18); }
  }
</style>
