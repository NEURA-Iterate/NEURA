<script lang="ts">
  import { onMount } from 'svelte'
  import { getHistory } from './api'
  import type { HistoryItem } from './types'
  import { batchColor, pct, short } from './format'

  let { onOpen }: { onOpen: (jobId: string) => void } = $props()
  let items = $state<HistoryItem[] | null>(null)
  let error = $state<string | null>(null)
  let query = $state('')
  let source = $state<'all' | 'upload' | 'demo'>('all')

  async function load() {
    error = null
    try { items = await getHistory() } catch (e) { error = (e as Error).message }
  }
  onMount(load)

  const shown = $derived(
    (items ?? []).filter((it) =>
      (source === 'all' || it.source === source) &&
      it.sample_id.toLowerCase().includes(query.trim().toLowerCase()),
    ),
  )
  const when = (iso: string) => {
    const d = new Date(iso)
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
  }
  const batches = ['Batch_1', 'Batch_2', 'Batch_3']
</script>

<section class="card">
  <div class="head">
    <h2>Classification history</h2>
    <div class="tools">
      <input type="search" placeholder="Filter by sample name" bind:value={query} />
      <select bind:value={source}>
        <option value="all">All runs</option>
        <option value="upload">Uploaded samples</option>
        <option value="demo">Training-sample re-runs</option>
      </select>
      <button onclick={load}>Refresh</button>
      <a class="button" href="/api/history.csv" download>Download CSV</a>
    </div>
  </div>
  <p class="muted small">Every classification is saved on the server with its sample name, results and images. Open a row to see the full explanation again.</p>

  {#if error}
    <div class="warn">Could not load history: {error}</div>
  {:else if !items}
    <p class="muted">Loading…</p>
  {:else if !shown.length}
    <p class="muted">{items.length ? 'No runs match the filter.' : 'No classifications yet.'}</p>
  {:else}
    <table>
      <thead>
        <tr>
          <th>When</th><th>Sample</th><th>Type</th><th>Predicted</th><th class="probs">P(B1 / B2 / B3)</th>
          <th>Confidence</th><th>Flags</th><th></th>
        </tr>
      </thead>
      <tbody>
        {#each shown as it (it.job_id)}
          <tr>
            <td class="nowrap">{when(it.created_at)}</td>
            <td><b>{it.sample_id}</b></td>
            <td class="muted small">
              {it.source === 'demo' ? 'training re-run' : 'upload'}
              {#if it.known_batch}<br />label {short(it.known_batch)}{/if}
            </td>
            {#if it.status === 'done' && it.predicted && it.probabilities}
              <td>
                <b style="color:{batchColor(it.predicted)}">{it.predicted.replace('_', ' ')}</b>
                {#if it.known_batch}
                  <span class={it.known_batch === it.predicted ? 'ok' : 'bad'}>{it.known_batch === it.predicted ? '✓' : '✗'}</span>
                {/if}
              </td>
              <td class="probs">
                <div class="stack" title={batches.map((b) => `${short(b)} ${pct(it.probabilities![b])}`).join(' · ')}>
                  {#each batches as b}<div style="width:{it.probabilities[b] * 100}%; background:{batchColor(b)}"></div>{/each}
                </div>
                <div class="small muted">{batches.map((b) => pct(it.probabilities![b])).join(' / ')}</div>
              </td>
              <td><span class="pill tier-{it.tier}">{it.tier}</span></td>
              <td class="small">
                {#if it.outlier}<span class="flag">outlier</span>{/if}
                {#if it.ambiguous}<span class="flag">ambiguous</span>{/if}
                {#if it.trust_flags}<span class="flag">{it.trust_flags}</span>{/if}
              </td>
              <td><button onclick={() => onOpen(it.job_id)}>Open</button></td>
            {:else if it.status === 'error'}
              <td colspan="5" class="small bad">Failed: {it.error}</td>
              <td></td>
            {:else}
              <td colspan="5" class="muted small">{it.status}…</td>
              <td><button onclick={() => onOpen(it.job_id)}>Follow</button></td>
            {/if}
          </tr>
        {/each}
      </tbody>
    </table>
    <p class="muted small">{shown.length} of {items.length} runs. ✓/✗ only for training-sample re-runs (known label; those samples were in the training set).</p>
  {/if}
</section>

<style>
  .head { display: flex; justify-content: space-between; align-items: center; gap: 12px; flex-wrap: wrap; }
  .tools { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .tools input { min-width: 200px; }
  a.button { display: inline-block; padding: 6px 12px; border: 1px solid #c8ccd4; border-radius: 6px; background: #fff; color: inherit; text-decoration: none; font-size: 0.9rem; }
  .probs { min-width: 180px; }
  .stack { display: flex; height: 12px; border-radius: 3px; overflow: hidden; background: #eee; }
  .nowrap { white-space: nowrap; }
  .ok { color: #1d6b33; font-weight: 700; }
  .bad { color: #b42318; font-weight: 600; }
  .flag { display: inline-block; margin: 0 4px 2px 0; padding: 0 6px; border-radius: 4px; background: #fde2e1; color: #9b1c1c; }
</style>
