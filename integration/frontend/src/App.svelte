<script lang="ts">
  import { onMount } from 'svelte'
  import { getOverview } from './lib/api'
  import type { Overview as OverviewT } from './lib/types'
  import Overview from './lib/Overview.svelte'
  import Classify from './lib/Classify.svelte'
  import History from './lib/History.svelte'

  let tab = $state<'classify' | 'history' | 'overview'>('classify')
  let overview = $state<OverviewT | null>(null)
  let error = $state<string | null>(null)

  onMount(async () => {
    try { overview = await getOverview() } catch (e) { error = (e as Error).message }
  })
</script>

<header>
  <div class="brand">
    <strong>NEURA</strong> · SEM batch classifier
  </div>
  <nav>
    <button class:active={tab === 'classify'} onclick={() => (tab = 'classify')}>Classify a sample</button>
    <button class:active={tab === 'history'} onclick={() => (tab = 'history')}>History</button>
    <button class:active={tab === 'overview'} onclick={() => (tab = 'overview')}>Classifier overview</button>
  </nav>
</header>

<main>
  {#if error}
    <div class="card warn">Could not reach the classifier API: {error}</div>
  {:else if !overview}
    <p class="muted">Loading classifier…</p>
  {:else if tab === 'classify'}
    <Classify {overview} />
  {:else if tab === 'history'}
    <History onOpen={(id) => {
      const url = new URL(location.href); url.searchParams.set('job', id); history.replaceState(null, '', url)
      tab = 'classify'; window.scrollTo(0, 0)
    }} />
  {:else}
    <Overview {overview} />
  {/if}
</main>

<style>
  header { position: sticky; top: 0; z-index: 10; display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap; padding: 12px 24px; background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(8px); border-bottom: 1px solid #eceef1; color: #1d1d1f; }
  .brand { font-size: 1.05rem; }
  nav { display: flex; gap: 4px; padding: 4px; background: #f1f3f6; border-radius: 14px; }
  nav button { background: transparent; color: #555; border: 1px solid transparent; border-radius: 10px; min-height: 40px; padding: 8px 16px; }
  nav button:hover:not(:disabled) { background: #e6e9ee; }
  nav button.active { background: #fff; color: #1d1d1f; box-shadow: 0 1px 3px rgba(16, 24, 40, 0.1); }
  main { max-width: 1280px; margin: 0 auto; padding: 20px 24px 60px; }
</style>
