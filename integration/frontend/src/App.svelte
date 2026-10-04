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
  header { display: flex; align-items: center; justify-content: space-between; padding: 10px 24px; background: #14213d; color: #fff; }
  .brand { font-size: 1.05rem; }
  nav { display: flex; gap: 6px; }
  nav button { background: transparent; color: #cfd6e4; border: 1px solid transparent; }
  nav button.active { background: #fff; color: #14213d; }
  main { max-width: 1280px; margin: 0 auto; padding: 20px 24px 60px; }
</style>
