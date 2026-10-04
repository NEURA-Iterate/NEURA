<script lang="ts">
  import { onMount } from 'svelte'
  import { getOverview } from './lib/api'
  import type { Overview as OverviewT } from './lib/types'
  import Overview from './lib/Overview.svelte'
  import Classify from './lib/Classify.svelte'
  import History from './lib/History.svelte'
  import Sidebar from './lib/Sidebar.svelte'

  type View = 'classify' | 'history' | 'overview'
  let view = $state<View>('classify')
  let jobId = $state<string | null>(new URLSearchParams(location.search).get('job'))
  let overview = $state<OverviewT | null>(null)
  let error = $state<string | null>(null)
  let historyVersion = $state(0)
  const narrow = () => window.matchMedia('(max-width: 900px)').matches
  let menuOpen = $state(!narrow() && localStorage.getItem('neura.menu') !== 'closed')

  onMount(async () => {
    try { overview = await getOverview() } catch (e) { error = (e as Error).message }
  })

  $effect(() => {
    const url = new URL(location.href)
    if (jobId) url.searchParams.set('job', jobId)
    else url.searchParams.delete('job')
    if (url.href !== location.href) history.replaceState(null, '', url)
  })

  function toggleMenu() {
    menuOpen = !menuOpen
    if (!narrow()) localStorage.setItem('neura.menu', menuOpen ? 'open' : 'closed')
  }
  function navigate(v: View) {
    view = v
    if (v === 'classify') jobId = null
    if (narrow()) menuOpen = false
    window.scrollTo(0, 0)
  }
  function openJob(id: string) {
    jobId = id
    view = 'classify'
    if (narrow()) menuOpen = false
    window.scrollTo(0, 0)
  }
</script>

<div class="layout">
  <Sidebar open={menuOpen} {view} {jobId} version={historyVersion}
    onnavigate={navigate} onopen={openJob} ontoggle={toggleMenu} />
  {#if menuOpen}<button class="scrim" aria-label="Close menu" onclick={toggleMenu}></button>{/if}

  <div class="content">
    {#if !menuOpen}
      <button class="menu" onclick={toggleMenu} aria-label="Open menu" title="Open menu">
        <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>
    {/if}
    <main>
      {#if error}
        <div class="card warn">Could not reach the classifier API: {error}</div>
      {:else if !overview}
        <p class="muted">Loading classifier…</p>
      {:else if view === 'classify'}
        <Classify {overview} {jobId} onjob={(id) => (jobId = id)} onchange={() => historyVersion++} />
      {:else if view === 'history'}
        <History onOpen={openJob} />
      {:else}
        <Overview {overview} />
      {/if}
    </main>
  </div>
</div>

<style>
  .layout { display: flex; min-height: 100vh; }
  .content { flex: 1; min-width: 0; position: relative; }
  .menu { position: fixed; top: 16px; left: 16px; z-index: 20; width: 44px; padding: 0; display: grid; place-items: center; box-shadow: 0 1px 3px rgba(16, 24, 40, 0.08); }
  main { max-width: 1280px; margin: 0 auto; padding: 24px 32px 60px; }
  .content:has(> .menu) main { padding-left: 76px; }
  .scrim { display: none; }
  @media (max-width: 900px) {
    .scrim { display: block; position: fixed; inset: 0; z-index: 25; min-height: 0; padding: 0; border: 0; border-radius: 0; background: rgba(16, 24, 40, 0.25); }
    main { padding: 72px 16px 40px; }
    .content:has(> .menu) main { padding-left: 16px; }
  }
</style>
