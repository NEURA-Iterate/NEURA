<script lang="ts">
  let { kpi, sample, baseline, lo, hi, fmt }: {
    kpi: string; sample: number; baseline: number; lo: number; hi: number; fmt: (x: number) => string
  } = $props()

  let v = $state<number | null>(null)
  const cur = $derived(v ?? sample)
  const span = $derived(hi - lo || 1)
  const t = $derived(Math.min(1, Math.max(0, (cur - lo) / span)))
  const at = (x: number) => `${(Math.min(1, Math.max(0, (x - lo) / span)) * 100).toFixed(1)}%`

  function rng(seed: number) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0
      let r = Math.imul(seed ^ (seed >>> 15), 1 | seed)
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296
    }
  }
  const W = 260, H = 130
  const r1 = rng(7)
  const particles = Array.from({ length: 18 }, (_, i) => ({
    x: 22 + (i % 6) * 43 + (r1() - 0.5) * 12, y: 22 + Math.floor(i / 6) * 43 + (r1() - 0.5) * 10,
    r: 15 + r1() * 5, a: r1() * 180,
  }))
  const pores = Array.from({ length: 40 }, () => ({ x: r1() * W, y: r1() * H, r: 3 + r1() * 5 }))
  const cracks = Array.from({ length: 48 }, (_, i) => {
    const p = particles[i % particles.length]
    const ang = r1() * Math.PI, len = 5 + r1() * 9, ox = (r1() - 0.5) * p.r, oy = (r1() - 0.5) * p.r * 0.6
    return { x1: p.x + ox, y1: p.y + oy, x2: p.x + ox + Math.cos(ang) * len, y2: p.y + oy + Math.sin(ang) * len }
  })
  const cells = Array.from({ length: 40 }, (_, i) => ({ x: (i % 8) * (W / 8), y: Math.floor(i / 8) * (H / 5), n: r1() - 0.5 }))

  const INFO: Record<string, { low: string; high: string; note: string }> = {
    frac_pore: {
      low: 'Less pore space: denser electrode, but slower electrolyte access to the particles.',
      high: 'More pore space: easier electrolyte access, but lower density and energy per volume.',
      note: 'Blue = pores between graphite particles.',
    },
    graphite_crack_density: {
      low: 'Fewer cracks inside graphite: particles are mechanically intact.',
      high: 'More cracks inside graphite: a sign of damage during processing (e.g. calendering), which can reduce cycle life.',
      note: 'Red = cracks inside graphite particles.',
    },
    si_cv_w256: {
      low: 'Si spread evenly: swelling during charging is shared across the electrode.',
      high: 'Si in patches: local swelling hot-spots that can crack the electrode during cycling.',
      note: 'Each square = one window; deeper orange = more Si.',
    },
    graphite_aspect_ratio_median: {
      low: 'Rounder graphite particles: more uniform packing and ion paths.',
      high: 'More elongated (flaky) graphite: particles align when pressed, which makes ion paths longer.',
      note: 'Elongation exaggerated so the difference is visible.',
    },
  }
  const info = $derived(INFO[kpi])
</script>

{#if info}
<div class="kx">
  <svg viewBox="0 0 {W} {H}" role="img" aria-label="schematic">
    <rect width={W} height={H} fill="#eef3ea" />
    {#if kpi === 'si_cv_w256'}
      {#each cells as c}
        <rect x={c.x + 1} y={c.y + 1} width={W / 8 - 2} height={H / 5 - 2} rx="3" fill="#e8822a"
          fill-opacity={Math.min(0.95, Math.max(0.04, 0.45 + c.n * (0.1 + t * 1.8)))} />
      {/each}
    {:else}
      {#if kpi === 'frac_pore'}
        {#each pores.slice(0, Math.round(4 + t * 36)) as p}<circle cx={p.x} cy={p.y} r={p.r * (0.8 + t * 0.6)} fill="#2a4bd7" fill-opacity="0.85" />{/each}
      {/if}
      {#each particles as p}
        {@const ar = kpi === 'graphite_aspect_ratio_median' ? 1 + t * 2.2 : 1.35}
        <ellipse cx={p.x} cy={p.y} rx={p.r * Math.sqrt(ar)} ry={p.r / Math.sqrt(ar)} transform="rotate({kpi === 'graphite_aspect_ratio_median' ? p.a * 0.15 : p.a} {p.x} {p.y})" fill="#5d6470" stroke="#3f444c" stroke-width="0.6" />
      {/each}
      {#if kpi === 'graphite_crack_density'}
        {#each cracks.slice(0, Math.round(3 + t * 45)) as c}<line {...c} stroke="#e02424" stroke-width="1.6" stroke-linecap="round" />{/each}
      {/if}
    {/if}
  </svg>
  <div class="track">
    <input type="range" min={lo} max={hi} step={span / 200} value={cur} oninput={(e) => (v = +e.currentTarget.value)} aria-label="value" />
    <span class="mk base" style="left:{at(baseline)}" title="Baseline median"></span>
    <span class="mk samp" style="left:{at(sample)}" title="This sample"></span>
  </div>
  <div class="btns">
    <button class:on={v === baseline} onclick={() => (v = baseline)}><i class="dot base"></i>Baseline {fmt(baseline)}</button>
    <button class:on={v === null || v === sample} onclick={() => (v = null)}><i class="dot samp"></i>This sample {fmt(sample)}</button>
  </div>
  <div class="small kxt"><b>{cur >= baseline ? 'Higher' : 'Lower'} than baseline means:</b> {cur >= baseline ? info.high : info.low}</div>
  <div class="muted tiny">{info.note} Schematic, not the real image.</div>
</div>
{/if}

<style>
  .kx { display: grid; gap: 6px; }
  svg { width: 100%; border-radius: 10px; display: block; }
  .track { position: relative; height: 22px; }
  .track input { width: 100%; margin: 0; }
  .mk { position: absolute; top: 17px; width: 10px; height: 10px; border-radius: 50%; transform: translateX(-50%); pointer-events: none; }
  .base, .dot.base { background: #4c9a5f; }
  .samp, .dot.samp { background: #111827; }
  .btns { display: flex; gap: 4px; flex-wrap: nowrap; margin-top: 4px; }
  .btns button { font-size: 0.72rem; padding: 3px 7px; white-space: nowrap; border-radius: 999px; border: 1px solid #d1d5db; background: #fff; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; }
  .btns button.on { border-color: #111827; font-weight: 700; }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .kxt { font-size: 0.82rem; color: #1f2937; }
  .tiny { font-size: 0.7rem; }
</style>
