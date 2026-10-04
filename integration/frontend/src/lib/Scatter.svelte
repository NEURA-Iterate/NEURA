<script lang="ts">
  import type { KpiMeta, TrainingRow, Source } from './types'
  import { batchColor, fmtKpi, short } from './format'

  interface Point { x: number; y: number; label: string }
  let {
    rows, source, x, y, meta, classes, point = null, title = '',
  }: {
    rows: TrainingRow[]; source: Source; x: string; y: string
    meta: Record<string, KpiMeta>; classes: string[]; point?: Point | null; title?: string
  } = $props()

  const W = 380, H = 300, M = { l: 54, r: 12, t: 24, b: 42 }
  const isLog = (k: string) => meta[k]?.transform === 'log'
  const tf = (k: string, v: number) => (isLog(k) ? Math.log10(v) : v)

  const pts = $derived(
    rows
      .map((r) => ({ id: r.image_id, b: r.batch, vx: r[source][x], vy: r[source][y] }))
      .filter((p) => Number.isFinite(p.vx) && Number.isFinite(p.vy) && (!isLog(x) || p.vx > 0) && (!isLog(y) || p.vy > 0)),
  )
  function domain(k: string, vals: number[]): [number, number] {
    const t = vals.map((v) => tf(k, v))
    let lo = Math.min(...t), hi = Math.max(...t)
    const pad = (hi - lo) * 0.08 || 0.1
    return [lo - pad, hi + pad]
  }
  const xs = $derived(domain(x, [...pts.map((p) => p.vx), ...(point && Number.isFinite(point.x) ? [point.x] : [])]))
  const ys = $derived(domain(y, [...pts.map((p) => p.vy), ...(point && Number.isFinite(point.y) ? [point.y] : [])]))
  const sx = (v: number) => M.l + ((tf(x, v) - xs[0]) / (xs[1] - xs[0])) * (W - M.l - M.r)
  const sy = (v: number) => H - M.b - ((tf(y, v) - ys[0]) / (ys[1] - ys[0])) * (H - M.t - M.b)
  function ticks(k: string, d: [number, number]) {
    return Array.from({ length: 5 }, (_, i) => {
      const t = d[0] + ((i + 0.5) / 5) * (d[1] - d[0])
      return isLog(k) ? 10 ** t : t
    })
  }
  let hover = $state<string | null>(null)
</script>

<figure class="scatter">
  {#if title}<figcaption>{title}</figcaption>{/if}
  <svg viewBox="0 0 {W} {H}" role="img" aria-label={title}>
    <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} class="plot" />
    {#each ticks(x, xs) as t}
      <text x={sx(t)} y={H - M.b + 14} class="tick" text-anchor="middle">{fmtKpi(t, meta[x])}</text>
    {/each}
    {#each ticks(y, ys) as t}
      <text x={M.l - 5} y={sy(t) + 3} class="tick" text-anchor="end">{fmtKpi(t, meta[y])}</text>
    {/each}
    <text x={(W + M.l) / 2} y={H - 6} class="axis" text-anchor="middle">{meta[x]?.label ?? x}{isLog(x) ? ' (log)' : ''}</text>
    <text transform="translate(12 {(H - M.b + M.t) / 2}) rotate(-90)" class="axis" text-anchor="middle">{meta[y]?.label ?? y}{isLog(y) ? ' (log)' : ''}</text>
    {#each pts as p (p.id)}
      <circle cx={sx(p.vx)} cy={sy(p.vy)} r={hover === p.id ? 6 : 4.5} fill={batchColor(p.b)} fill-opacity="0.8"
        stroke="#fff" stroke-width="1" role="presentation"
        onmouseenter={() => (hover = p.id)} onmouseleave={() => (hover = null)}>
        <title>{p.id.replace('img_', '')} ({short(p.b)}): {fmtKpi(p.vx, meta[x])}, {fmtKpi(p.vy, meta[y])}</title>
      </circle>
    {/each}
    {#if point && Number.isFinite(point.x) && Number.isFinite(point.y)}
      <g transform="translate({sx(point.x)} {sy(point.y)})">
        <path d="M0 -9 L2.6 -2.8 L9 -2.8 L3.8 1.2 L5.6 7.6 L0 3.8 L-5.6 7.6 L-3.8 1.2 L-9 -2.8 L-2.6 -2.8 Z" fill="#111" stroke="#ffd400" stroke-width="1.2" />
        <text x="11" y="4" class="pointlabel">{point.label}</text>
      </g>
    {/if}
    {#each classes as c, i}
      <g transform="translate({M.l + 6 + i * 50} {M.t - 8})">
        <circle r="4" fill={batchColor(c)} /><text x="7" y="3.5" class="tick">{short(c)}</text>
      </g>
    {/each}
  </svg>
</figure>

<style>
  .scatter { margin: 0; }
  figcaption { font-weight: 600; font-size: 0.85rem; margin-bottom: 2px; }
  svg { width: 100%; max-width: 420px; height: auto; }
  .plot { fill: #fafafa; stroke: #ddd; }
  .tick { font-size: 9px; fill: #555; }
  .axis { font-size: 10.5px; fill: #222; }
  .pointlabel { font-size: 10px; font-weight: 700; fill: #111; }
</style>
