<script lang="ts">
  import type { ByBatch } from './types'
  import { batchColor, pct, short } from './format'

  let { classes, probabilities, interval = null, compact = false }: {
    classes: string[]; probabilities: ByBatch<number>; interval?: ByBatch<[number, number]> | null; compact?: boolean
  } = $props()
</script>

<div class="bars" class:compact>
  {#each classes as c}
    {@const p = probabilities[c] ?? 0}
    {@const iv = interval?.[c]}
    <div class="row">
      <span class="name">{compact ? short(c) : c.replace('_', ' ')}</span>
      <div class="track">
        <div class="fill" style="width:{p * 100}%; background:{batchColor(c)}"></div>
        {#if iv}
          <div class="ci" style="left:{iv[0] * 100}%; width:{Math.max((iv[1] - iv[0]) * 100, 0.4)}%"
            title="Uncertainty: {pct(iv[0])}–{pct(iv[1])}"></div>
        {/if}
      </div>
      <span class="val">{pct(p)}{#if iv && !compact}<small> ± {pct((iv[1] - iv[0]) / 2)}</small>{/if}</span>
    </div>
  {/each}
</div>

<style>
  .bars { display: grid; gap: 12px; }
  .row { display: grid; grid-template-columns: 90px 1fr 140px; align-items: center; gap: 12px; }
  .compact .row { grid-template-columns: 26px 1fr 40px; gap: 5px; }
  .name { font-size: 1.1rem; font-weight: 500; }
  .compact .name { font-size: 0.85rem; font-weight: 400; }
  .track { position: relative; height: 24px; background: #eee; border-radius: 6px; overflow: hidden; }
  .compact .track { height: 10px; }
  .fill { height: 100%; }
  .ci { position: absolute; top: 40%; height: 20%; background: #111; opacity: 0.75; }
  .ci::before, .ci::after { content: ''; position: absolute; top: -150%; height: 400%; width: 2px; background: #111; }
  .ci::before { left: 0; } .ci::after { right: 0; }
  .val { font-variant-numeric: tabular-nums; font-size: 1.25rem; font-weight: 600; white-space: nowrap; }
  .val small { color: #666; font-size: 0.95rem; font-weight: 500; }
  .compact .val { font-size: 0.85rem; font-weight: 400; }
  @media (max-width: 600px) { .row { grid-template-columns: 70px 1fr 118px; gap: 8px; } .name { font-size: 1rem; } .val { font-size: 1.1rem; } .val small { font-size: 0.85rem; } }
</style>
