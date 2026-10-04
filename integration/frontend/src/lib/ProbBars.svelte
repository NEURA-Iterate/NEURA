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
            title="90% bootstrap interval {pct(iv[0])}–{pct(iv[1])}"></div>
        {/if}
      </div>
      <span class="val">{pct(p)}{#if iv && !compact}<small> [{pct(iv[0])}–{pct(iv[1])}]</small>{/if}</span>
    </div>
  {/each}
</div>

<style>
  .bars { display: grid; gap: 6px; }
  .row { display: grid; grid-template-columns: 72px 1fr 130px; align-items: center; gap: 8px; }
  .compact .row { grid-template-columns: 26px 1fr 40px; gap: 5px; }
  .name { font-size: 0.85rem; }
  .track { position: relative; height: 18px; background: #eee; border-radius: 3px; overflow: hidden; }
  .compact .track { height: 10px; }
  .fill { height: 100%; }
  .ci { position: absolute; top: 40%; height: 20%; background: #111; opacity: 0.75; }
  .ci::before, .ci::after { content: ''; position: absolute; top: -150%; height: 400%; width: 2px; background: #111; }
  .ci::before { left: 0; } .ci::after { right: 0; }
  .val { font-variant-numeric: tabular-nums; font-size: 0.85rem; }
  .val small { color: #666; }
</style>
