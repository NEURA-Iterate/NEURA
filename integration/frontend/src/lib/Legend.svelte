<script lang="ts">
  import type { Legend } from './types'
  import { legendItems } from './format'
  let { legend }: { legend: Legend | undefined } = $props()
</script>

<div class="legend">
  {#each legendItems(legend) as it}
    <span><i style="background:{it.color}"></i>{it.name}</span>
  {/each}
</div>

<style>
  .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.8rem; }
  i { display: inline-block; width: 12px; height: 12px; margin-right: 4px; vertical-align: -2px; border: 1px solid #0003; }
</style>
