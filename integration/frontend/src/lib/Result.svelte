<script lang="ts">
  import type { Overview, Result, Source, Images } from './types'
  import { batchColor, flagText, fmtKpi, pct, short, SOURCE_LABEL } from './format'
  import ProbBars from './ProbBars.svelte'
  import Scatter from './Scatter.svelte'
  import Legend from './Legend.svelte'

  let { result, overview }: { result: Result; overview: Overview } = $props()
  const classes = $derived(overview.classes)
  const meta = $derived(overview.kpi_meta)
  const pred = $derived(result.prediction)

  // pairwise comparison
  let pairA = $state<string>('')
  let pairB = $state<string>('')
  $effect(() => { pairA = result.prediction.predicted; pairB = result.prediction.runner_up })
  const pairRows = $derived(
    result.contributions
      .map((c) => ({ ...c, d: (c.per_batch[pairA] ?? 0) - (c.per_batch[pairB] ?? 0) }))
      .sort((a, b) => Math.abs(b.d) - Math.abs(a.d)),
  )
  const pairTotal = $derived(pairRows.reduce((s, r) => s + r.d, 0))
  const maxAbs = (xs: number[]) => Math.max(1e-9, ...xs.map(Math.abs))
  const pairMax = $derived(maxAbs(pairRows.map((r) => r.d)))
  const evMax = $derived(maxAbs(result.contributions.map((c) => c.evidence)))

  // feature tables
  let featSource = $state<Source>('rule')
  const used = $derived(new Set(overview.classifiers[featSource].kpis))
  const kpiList = $derived(Object.keys(meta).sort((a, b) => Number(used.has(b)) - Number(used.has(a))))
  const sampleVal = (s: Source, k: string) => (result.kpis[s] as any)?.[k]?.value as number | undefined

  // visuals
  type Panel = 'bse' | 'rule' | 'learned' | 'cracks'
  let panel = $state<Panel>('rule')
  let zoom = $state(true)
  const imgKey = (p: Panel, z: boolean): keyof Images =>
    (z ? { bse: 'zoom_bse', rule: 'zoom_rule', learned: 'zoom_learned', cracks: 'zoom_cracks' }
       : { bse: 'bse', rule: 'rule_overlay', learned: 'learned_overlay', cracks: 'cracks' })[p] as keyof Images

  const PRESETS: { source: Source; x: string; y: string; title: string }[] = [
    { source: 'rule', x: 'frac_pore', y: 'graphite_crack_density', title: 'Porosity vs graphite cracks (rule)' },
    { source: 'rule', x: 'si_cv_w256', y: 'frac_pore', title: 'Si heterogeneity vs porosity (rule)' },
    { source: 'learned', x: 'frac_pore', y: 'graphite_aspect_ratio_median', title: 'Porosity vs graphite shape (DINO)' },
    { source: 'learned', x: 'si_cv_w256', y: 'graphite_aspect_ratio_median', title: 'Si heterogeneity vs graphite shape (DINO)' },
  ]
  const coreKpis = $derived(
    overview.classifiers.rule.kpis.flatMap((k) => {
      const st = overview.batch_stats?.rule?.[k]?.[pred.predicted]
      if (!st || !meta[k]) return []
      const value = sampleVal('rule', k)
      const all = Object.values(overview.batch_stats.rule[k]).flatMap((b) => [b.p25, b.p75])
      if (value !== undefined && Number.isFinite(value)) all.push(value)
      const lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.1 || 1
      const pos = (x: number) => ((x - (lo - pad)) / (hi - lo + 2 * pad)) * 100
      const status = value === undefined || !Number.isFinite(value) ? 'na' : value < st.p25 ? 'below' : value > st.p75 ? 'above' : 'within'
      return [{ k, label: meta[k].label, value, status, pos, ...st }]
    }),
  )
  const flags = $derived(
    Array.isArray(result.trust_flags) ? result.trust_flags.filter(Boolean) : result.trust_flags ? String(result.trust_flags).split(/[;,\s]+/).filter(Boolean) : [],
  )
</script>

<section class="card head">
  <div class="sample">
    <span class="muted">Sample</span> <b>{result.sample_id}</b>
    {#if result.known_batch}
      <div class="muted">Known label: <b>{result.known_batch.replace('_', ' ')}</b>
        {result.known_batch === pred.predicted ? '✓ matches' : '✗ differs'} (training sample, so this is an in-sample check)</div>
    {/if}
  </div>
  <div class="verdict">
    <div class="muted">Most likely</div>
    <div class="big" style="color:{batchColor(pred.predicted)}">{pred.predicted.replace('_', ' ')}</div>
    <div class="conf"><span class="pctv">{pct(pred.probabilities[pred.predicted])}</span> <span class="pill tier-{pred.tier}">{pred.tier === 'review' ? 'needs review' : `${pred.tier} confidence`}</span></div>
  </div>
  <div class="flags">
    {#if pred.outlier}<div class="warn"><b>Unusual sample:</b> its KPIs are atypical for every known batch (typicality p &lt; 0.01). It may belong to none of them.</div>{/if}
    {#if pred.ambiguous && !pred.outlier}<div class="warn"><b>Ambiguous:</b> no batch reaches 70%.</div>{/if}
    {#each flags as f}{@const m = flagText(f)}<div class="warn"><b>{m.title}:</b> {m.body}</div>{/each}
    {#each result.warnings as w}<div class="warn">{w}</div>{/each}
  </div>
</section>

<div class="cols">
  <section class="card">
    <h2>Batch probabilities</h2>
    <ProbBars {classes} probabilities={pred.probabilities} interval={pred.interval} />
    <p class="muted small">± is the uncertainty: how far each probability could move with a different set of reference samples (shown by the black whiskers).</p>
    <h3>Key measurements vs {pred.predicted.replace('_', ' ')}</h3>
    <div class="kpis">
      {#each coreKpis as r}
        <div class="krow">
          <div class="klab">{r.label}</div>
          <div class="kval"><b>{fmtKpi(r.value, meta[r.k])}</b> <span class="kstat {r.status}">{r.status === 'within' ? 'typical' : r.status === 'above' ? 'above typical' : r.status === 'below' ? 'below typical' : ''}</span></div>
          <div class="krange">
            <div class="band" style="left:{r.pos(r.p25)}%; width:{r.pos(r.p75) - r.pos(r.p25)}%; background:{batchColor(pred.predicted)}"></div>
            <div class="med" style="left:{r.pos(r.median)}%"></div>
            {#if r.value !== undefined}<div class="dot" style="left:{r.pos(r.value)}%"></div>{/if}
          </div>
          <div class="muted small ktyp">{pred.predicted.replace('_', ' ')} typical: {fmtKpi(r.p25, meta[r.k])} – {fmtKpi(r.p75, meta[r.k])}</div>
        </div>
      {/each}
    </div>
    <p class="muted small">Coloured band: middle 50% of {pred.predicted.replace('_', ' ')} training samples; thin line: its median; dot: this sample.</p>
  </section>

  <section class="card">
    <h2>Why</h2>
    <ul class="expl">{#each result.explanation as e}<li>{e}</li>{/each}</ul>
    <h3>Evidence for {short(pred.predicted)} over {short(pred.runner_up)}, per KPI</h3>
    <div class="ev">
      {#each result.contributions as c}
        <span class="lab">{c.label} <small>({c.source === 'rule' ? 'rule' : 'DINO'})</small></span>
        <span class="val">{c.display}</span>
        <div class="axis">
          <div class="bar" style="{c.evidence >= 0 ? 'left:50%' : `right:50%`}; width:{(Math.abs(c.evidence) / evMax) * 50}%; background:{c.evidence >= 0 ? batchColor(pred.predicted) : batchColor(pred.runner_up)}"></div>
        </div>
      {/each}
    </div>
    <p class="muted small">Bars right favour {short(pred.predicted)}, left favour {short(pred.runner_up)}. Length = half the log-likelihood ratio (each classifier has half a vote).</p>
  </section>
</div>

<section class="card">
  <h2>Compare two batches</h2>
  <div class="pairsel">
    <select bind:value={pairA}>{#each classes as c}<option value={c}>{c.replace('_', ' ')}</option>{/each}</select>
    vs
    <select bind:value={pairB}>{#each classes as c}<option value={c}>{c.replace('_', ' ')}</option>{/each}</select>
    <span class="muted">
      {#if pairA !== pairB}
        P({short(pairA)}) / P({short(pairB)}) = {pct(pred.probabilities[pairA])} / {pct(pred.probabilities[pairB])}
        ≈ e<sup>{pairTotal.toFixed(2)}</sup> = {Math.exp(pairTotal).toFixed(pairTotal > 4 ? 0 : 2)}×
      {/if}
    </span>
  </div>
  {#if pairA !== pairB}
    <div class="ev">
      {#each pairRows as r}
        <span class="lab">{r.label} <small>({r.source === 'rule' ? 'rule' : 'DINO'})</small></span>
        <span class="val">{r.d >= 0 ? '+' : ''}{r.d.toFixed(2)}</span>
        <div class="axis"><div class="bar" style="{r.d >= 0 ? 'left:50%' : 'right:50%'}; width:{(Math.abs(r.d) / pairMax) * 50}%; background:{batchColor(r.d >= 0 ? pairA : pairB)}"></div></div>
      {/each}
    </div>
    <p class="muted small">Sum of the bars = log of the probability ratio. Uniform batch priors.</p>
  {/if}
</section>

<section class="card">
  <h2>Comparison with the baseline (Batch 3)</h2>
  <p>
    {#if result.baseline.consistent_with_baseline}
      <b>Consistent with Batch 3</b>: its KPIs fall within the Batch 3 spread
    {:else}
      <b>Differs from Batch 3</b>: its KPIs are atypical for Batch 3
    {/if}
    <span class="muted">(typicality p: rule {result.baseline.typicality.rule.toFixed(3)}, DINO {result.baseline.typicality.learned.toFixed(3)}; ≥ 0.05 = consistent)</span>
  </p>
  <div class="tscroll"><table>
    <thead><tr><th>KPI</th><th>Masks</th><th class="num">Sample</th><th class="num">B3 median</th><th class="num">B3 5–95%</th><th>Deviation (z)</th><th class="num">B3 percentile</th></tr></thead>
    <tbody>
      {#each result.baseline.kpis as b}
        <tr class:used={overview.classifiers[b.source].kpis.includes(b.kpi)}>
          <td>{b.label}</td><td class="muted">{b.source === 'rule' ? 'rule' : 'DINO'}</td>
          <td class="num">{fmtKpi(b.value, meta[b.kpi])}</td><td class="num">{fmtKpi(b.median, meta[b.kpi])}</td>
          <td class="num">{fmtKpi(b.p05, meta[b.kpi])} – {fmtKpi(b.p95, meta[b.kpi])}</td>
          <td><div class="zaxis"><div class="zbar {b.direction}" style="{(b.z ?? 0) >= 0 ? 'left:50%' : 'right:50%'}; width:{Math.min(Math.abs(b.z ?? 0), 3) / 3 * 50}%"></div></div>
            <small>{b.z !== null ? `${b.z >= 0 ? '+' : ''}${b.z.toFixed(1)}` : '–'} {b.direction !== 'typical' ? b.direction : ''}</small></td>
          <td class="num">{b.percentile !== null ? `${b.percentile.toFixed(0)}%` : '–'}</td>
        </tr>
      {/each}
    </tbody>
  </table></div>
  <p class="muted small">Bold rows are KPIs the classifiers use. z on the log/logit scale; |z| &lt; 1 counts as typical.</p>
</section>

<section class="card">
  <h2>Features vs each batch</h2>
  <div class="pairsel">
    <label>Masks <select bind:value={featSource}><option value="rule">{SOURCE_LABEL.rule}</option><option value="learned">{SOURCE_LABEL.learned}</option></select></label>
  </div>
  <div class="tscroll"><table>
    <thead>
      <tr><th>KPI</th><th class="num">This sample</th>
        {#each classes as c}<th class="num" style="color:{batchColor(c)}">{short(c)} median [IQR]</th>{/each}
        {#each classes as c}<th class="num" style="color:{batchColor(c)}">{short(c)} representative</th>{/each}
      </tr>
    </thead>
    <tbody>
      {#each kpiList as k}
        {@const sv = sampleVal(featSource, k)}
        <tr class:used={used.has(k)}>
          <td>{meta[k].label}{meta[k].unit && meta[k].unit !== '%' ? ` (${meta[k].unit})` : ''}</td>
          <td class="num"><b>{fmtKpi(sv, meta[k])}</b>
            {#if featSource === 'rule' && Number.isFinite(result.kpis.rule[k]?.lo) && Number.isFinite(result.kpis.rule[k]?.hi)}<br /><small class="muted">{fmtKpi(result.kpis.rule[k].lo, meta[k])}–{fmtKpi(result.kpis.rule[k].hi, meta[k])}</small>{/if}</td>
          {#each classes as c}
            {@const st = overview.batch_stats?.[featSource]?.[k]?.[c]}
            <td class="num">{st ? fmtKpi(st.median, meta[k]) : '–'}{#if st}<br /><small class="muted">{fmtKpi(st.p25, meta[k])}–{fmtKpi(st.p75, meta[k])}</small>{/if}</td>
          {/each}
          {#each classes as c}
            <td class="num">{fmtKpi(overview.representatives?.[c]?.kpis?.[featSource]?.[k], meta[k])}</td>
          {/each}
        </tr>
      {/each}
    </tbody>
  </table></div>
  <p class="muted small">Bold rows are used by this classifier. Small numbers under the sample (rule masks) are the range when pixels at phase boundaries are reassigned.</p>
</section>

<section class="card">
  <h2>Where the sample sits</h2>
  <div class="charts">
    {#each PRESETS as p}
      <Scatter rows={overview.training} source={p.source} x={p.x} y={p.y} {meta} {classes} title={p.title}
        point={{ x: sampleVal(p.source, p.x) ?? NaN, y: sampleVal(p.source, p.y) ?? NaN, label: 'this sample' }} />
    {/each}
  </div>
</section>

<section class="card">
  <h2>Masks next to a representative sample of each batch</h2>
  <div class="pairsel">
    {#each [['bse', 'BSE'], ['rule', 'Rule masks'], ['learned', 'DINO masks'], ['cracks', 'Graphite cracks']] as [k, l]}
      <button class:sel={panel === k} onclick={() => (panel = k as Panel)}>{l}</button>
    {/each}
    <label><input type="checkbox" bind:checked={zoom} /> zoom (600 px, full resolution)</label>
    <Legend legend={result.legend} />
  </div>
  <div class="vis" class:zoomed={zoom}>
    <figure>
      <img src={result.images[imgKey(panel, zoom)]} alt="sample {panel}" />
      <figcaption><b>This sample</b> → {pred.predicted.replace('_', ' ')}</figcaption>
    </figure>
    {#each classes as c}
      {@const rep = overview.representatives?.[c]}
      {#if rep}
        <figure>
          <img src={rep.images[imgKey(panel, zoom)]} alt="{c} representative {panel}" />
          <figcaption style="color:{batchColor(c)}"><b>{c.replace('_', ' ')}</b> representative <span class="muted">{rep.image_id}</span></figcaption>
        </figure>
      {/if}
    {/each}
  </div>
  <p class="muted small">Representative = the training sample most typical of its batch under the final model.</p>
</section>

<style>
  .head { display: grid; gap: 14px; }
  .sample { font-size: 0.95rem; }
  .flags { display: grid; gap: 6px; }
  .flags:empty { display: none; }
  .verdict { text-align: center; padding: 8px 0 12px; }
  .big { font-size: 3rem; font-weight: 750; line-height: 1.1; margin: 4px 0 8px; letter-spacing: -0.01em; }
  .conf { display: inline-flex; align-items: center; gap: 10px; }
  .pctv { font-size: 1.6rem; font-weight: 600; }
  .conf .pill { font-size: 0.95rem; padding: 4px 12px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .cols > * { min-width: 0; }
  .tscroll { overflow-x: auto; }
  .kpis { display: grid; gap: 14px; margin-top: 8px; }
  .krow { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; align-items: center; }
  .klab { font-weight: 500; }
  .kval { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
  .krange { grid-column: 1 / -1; position: relative; height: 12px; background: #eef0f3; border-radius: 999px; }
  .band { position: absolute; top: 0; bottom: 0; opacity: 0.35; border-radius: 999px; }
  .med { position: absolute; top: -2px; bottom: -2px; width: 2px; background: #555; transform: translateX(-1px); }
  .dot { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #1d1d1f; border: 2px solid #fff; box-shadow: 0 0 0 1px #1d1d1f; transform: translate(-50%, -50%); }
  .ktyp { grid-column: 1 / -1; }
  .kstat { font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-left: 4px; }
  .kstat.within { background: #dff3e4; color: #1d6b33; }
  .kstat.above, .kstat.below { background: #fff1d6; color: #8a5a00; }
  .small { font-size: 0.78rem; }
  .expl { margin: 0 0 12px; padding-left: 18px; display: grid; gap: 4px; }
  .ev { display: grid; grid-template-columns: minmax(150px, 1.2fr) 70px 1.5fr; gap: 4px 8px; align-items: center; font-size: 0.82rem; }
  .ev .val { text-align: right; font-variant-numeric: tabular-nums; }
  .axis, .zaxis { position: relative; height: 12px; background: linear-gradient(#bbb, #bbb) 50% / 1px 100% no-repeat; }
  .zaxis { display: inline-block; width: 90px; vertical-align: middle; margin-right: 6px; }
  .bar, .zbar { position: absolute; top: 1px; bottom: 1px; border-radius: 2px; }
  .zbar.typical { background: #9aa; } .zbar.higher { background: #c0504d; } .zbar.lower { background: #4f81bd; }
  .pairsel { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; font-size: 0.88rem; }
  button.sel { background: #14213d; color: #fff; border-color: #14213d; }
  tr.used td { font-weight: 600; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(300px, 100%), 1fr)); gap: 12px; }
  .vis { display: grid; grid-template-columns: 1fr; gap: 10px; }
  .vis.zoomed { grid-template-columns: repeat(4, 1fr); }
  .vis figure { margin: 0; }
  .vis img { width: 100%; border: 1px solid #ddd; border-radius: 4px; background: #000; image-rendering: auto; }
  figcaption { font-size: 0.8rem; }
  @media (max-width: 600px) { .ev { grid-template-columns: minmax(0, 1.2fr) 56px minmax(0, 1fr); } .big { font-size: 2.5rem; } }
  @media (max-width: 1000px) { .cols { grid-template-columns: 1fr; } .vis.zoomed { grid-template-columns: 1fr 1fr; } }
</style>
