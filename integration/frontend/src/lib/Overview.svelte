<script lang="ts">
  import type { Overview, Source } from './types'
  import { batchColor, fmtKpi, pct, short, SOURCE_LABEL } from './format'
  import Scatter from './Scatter.svelte'

  let { overview }: { overview: Overview } = $props()
  const v = $derived(overview.validation)
  const classes = $derived(overview.classes)

  const PRESETS: { source: Source; x: string; y: string; title: string }[] = [
    { source: 'rule', x: 'frac_pore', y: 'graphite_crack_density', title: 'Porosity vs graphite cracks (rule masks)' },
    { source: 'rule', x: 'si_cv_w256', y: 'frac_pore', title: 'Si heterogeneity vs porosity (rule masks)' },
    { source: 'learned', x: 'frac_pore', y: 'graphite_aspect_ratio_median', title: 'Porosity vs graphite shape (DINO masks)' },
    { source: 'learned', x: 'si_cv_w256', y: 'graphite_aspect_ratio_median', title: 'Si heterogeneity vs graphite shape (DINO masks)' },
  ]
  const kpiNames = $derived(Object.keys(overview.kpi_meta))
  let cSource = $state<Source>('rule')
  let cx = $state('si_fraction_of_solids')
  let cy = $state('si_ecd_d50')

  const rows: [string, 'rule' | 'learned' | 'combined'][] = [
    ['Rule-mask KPIs', 'rule'], ['DINO-mask KPIs', 'learned'], ['Combined (final model)', 'combined'],
  ]
  const perSample = $derived(
    [...v.per_sample].sort((a, b) => a.true.localeCompare(b.true) || a.image_id.localeCompare(b.image_id)),
  )
  let pairA = $state('Batch_1')
  let pairB = $state('Batch_2')
  const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const pairRows = $derived.by(() => {
    const kpis = [...new Set([...overview.classifiers.rule.kpis, ...overview.classifiers.learned.kpis])]
    const rows = kpis.flatMap((k) => {
      const srcs = (['rule', 'learned'] as const).filter((s) => overview.classifiers[s].kpis.includes(k))
      const stat = (b: string) => {
        const sts = srcs.map((s) => overview.batch_stats?.[s]?.[k]?.[b]).filter((x): x is NonNullable<typeof x> => !!x)
        return sts.length ? { median: avg(sts.map((x) => x.median)), p25: avg(sts.map((x) => x.p25)), p75: avg(sts.map((x) => x.p75)) } : null
      }
      const a = stat(pairA), b = stat(pairB)
      if (!a || !b) return []
      const xs = [a.p25, a.p75, b.p25, b.p75]
      const lo = Math.min(...xs), hi = Math.max(...xs), pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.05 || 1
      const pos = (x: number) => ((x - (lo - pad)) / (hi - lo + 2 * pad)) * 100
      const spread = (a.p75 - a.p25 + b.p75 - b.p25) / 2
      const sep = Math.abs(a.median - b.median) / Math.max(spread, 1e-9)
      const rel = b.median ? (a.median - b.median) / Math.abs(b.median) : 0
      return [{ k, label: overview.kpi_meta[k]?.label ?? k, a, b, pos, sep, rel }]
    })
    const top = Math.max(1e-9, ...rows.map((r) => r.sep))
    return rows.map((r) => ({ ...r, weight: r.sep / top })).sort((x, y) => y.sep - x.sep)
  })
  const predOf = (s: (typeof perSample)[number]) =>
    classes.reduce((best, c) => ((s as any)[`P_${c}`] > (s as any)[`P_${best}`] ? c : best), classes[0])
</script>

<div class="grid">
  <section class="card">
    <h2>How it works</h2>
    <ol class="pipeline">
      <li><b>One sample = one location</b>, imaged with three detectors: BSE, ETD (or SE) and Inlens.</li>
      <li><b>Two segmentations</b> split the image into Si, graphite, pore and binder: hand-written <i>rules</i>, and a <i>DINOv2 + fusion</i> network trained to reproduce them.</li>
      <li><b>KPIs</b> are measured on each set of masks, e.g. porosity, graphite cracks (from ETD), graphite shape, Si heterogeneity.</li>
      <li><b>Two Bayesian classifiers</b> (one per mask set) give P(Batch 1/2/3) from 3 KPIs each.</li>
      <li><b>Final answer</b> = geometric mean of the two, P ∝ √(P<sub>rule</sub> · P<sub>DINO</sub>), so each gets half a vote.</li>
    </ol>
    <p class="muted">
      Final classifiers are fitted on all {overview.training.length} labelled samples
      ({classes.map((c) => `${overview.n_per_batch[c]} ${short(c)}`).join(' / ')}).
      Rule KPIs: {overview.classifiers.rule.kpis.map((k) => overview.kpi_meta[k]?.label ?? k).join(', ')}.
      DINO KPIs: {overview.classifiers.learned.kpis.map((k) => overview.kpi_meta[k]?.label ?? k).join(', ')}.
    </p>
  </section>

  <section class="card">
    <h2>Validation (held out samples)</h2>
    <p class="muted">
      17 rounds; each round trains without one sample per batch and predicts those 3 → 51 predictions
      (17 per batch). Batch 1 and 2 have 7 samples, so their samples are held out 2–3 times;
      "Samples correct" averages those repeats into one prediction per sample (31).
    </p>
    <div class="tscroll">
    <table>
      <thead><tr><th>Classifier</th><th class="num">Correct</th>
        {#each classes as c}<th class="num">{short(c)} acc.</th>{/each}
        <th class="num">Samples correct</th><th class="num">Log-loss</th></tr></thead>
      <tbody>
        {#each rows as [label, key]}
          {@const s = v.summary[key]}
          <tr class:best={key === 'combined'}>
            <td>{label}</td><td class="num">{s.correct}/{s.n}</td>
            {#each classes as c}<td class="num">{(s as any)[`acc_${c}`]?.toFixed(2)}</td>{/each}
            <td class="num">{s.image_correct}/{overview.training.length}</td>
            <td class="num">{s.log_loss.toFixed(2)}</td>
          </tr>
        {/each}
      </tbody>
    </table>
    </div>
    {#if v.note}<p class="muted">{v.note}</p>{/if}

    <div class="two">
      <div>
        <h3>Held-out probabilities per sample (combined)</h3>
        <div class="stack">
          {#each perSample as s (s.image_id)}
            {@const wrong = predOf(s) !== s.true}
            <div class="srow" title="{s.image_id}: {classes.map((c) => `${short(c)} ${pct((s as any)[`P_${c}`])}`).join(', ')}">
              <span class="sid" style="color:{batchColor(s.true)}">{s.image_id.replace('img_', '')}</span>
              <div class="sbar">
                {#each classes as c}<div style="width:{(s as any)[`P_${c}`] * 100}%; background:{batchColor(c)}"></div>{/each}
              </div>
              <span class="mark">{wrong ? '✗' : ''}{s.trust_flags ? '*' : ''}</span>
            </div>
          {/each}
        </div>
        <p class="muted">Name colour = true batch; ✗ = wrong on average; * = image-quality flag.</p>
      </div>
      <div>
        <h3>Confusion (51 predictions)</h3>
        <table class="conf">
          <thead><tr><th>true ↓ / predicted →</th>{#each classes as c}<th class="num">{short(c)}</th>{/each}</tr></thead>
          <tbody>
            {#each classes as t}
              <tr><td>{short(t)}</td>
                {#each classes as p}
                  {@const n = v.confusion?.[t]?.[p] ?? 0}
                  <td class="num" style="background:{t === p ? `rgba(85,168,104,${n / 20})` : n ? `rgba(220,60,60,${n / 10})` : ''}">{n}</td>
                {/each}
              </tr>
            {/each}
          </tbody>
        </table>
        <h3 style="margin-top:14px">Is the confidence tier trustworthy?</h3>
        <table>
          <thead><tr><th>Tier</th><th>Rule</th><th class="num">Predictions</th><th class="num">Accuracy</th></tr></thead>
          <tbody>
            {#each v.tiers as t}
              <tr><td><span class="pill tier-{t.tier}">{t.tier}</span></td>
                <td class="muted">{t.tier === 'high' ? 'P ≥ 90%' : t.tier === 'medium' ? '60–90%' : '< 60%'}</td>
                <td class="num">{t.n}</td><td class="num">{pct(t.accuracy)}</td></tr>
            {/each}
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Feature space</h2>
    <p class="muted">Each dot is one labelled sample. Batch 3 separates on porosity and cracks; Batch 1 and 2 overlap and differ weakly in Si heterogeneity and graphite shape.</p>
    <div class="charts">
      {#each PRESETS as p}
        <Scatter rows={overview.training} source={p.source} x={p.x} y={p.y} meta={overview.kpi_meta} {classes} title={p.title} />
      {/each}
    </div>
    <div class="custom">
      <h3>Explore any pair</h3>
      <label>Masks <select bind:value={cSource}>
        <option value="rule">{SOURCE_LABEL.rule}</option><option value="learned">{SOURCE_LABEL.learned}</option></select></label>
      <label>x <select bind:value={cx}>{#each kpiNames as k}<option value={k}>{overview.kpi_meta[k].label}</option>{/each}</select></label>
      <label>y <select bind:value={cy}>{#each kpiNames as k}<option value={k}>{overview.kpi_meta[k].label}</option>{/each}</select></label>
      <Scatter rows={overview.training} source={cSource} x={cx} y={cy} meta={overview.kpi_meta} {classes} />
    </div>
  </section>

  <section class="card">
    <h2>Compare two batches</h2>
    <div class="pairsel">
      <select bind:value={pairA}>{#each classes as c}<option value={c}>{c.replace('_', ' ')}</option>{/each}</select>
      <span class="muted">vs</span>
      <select bind:value={pairB}>{#each classes as c}<option value={c}>{c.replace('_', ' ')}</option>{/each}</select>
    </div>
    {#if pairA === pairB}
      <p class="muted">Pick two different batches.</p>
    {:else}
      <div class="cmp">
        {#each pairRows as r (r.k)}
          <div class="crow">
            <div class="chead"><span class="clab">{r.label}</span>
              <span class="ctag" style="background:{batchColor(pairA)}1f; color:{batchColor(pairA)}">
                {short(pairA)} {r.rel >= 0 ? 'higher' : 'lower'} by {pct(Math.abs(r.rel))}
                <span class="cweight"><span style="width:{r.weight * 100}%; background:{batchColor(pairA)}"></span></span>
              </span>
            </div>
            <div class="cline">
              {#each [[pairA, r.a], [pairB, r.b]] as [b, st], i}
                {@const s = st as { median: number; p25: number; p75: number }}
                <div class="cband" style="left:{r.pos(s.p25)}%; width:{Math.max(r.pos(s.p75) - r.pos(s.p25), 0.8)}%; background:{batchColor(b as string)}; top:{3 + i * 11}px"></div>
                <div class="cmed" style="left:{r.pos(s.median)}%; background:{batchColor(b as string)}; top:{1 + i * 11}px"></div>
              {/each}
            </div>
            <div class="small">
              <span style="color:{batchColor(pairA)}">{short(pairA)} typical {fmtKpi(r.a.median, overview.kpi_meta[r.k])}</span> ·
              <span style="color:{batchColor(pairB)}">{short(pairB)} typical {fmtKpi(r.b.median, overview.kpi_meta[r.k])}</span>
            </div>
          </div>
        {/each}
      </div>
      <p class="muted small">KPIs used by the classifiers, most different first (rule and DINO values averaged). Bands: middle 50% of each batch's training samples, tick: median. The small bar shows how clearly the KPI separates the two batches.</p>
    {/if}
  </section>

  {#if overview.figures?.length}
    <section class="card">
      <h2>Research figures</h2>
      <div class="figs">
        {#each overview.figures as f}
          <a href={f.url} target="_blank" rel="noreferrer"><img src={f.url} alt={f.title} loading="lazy" /><span>{f.title}</span></a>
        {/each}
      </div>
    </section>
  {/if}

  <section class="card">
    <h2>Limits and uncertainty</h2>
    <ul class="muted">
      <li>QC decision: accept if P(baseline) ≥ 70% and its uncertainty range stays above 50%, reject if ≤ 30% and it stays below 50%, otherwise investigate. These limits are a policy choice for the QC team, not fitted to data.</li>
      <li>A sample whose measurements are unusual for every reference batch (typicality p &lt; 0.01) is flagged as possibly new variation and sent to investigate, rather than forced into a known batch.</li>
      <li>Probability uncertainty comes from refitting the classifiers on 200 bootstrap resamples of the 31 reference samples. Measurement uncertainty comes from reassigning pixels at phase boundaries.</li>
      <li>Reliability checks: porosity and Si heterogeneity agree closely between rule and DINO masks (ρ ≈ 0.96); graphite aspect ratio less so (ρ ≈ 0.6).</li>
      <li>Si heterogeneity is dominated by where the few Si agglomerates fall: its left-half and right-half values of the same image barely agree (ρ ≈ 0). Treat it as weak evidence; more image area per sample would help.</li>
      <li>Only 31 labelled samples (7 / 7 / 17); the validation numbers are optimistic because many variants were tried on them.</li>
      <li>The DINO segmentation was trained on rule masks, so it is not independently validated; expert-labelled masks are needed for that.</li>
      <li>Sizes are in pixels (the TIFFs carry no pixel size), so all samples must share the same magnification.</li>
    </ul>
  </section>
</div>

<style>
  .pipeline { margin: 0 0 8px; padding-left: 20px; display: grid; gap: 3px; }
  tr.best td { font-weight: 700; background: #f2f8f3; }
  .two { display: grid; grid-template-columns: 1.3fr 1fr; gap: 24px; margin-top: 14px; }
  .stack { display: grid; gap: 2px; }
  .srow { display: grid; grid-template-columns: 80px 1fr 22px; gap: 6px; align-items: center; font-size: 0.75rem; }
  .sid { font-family: ui-monospace, monospace; }
  .sbar { display: flex; height: 11px; border-radius: 2px; overflow: hidden; }
  .mark { color: #c22; font-weight: 700; }
  .conf td, .conf th { text-align: center; }
  .charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .custom { margin-top: 12px; max-width: 440px; }
  .custom label { margin-right: 10px; font-size: 0.85rem; }
  .figs { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
  .figs a { display: grid; gap: 4px; color: inherit; text-decoration: none; font-size: 0.8rem; }
  .figs img { width: 100%; border: 1px solid #eee; border-radius: 4px; }
  .pairsel { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }
  .tscroll { overflow-x: auto; }
  .cmp { display: grid; gap: 18px; max-width: 760px; }
  .crow { display: grid; gap: 6px; }
  .chead { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
  .clab { font-weight: 500; }
  .ctag { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
  .cweight { display: inline-block; width: 40px; height: 6px; border-radius: 999px; background: rgba(0, 0, 0, 0.08); overflow: hidden; }
  .cweight span { display: block; height: 100%; border-radius: 999px; }
  .cline { position: relative; height: 24px; background: linear-gradient(#e3e6ea, #e3e6ea) 0 50% / 100% 2px no-repeat; }
  .cband { position: absolute; height: 7px; border-radius: 999px; opacity: 0.45; }
  .cmed { position: absolute; width: 3px; height: 11px; border-radius: 2px; transform: translateX(-1.5px); }
  .small { font-size: 0.82rem; }
  @media (max-width: 900px) { .two { grid-template-columns: 1fr; } }
</style>
