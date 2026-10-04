<script lang="ts">
  import type { Overview, Result, Source, Images } from './types'
  import { batchColor, batchName, flagText, fmtKpi, pct, short, SOURCE_LABEL } from './format'
  import ProbBars from './ProbBars.svelte'
  import Scatter from './Scatter.svelte'
  import Legend from './Legend.svelte'

  let { result, overview }: { result: Result; overview: Overview } = $props()
  const classes = $derived(overview.classes)
  const meta = $derived(overview.kpi_meta)
  const pred = $derived(result.prediction)


  // feature tables
  let featSource = $state<Source>('rule')
  let showAllBaseline = $state(false)
  const isUsed = (b: { source: Source; kpi: string }) => overview.classifiers[b.source].kpis.includes(b.kpi)
  const baselineRows = $derived(showAllBaseline ? result.baseline.kpis : result.baseline.kpis.filter(isUsed))
  const used = $derived(new Set(overview.classifiers[featSource].kpis))
  let showAllFeatures = $state(false)
  const kpiList = $derived(
    Object.keys(meta)
      .filter((k) => showAllFeatures || used.has(k))
      .sort((a, b) => Number(used.has(b)) - Number(used.has(a))),
  )
  const sampleVal = (s: Source, k: string) => (result.kpis[s] as any)?.[k]?.value as number | undefined

  // visuals
  type Panel = 'bse' | 'rule' | 'learned' | 'cracks'
  let panel = $state<Panel>('rule')
  let zoom = $state(true)
  const imgKey = (p: Panel, z: boolean): keyof Images =>
    (z ? { bse: 'zoom_bse', rule: 'zoom_rule', learned: 'zoom_learned', cracks: 'zoom_cracks' }
       : { bse: 'bse', rule: 'rule_overlay', learned: 'learned_overlay', cracks: 'cracks' })[p] as keyof Images

  const PRESETS: { source: Source; x: string; y: string; title: string }[] = [
    { source: 'rule', x: 'frac_pore', y: 'graphite_crack_density', title: 'Porosity vs graphite cracks' },
    { source: 'rule', x: 'si_cv_w256', y: 'frac_pore', title: 'Si heterogeneity vs porosity' },
    { source: 'learned', x: 'frac_pore', y: 'graphite_aspect_ratio_median', title: 'Porosity vs graphite shape' },
    { source: 'learned', x: 'si_cv_w256', y: 'graphite_aspect_ratio_median', title: 'Si heterogeneity vs graphite shape' },
  ]
  const KPI_VISUAL: Record<string, { key: keyof Images; caption: string }> = {
    frac_pore: { key: 'kpi_pore', caption: 'Pores shown in blue across the whole image.' },
    graphite_crack_density: { key: 'kpi_cracks', caption: 'Cracks inside graphite in red, in the most cracked region (full resolution).' },
    si_cv_w256: { key: 'kpi_si', caption: 'Si share of each 256 px window: pale = little Si, deep orange = a lot. Patchier colours mean higher heterogeneity.' },
  }
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
      const v = KPI_VISUAL[k]
      const src = v ? result.images?.[v.key] : undefined
      return [{ k, label: meta[k].label, value, status, pos, img: src ? { src, caption: v.caption } : null, noImg: !!v && !src, ...st }]
    }),
  )
  const whyRows = $derived.by(() => {
    const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    const groups = new Map<string, typeof result.contributions>()
    for (const c of result.contributions) groups.set(c.kpi, [...(groups.get(c.kpi) ?? []), c])
    const rows = [...groups.entries()].map(([kpi, cs]) => {
      const value = avg(cs.map((c) => c.value))
      const batches = classes.map((b) => {
        const sts = cs.map((c) => overview.batch_stats?.[c.source]?.[kpi]?.[b]).filter((x): x is NonNullable<typeof x> => !!x)
        const st = sts.length ? { median: avg(sts.map((x) => x.median)), p25: avg(sts.map((x) => x.p25)), p75: avg(sts.map((x) => x.p75)) } : undefined
        return { b, st }
      })
      const xs = [value, ...batches.flatMap((x) => (x.st ? [x.st.p25, x.st.p75, x.st.median] : []))].filter(Number.isFinite)
      const lo = Math.min(...xs), hi = Math.max(...xs), pad = (hi - lo) * 0.12 || Math.abs(hi) * 0.05 || 1
      const pos = (x: number) => ((x - (lo - pad)) / (hi - lo + 2 * pad)) * 100
      const ll = classes.map((b) => cs.reduce((t, c) => t + (c.per_batch[b] ?? -Infinity), 0))
      const order = classes.map((b, i) => ({ b, l: ll[i] })).sort((a, b) => b.l - a.l)
      const best = order[0].b, runnerUp = order[1]?.b
      // This sample's evidence: log-odds of the favoured batch over the next most compatible one.
      const margin = order.length > 1 && Number.isFinite(order[0].l - order[1].l) ? order[0].l - order[1].l : 0
      const strength = margin < 0.25 ? 'slight' : margin < 1 ? 'moderate' : 'strong'
      return { kpi, label: cs[0].label, value, display: fmtKpi(value, meta[kpi]), batches, pos, best, runnerUp, margin, strength }
    })
    const top = Math.max(1, ...rows.map((r) => r.margin))
    return rows.map((r) => ({ ...r, weight: r.margin / top })).sort((a, b) => b.margin - a.margin)
  })
  const flags = $derived(
    Array.isArray(result.trust_flags) ? result.trust_flags.filter(Boolean) : result.trust_flags ? String(result.trust_flags).split(/[;,\s]+/).filter(Boolean) : [],
  )

  const KPI_WHY: Record<string, string> = {
    frac_pore: 'Pore space affects electrolyte access and electrode density.',
    graphite_crack_density: 'Cracks inside graphite can indicate mechanical damage during processing, e.g. calendering.',
    si_cv_w256: 'Uneven Si distribution can create local swelling hot-spots during cycling.',
    graphite_aspect_ratio_median: 'Graphite particle shape reflects the supplied powder and how it was processed.',
  }
  const drivers = $derived.by(() => {
    const groups = new Map<string, typeof result.baseline.kpis>()
    for (const b of result.baseline.kpis.filter(isUsed)) groups.set(b.kpi, [...(groups.get(b.kpi) ?? []), b])
    const avg = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
    return [...groups.entries()]
      .map(([kpi, bs]) => {
        const zs = bs.map((b) => b.z).filter((z): z is number => z !== null && Number.isFinite(z))
        const z = zs.length ? avg(zs) : 0
        const value = avg(bs.map((b) => b.value)), median = avg(bs.map((b) => b.median))
        const rel = median ? ((value - median) / Math.abs(median)) * 100 : 0
        const dir = Math.abs(z) < 1 ? 'typical' : z > 0 ? 'higher' : 'lower'
        return { kpi, label: bs[0].label, z, value, median, rel, dir, why: KPI_WHY[kpi] ?? '' }
      })
      .sort((a, b) => Math.abs(b.z) - Math.abs(a.z))
  })
  // QC decision against the approved baseline (Batch 3). Thresholds are a policy choice, not fitted.
  const ACCEPT_AT = 0.7, REJECT_AT = 0.3
  const qc = $derived.by(() => {
    const p3 = pred.probabilities.Batch_3 ?? 0
    const [lo3, hi3] = pred.interval?.Batch_3 ?? [p3, p3]
    const closest = pred.predicted === 'Batch_3' ? pred.runner_up : pred.predicted
    const reasons: string[] = []
    if (pred.outlier) reasons.push("This sample doesn't look like any batch we've seen before, including the baseline. It could be a new kind of change.")
    for (const d of drivers.filter((d) => d.dir !== 'typical'))
      reasons.push(`${d.label} is ${Math.abs(d.rel).toFixed(0)}% ${d.dir} than the baseline.`)
    if (flags.length) reasons.push('The image has quality problems (see below), so the measurements are less certain.')
    for (const s of segRows.filter((s) => s.level === 'large'))
      reasons.push(`The two segmentation methods disagree on ${s.label.toLowerCase()} by ${s.diff.toFixed(0)}%.`)
    if (!pred.outlier && p3 >= ACCEPT_AT && !flags.length && lo3 >= 0.5) {
      if (!drivers.some((d) => d.dir !== 'typical')) reasons.push('All key measurements are within the normal baseline range.')
      return { kind: 'accept', word: 'Accept', line: `Consistent with the approved baseline: ${pct(p3)} probability it matches Batch 3.`, reasons }
    }
    if (!pred.outlier && p3 <= REJECT_AT && hi3 <= 0.5)
      return { kind: 'reject', word: 'Reject', line: `Changed from the baseline: only ${pct(p3)} probability it matches Batch 3. It most resembles ${batchName(closest)}, a known type of variation.`, reasons }
    if (p3 > REJECT_AT && p3 < ACCEPT_AT) reasons.push(`The probability of matching the baseline (${pct(p3)}) is between the accept (${pct(ACCEPT_AT)}) and reject (${pct(REJECT_AT)}) limits.`)
    if ((p3 >= ACCEPT_AT && lo3 < 0.5) || (p3 <= REJECT_AT && hi3 > 0.5)) reasons.push(`The uncertainty is wide: the baseline probability could plausibly be anywhere from ${pct(lo3)} to ${pct(hi3)}.`)
    return { kind: 'investigate', word: 'Investigate', line: 'Not conclusive. A materials expert should review this sample before a decision is made.', reasons }
  })

  // Segmentation uncertainty for this upload: rule vs DINO masks, and rule boundary reassignment range.
  const segRows = $derived.by(() => {
    const kpis = [...new Set([...overview.classifiers.rule.kpis, ...overview.classifiers.learned.kpis])]
    return kpis.flatMap((k) => {
      const rv = result.kpis.rule?.[k]?.value, dv = result.kpis.learned?.[k]?.value
      const lo = result.kpis.rule?.[k]?.lo, hi = result.kpis.rule?.[k]?.hi
      if (!meta[k] || (rv === undefined && dv === undefined)) return []
      const both = Number.isFinite(rv) && Number.isFinite(dv)
      const diff = both ? (Math.abs(rv! - dv!) / ((Math.abs(rv!) + Math.abs(dv!)) / 2 || 1)) * 100 : NaN
      const base = Number.isFinite(rv) ? rv! : dv!
      const band = Number.isFinite(lo) && Number.isFinite(hi) && base ? (Math.max(Math.abs(hi! - base), Math.abs(base - lo!)) / Math.abs(base)) * 100 : NaN
      const worst = Math.max(Number.isFinite(diff) ? diff : 0, Number.isFinite(band) ? band : 0)
      const level = worst >= 25 ? 'large' : worst >= 10 ? 'moderate' : 'small'
      return [{ k, label: meta[k].label, rv, dv, lo, hi, diff, band, level }]
    })
  })
</script>

<section class="card head">
  <div class="sample">
    <span class="muted">Sample</span> <b>{result.sample_id}</b>
    {#if result.known_batch}
      <div class="muted">Known label: <b>{result.known_batch.replace('_', ' ')}</b>
        {result.known_batch === pred.predicted ? '✓ matches' : '✗ differs'} (training sample, so this is an in-sample check)</div>
    {/if}
  </div>
  <div class="qc qc-{qc.kind}">
    <div class="muted">QC decision vs the approved baseline</div>
    <div class="qcword">{qc.word}</div>
    <div class="qcline">{qc.line}</div>
    {#if qc.reasons.length}<ul class="qcwhy">{#each qc.reasons as r}<li>{r}</li>{/each}</ul>{/if}
  </div>
  <div class="verdict">
    <div class="muted">Closest match</div>
    <div class="big" style="color:{batchColor(pred.predicted)}">{batchName(pred.predicted)}</div>
    <div class="conf"><span class="pctv">{pct(pred.probabilities[pred.predicted])}</span> <span class="pill tier-{pred.tier}">{pred.tier === 'review' ? 'needs review' : `${pred.tier} confidence`}</span></div>
    <div class="muted small policy">Accept if P(baseline) ≥ {pct(ACCEPT_AT)} and its uncertainty range stays above 50%; reject if ≤ {pct(REJECT_AT)} and it stays below 50%; otherwise investigate. These limits are a policy choice for the QC team, not fitted to data.</div>
  </div>
  <div class="flags">
    {#if pred.outlier}<div class="warn"><b>Unlike anything seen before:</b> this sample doesn't closely match the baseline or either known variant, so the "closest match" above is only a rough guide.</div>{/if}
    {#if pred.ambiguous && !pred.outlier}<div class="warn"><b>Ambiguous:</b> no batch reaches 70%.</div>{/if}
    {#each flags as f}{@const m = flagText(f)}<div class="warn"><b>{m.title}:</b> {m.body}</div>{/each}
    {#each result.warnings as w}<div class="warn">{w}</div>{/each}
  </div>
</section>

<section class="card">
  <h2>What's different from the baseline</h2>
  <ul class="drivers">
    {#each drivers as d}
      <li class="drv-{d.dir}">
        <span class="dtag">{d.dir === 'typical' ? 'within baseline' : d.dir === 'higher' ? '▲ higher' : '▼ lower'}</span>
        <div>
          <b>{d.label}</b>
          {#if d.dir === 'typical'}
            is within the normal baseline range ({fmtKpi(d.value, meta[d.kpi])} vs baseline median {fmtKpi(d.median, meta[d.kpi])}).
          {:else}
            is {Math.abs(d.rel).toFixed(0)}% {d.dir} than the baseline median ({fmtKpi(d.value, meta[d.kpi])} vs {fmtKpi(d.median, meta[d.kpi])}; {Math.abs(d.z).toFixed(1)} standard deviations).
          {/if}
          {#if d.why}<div class="muted small">{d.why}</div>{/if}
        </div>
      </li>
    {/each}
  </ul>
  <p class="muted small">Values average the rule and DINO measurements where both exist; measurements are in pixels (the images carry no scale). Within 1 standard deviation of the baseline counts as normal.</p>
</section>

<section class="card">
  <h2>Segmentation uncertainty for this sample</h2>
  <p class="muted small">Each KPI is measured on two independent segmentations of the image (rule-based and DINO). If they disagree, or if moving the phase boundaries by a pixel or two changes the value a lot, that KPI is less certain for this sample.</p>
  <div class="tscroll"><table>
    <thead><tr><th>KPI</th><th class="num">Rule masks</th><th class="num">DINO masks</th><th class="num">Rule vs DINO</th><th class="num">Boundary shift</th><th>Certainty</th></tr></thead>
    <tbody>
      {#each segRows as s}
        <tr>
          <td>{s.label}</td>
          <td class="num">{fmtKpi(s.rv, meta[s.k])}</td>
          <td class="num">{fmtKpi(s.dv, meta[s.k])}</td>
          <td class="num">{Number.isFinite(s.diff) ? `${s.diff.toFixed(0)}% apart` : '–'}</td>
          <td class="num">{Number.isFinite(s.band) ? `±${s.band.toFixed(0)}%` : '–'}{#if Number.isFinite(s.lo) && Number.isFinite(s.hi)}<br /><small class="muted">{fmtKpi(s.lo, meta[s.k])}–{fmtKpi(s.hi, meta[s.k])}</small>{/if}</td>
          <td><span class="seg seg-{s.level}">{s.level === 'small' ? 'reliable' : s.level === 'moderate' ? 'some doubt' : 'uncertain'}</span></td>
        </tr>
      {/each}
    </tbody>
  </table></div>
  <p class="muted small">Reliable: both checks within 10%. Some doubt: 10–25%. Uncertain: over 25%.</p>
</section>

<div class="cols">
  <section class="card">
    <h2>Probabilities</h2>
    <ProbBars {classes} probabilities={pred.probabilities} interval={pred.interval} />
    <p class="muted small">± is the uncertainty: how far each probability could move with a different set of reference samples (shown by the black whiskers).</p>
    <h3>Key measurements</h3>
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
          {#if r.img}
            <figure class="kimg">
              <a href={r.img.src} target="_blank" rel="noreferrer"><img src={r.img.src} alt={r.img.caption} loading="lazy" /></a>
              <figcaption class="muted small">{r.img.caption}</figcaption>
            </figure>
          {:else if r.noImg}
            <p class="muted small">No image for this run: it was classified before images were added (or by a backend that hasn't been restarted). Re-run the sample to see it.</p>
          {/if}
        </div>
      {/each}
    </div>
    <p class="muted small">Coloured band: middle 50% of {pred.predicted.replace('_', ' ')} training samples; thin line: its median; dot: this sample.</p>
  </section>

  <div class="colstack">
    <section class="card">
      <h2>KPIs</h2>
      <div class="why">
        {#each whyRows as r}
          <div class="wrow">
            <div class="whead">
              <span class="wlab">{r.label}</span>
              <span class="wval">{r.display}</span>
            </div>
            <div class="wline">
              {#each r.batches as bt, i}
                {#if bt.st}
                  <div class="wband" style="left:{r.pos(bt.st.p25)}%; width:{Math.max(r.pos(bt.st.p75) - r.pos(bt.st.p25), 0.8)}%; background:{batchColor(bt.b)}; top:{2 + i * 10}px"></div>
                  <div class="wmed" style="left:{r.pos(bt.st.median)}%; background:{batchColor(bt.b)}; top:{i * 10}px"></div>
                {/if}
              {/each}
              <div class="wdot" style="left:{r.pos(r.value)}%"></div>
            </div>
            <div class="wfoot">
              <span class="small">
                {#each r.batches as bt, i}{#if bt.st}{i ? ' · ' : ''}<span style="color:{batchColor(bt.b)}">{short(bt.b)} typical {fmtKpi(bt.st.median, meta[r.kpi])}</span>{/if}{/each}
              </span>
              {#if r.best}
                <span class="wfav" style="background:{batchColor(r.best)}1f; color:{batchColor(r.best)}">
                  favours {short(r.best)} · {r.strength}
                  <span class="wweight" title="{Math.exp(r.margin).toFixed(1)}× more likely under {short(r.best)} than {r.runnerUp ? short(r.runnerUp) : ''}"><span style="width:{Math.max(4, r.weight * 100)}%; background:{batchColor(r.best)}"></span></span>
                </span>
              {/if}
            </div>
          </div>
        {/each}
      </div>
      <p class="muted small">Black dot: this sample. Coloured bands: middle 50% of each batch's training samples, tick: its median. "Favours" names the batch this sample's value fits best; the bar shows how strongly (vs the next-best batch), so overlapping bands give a short bar.</p>
    </section>
  <section class="card">
    <h2>Where the sample sits</h2>
    <div class="charts">
      {#each PRESETS as p}
        <Scatter rows={overview.training} source={p.source} x={p.x} y={p.y} {meta} {classes} title={p.title}
          point={{ x: sampleVal(p.source, p.x) ?? NaN, y: sampleVal(p.source, p.y) ?? NaN, label: 'this sample' }} />
      {/each}
    </div>
    <p class="muted small">Shaded regions: the batch that is most likely at each point, judging by the two plotted measurements only.</p>
  </section>
  </div>
</div>

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
      {#each baselineRows as b}
        <tr class:used={showAllBaseline && isUsed(b)}>
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
  <div class="btoggle">
    <button onclick={() => (showAllBaseline = !showAllBaseline)}>
      {showAllBaseline ? 'Show only classifier KPIs' : `Show all ${result.baseline.kpis.length} measurements`}
    </button>
    <span class="muted small">{showAllBaseline ? 'Bold rows are KPIs the classifiers use. ' : ''}z on the log/logit scale; |z| &lt; 1 counts as typical.</span>
  </div>
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
        <tr class:used={showAllFeatures && used.has(k)}>
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
  <div class="btoggle">
    <button onclick={() => (showAllFeatures = !showAllFeatures)}>
      {showAllFeatures ? 'Show only classifier KPIs' : `Show all ${Object.keys(meta).length} measurements`}
    </button>
    <span class="muted small">{showAllFeatures ? 'Bold rows are used by this classifier. ' : ''}{featSource === 'rule' ? 'Small numbers under the sample are the range when pixels at phase boundaries are reassigned.' : ''}</span>
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
  .qc { text-align: center; border-radius: 16px; padding: 18px 20px; }
  .qc-accept { background: #e8f5ec; color: #1a7f37; }
  .qc-investigate { background: #fff4e0; color: #9a5b00; }
  .qc-reject { background: #fdecec; color: #c62828; }
  .qcword { font-size: 2.6rem; font-weight: 800; letter-spacing: 0.02em; line-height: 1.15; }
  .qcline { font-size: 1.05rem; color: #1f2937; margin-top: 4px; }
  .qcwhy { text-align: left; display: inline-block; margin: 10px auto 0; color: #374151; font-size: 0.95rem; }
  .policy { max-width: 640px; margin: 10px auto 0; }
  .seg { font-size: 0.8rem; font-weight: 700; border-radius: 999px; padding: 3px 10px; white-space: nowrap; }
  .seg-small { background: #e8f5ec; color: #1a7f37; }
  .seg-moderate { background: #fff4e0; color: #9a5b00; }
  .seg-large { background: #fdecec; color: #c62828; }
  .drivers { list-style: none; padding: 0; margin: 0; display: grid; gap: 12px; }
  .drivers li { display: grid; grid-template-columns: 130px 1fr; gap: 12px; align-items: start; }
  .dtag { font-size: 0.85rem; font-weight: 700; border-radius: 999px; padding: 4px 10px; text-align: center; background: #eef0f3; color: #4b5563; }
  .drv-higher .dtag, .drv-lower .dtag { background: #fdecec; color: #c62828; }
  .drv-typical .dtag { background: #e8f5ec; color: #1a7f37; }
  @media (max-width: 600px) { .drivers li { grid-template-columns: 1fr; } .qcword { font-size: 2rem; } }
  .big { font-size: 3rem; font-weight: 750; line-height: 1.1; margin: 4px 0 8px; letter-spacing: -0.01em; }
  .conf { display: inline-flex; align-items: center; gap: 10px; }
  .pctv { font-size: 1.6rem; font-weight: 600; }
  .conf .pill { font-size: 0.95rem; padding: 4px 12px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .cols > * { min-width: 0; }
  .colstack { display: grid; gap: 16px; align-content: start; min-width: 0; }
  .colstack .charts { grid-template-columns: repeat(auto-fill, minmax(min(200px, 100%), 1fr)); }
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
  .kimg { grid-column: 1 / -1; margin: 6px 0 0; }
  .kimg img { display: block; width: 100%; border-radius: 12px; border: 1px solid #eceef1; }
  .kimg figcaption { margin-top: 4px; }
  .kstat { font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-left: 4px; }
  .kstat.within { background: #dff3e4; color: #1d6b33; }
  .kstat.above, .kstat.below { background: #fff1d6; color: #8a5a00; }
  .small { font-size: 0.78rem; }
  .why { display: grid; gap: 18px; margin: 12px 0; }
  .wrow { display: grid; gap: 6px; }
  .btoggle { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin-top: 12px; }
  .whead { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
  .wlab { font-weight: 500; }
  .wval { font-weight: 700; font-size: 1.05rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .wline { position: relative; height: 32px; background: linear-gradient(#e3e6ea, #e3e6ea) 0 50% / 100% 2px no-repeat; }
  .wband { position: absolute; height: 7px; border-radius: 999px; opacity: 0.4; }
  .wmed { position: absolute; width: 3px; height: 11px; border-radius: 2px; transform: translateX(-1.5px); }
  .wdot { position: absolute; top: 50%; width: 14px; height: 14px; border-radius: 50%; background: #1d1d1f; border: 2px solid #fff; box-shadow: 0 0 0 1px #1d1d1f; transform: translate(-50%, -50%); }
  .wfoot { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
  .wfav { display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
  .wweight { display: inline-block; width: 40px; height: 6px; border-radius: 999px; background: rgba(0, 0, 0, 0.08); overflow: hidden; }
  .wweight span { display: block; height: 100%; border-radius: 999px; }
  .zaxis { position: relative; height: 12px; background: linear-gradient(#bbb, #bbb) 50% / 1px 100% no-repeat; }
  .zaxis { display: inline-block; width: 90px; vertical-align: middle; margin-right: 6px; }
  .zbar { position: absolute; top: 1px; bottom: 1px; border-radius: 2px; }
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
  @media (max-width: 600px) { .big { font-size: 2.5rem; } }
  @media (max-width: 1000px) { .cols { grid-template-columns: 1fr; } .vis.zoomed { grid-template-columns: 1fr 1fr; } }
</style>
