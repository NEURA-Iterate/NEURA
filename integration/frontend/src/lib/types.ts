export type Batch = string
export type Source = 'rule' | 'learned'
export type ByBatch<T> = Record<Batch, T>

export interface KpiMeta { label: string; unit: string; transform: string }
export interface Summary {
  correct: number; n: number; image_correct: number; log_loss: number
  acc_Batch_1: number; acc_Batch_2: number; acc_Batch_3: number
}
export interface PerSample {
  image_id: string; true: Batch; P_Batch_1: number; P_Batch_2: number; P_Batch_3: number
  n_rounds: number; trust_flags: string | null
}
export interface Figure { name: string; title: string; url: string }
export interface Images {
  bse: string; rule_overlay: string; learned_overlay: string; cracks: string
  zoom_bse: string; zoom_rule: string; zoom_learned: string; zoom_cracks: string
}
export type Legend = Record<string, string | { label: string; color: string }> | { name: string; color: string }[]
export interface Representative {
  image_id: string; images: Images; kpis: Record<Source, Record<string, number>>
}
export interface TrainingRow {
  image_id: string; batch: Batch; trust_flags: string | null
  rule: Record<string, number>; learned: Record<string, number>
}
export interface Stat { median: number; p25: number; p75: number }
export interface Overview {
  classes: Batch[]
  n_per_batch: ByBatch<number>
  kpi_meta: Record<string, KpiMeta>
  classifiers: Record<Source, { kpis: string[] }>
  validation: {
    summary: Record<'rule' | 'learned' | 'combined', Summary>
    per_sample: PerSample[]
    confusion: Record<Batch, Record<Batch, number>>
    tiers: { tier: string; n: number; accuracy: number }[]
    note: string
  }
  training: TrainingRow[]
  batch_stats: Record<Source, Record<string, ByBatch<Stat>>>
  figures: Figure[]
  representatives: ByBatch<Representative>
  legend: Legend
}
export interface Contribution {
  source: Source; kpi: string; label: string; value: number; display: string
  per_batch: ByBatch<number>; favours: Batch; evidence: number
}
export interface BaselineKpi {
  source: Source; kpi: string; label: string; value: number; median: number
  p05: number; p95: number; z: number | null; percentile: number | null; direction: 'higher' | 'lower' | 'typical'
}
export interface ClassifierResult {
  kpis: Record<string, number | null>; probabilities: ByBatch<number>; interval: ByBatch<[number, number]>
  typicality: ByBatch<number>; evidence: Record<string, number>
}
export interface Result {
  sample_id: string; known_batch: Batch | null; timings: Record<string, number>
  warnings: string[]; trust_flags: string | string[] | null
  prediction: {
    predicted: Batch; runner_up: Batch; probabilities: ByBatch<number>
    interval: ByBatch<[number, number]>; tier: 'high' | 'medium' | 'review'
    ambiguous: boolean; outlier: boolean
  }
  classifiers: Record<Source, ClassifierResult>
  contributions: Contribution[]
  explanation: string[]
  kpis: { rule: Record<string, { value: number; lo?: number; hi?: number }>; learned: Record<string, { value: number }> }
  baseline: {
    batch: Batch; typicality: Record<Source, number>; consistent_with_baseline: boolean
    kpis: BaselineKpi[]
  }
  images: Images
  legend: Legend
}
export interface Job {
  status: 'queued' | 'running' | 'done' | 'error'
  step: string; progress: number; result?: Result; error?: string
  sample_id?: string; created_at?: string
}

export interface HistoryItem {
  job_id: string; sample_id: string; known_batch: string | null; source: 'upload' | 'demo'
  created_at: string; status: Job['status']; error: string | null
  predicted: string | null; probabilities: Record<string, number> | null
  interval: Record<string, [number, number]> | null
  tier: string | null; ambiguous: boolean | null; outlier: boolean | null; trust_flags: string | null
}
