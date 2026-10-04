import type { KpiMeta, Legend } from './types'

export const BATCH_COLORS: Record<string, string> = {
  Batch_1: '#4c72b0',
  Batch_2: '#dd8452',
  Batch_3: '#55a868',
}
export const batchColor = (b: string) => BATCH_COLORS[b] ?? '#888'
export const short = (b: string) => b.replace('Batch_', 'B')
export const pct = (p: number, d = 0) => {
  const v = p * 100
  if (d === 0 && v > 99 && v < 100) return `${v.toFixed(1)}%`
  if (d === 0 && v > 0 && v < 1) return '<1%'
  return `${v.toFixed(d)}%`
}

export function fmtKpi(value: number | null | undefined, meta?: KpiMeta): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '–'
  if (meta?.unit === '%') return `${(value * 100).toFixed(1)}%`
  const a = Math.abs(value)
  return a >= 100 ? value.toFixed(0) : a >= 10 ? value.toFixed(1) : value.toFixed(2)
}

export function legendItems(legend: Legend | undefined): { name: string; color: string }[] {
  if (!legend) return []
  if (Array.isArray(legend)) return legend
  return Object.entries(legend).map(([name, v]) =>
    typeof v === 'string' ? { name, color: v } : { name: (v as { label?: string }).label ?? name, color: (v as { color: string }).color },
  )
}

export const SOURCE_LABEL: Record<string, string> = { rule: 'Rule masks', learned: 'DINO masks' }

const FLAG_TEXT: Record<string, { title: string; body: string }> = {
  low_si_contrast: { title: 'Low silicon contrast', body: 'silicon is hard to tell apart from its surroundings in the BSE image, so silicon measurements may be less reliable.' },
  polishing_streaks: { title: 'Polishing streaks', body: 'surface scratches are visible and may be mistaken for cracks or pores.' },
  charging: { title: 'Charging artefacts', body: 'bright patches from beam charging may distort the segmentation.' },
  detector_disagreement: { title: 'Detectors disagree', body: 'the BSE, SE and Inlens images do not line up well. Check they show the same location at the same magnification.' },
}
export const flagText = (f: string) =>
  FLAG_TEXT[f] ?? { title: 'Image-quality flag', body: f.replace(/_/g, ' ') }
