"""End-to-end QC assessment: one SEM field (BSE + optional ETD/SE + Inlens) in, a QC answer out.

The answer has four parts:
1. batch assignment (Batch_1 / Batch_2 / Batch_3) with probabilities, from a Bayesian Student-t
   classifier (``neura_uq``) on three decision KPIs;
2. how each decision KPI sits against the approved Batch_3 baseline, in materials terms;
3. a verdict, ACCEPT / INVESTIGATE / REJECT, relative to the baseline;
4. uncertainty: segmentation (KPI ranges over perturbed label maps, propagated into the
   classifier), model (leave-one-out reliability on the reference), out-of-distribution
   typicality and image-quality flags.
"""

from __future__ import annotations

import base64
import html
import io
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config
from .data import BSE_SUFFIX, BSEImage, find_bse_images, load_bse
from .kpis import compute_kpis, kpis_from_labels
from .pipeline import segment_image
from .qc import overlay
from .report import trust_flags
from .segment import GRAPHITE, SI
from .uncertainty import edge_blur_px, perturbed_labels

BASELINE = "Batch_3"
CLASSES = ("Batch_1", "Batch_2", "Batch_3")
REFERENCE_CSV = Path(str(resources.files("anode_qc") / "reference" / "reference_kpis.csv"))


@dataclass(frozen=True)
class KPI:
    name: str
    label: str
    transform: str
    scale: float
    unit: str
    higher: str
    lower: str


DECISION_KPIS = (
    KPI(
        "frac_pore",
        "Porosity",
        "logit",
        100.0,
        "%",
        "more open packing (less densification / calendering)",
        "denser packing (more densification / calendering)",
    ),
    KPI(
        "graphite_crack_density",
        "Graphite crack density",
        "log",
        1.0,
        "px per 1e4 graphite px",
        "more cracked graphite, mainly at pore-facing flake edges",
        "less cracked graphite",
    ),
    KPI(
        "si_cv_w256",
        "Si spread unevenness (CV, 256 px windows)",
        "log",
        1.0,
        "",
        "patchier Si dispersion (Si-rich and Si-poor areas)",
        "more even Si dispersion",
    ),
)
DECISION_NAMES = [k.name for k in DECISION_KPIS]
# Shown for context only, never used for the decision.
CONTEXT_KPIS = {
    "si_fraction_of_solids": ("Si fraction of solids", 100.0, "%", ""),
    "si_ecd_d50": ("Si particle size D50", 1.0, "px", "unreliable when Si contrast is low"),
    "graphite_aspect_ratio_median": ("Graphite aspect ratio (median)", 1.0, "", "depends on segmentation"),
    "cbd_fraction_of_solids": ("Binder / carbon-black fraction of solids", 100.0, "%", "NO GROUND TRUTH"),
}
QC_COLUMNS = [
    "qc_threshold_method",
    "qc_si_peak_n",
    "qc_streak_severity",
    "qc_charging_index",
    "qc_modal_agreement",
]
TIER_HIGH, TIER_LOW, TYPICALITY_ALPHA, OOD_ALPHA = 0.9, 0.6, 0.05, 0.01
Z_OUTSIDE = 2.0


def _neura_uq():
    try:
        from neura_uq import classifier
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "anode-qc assess needs the neura_uq classifier: pip install -e ./bayesian_kpi_classifier"
        ) from e
    return classifier


# ----------------------------------------------------------------------------- measurement


def load_triplet(im: BSEImage, cfg: Config) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    raw = load_bse(im.path, cfg.preprocess.border_px)
    if not cfg.multimodal.enabled:
        return raw, None, None
    etd, inl = (
        load_bse(p, cfg.preprocess.border_px) if p else None
        for p in (im.sibling("etd"), im.sibling("inlens"))
    )
    return raw, etd, inl


def _thumbnail(raw: np.ndarray, labels: np.ndarray, width: int = 900) -> bytes:
    img = Image.fromarray(overlay(raw, labels))
    img = img.resize((width, max(1, round(img.height * width / img.width))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def measure(im: BSEImage, cfg: Config) -> dict:
    """Segment one field and return its KPIs, the decision-KPI values on perturbed label maps and a thumbnail."""
    raw, etd, inl = load_triplet(im, cfg)
    res = segment_image(raw, cfg, etd, inl)
    k, _, _ = compute_kpis(res, cfg.kpis, cfg.multimodal.crack_k)
    etd_n = res.mm.etd_n if res.mm is not None else None
    blur = edge_blur_px(res.n, res.labels == SI, res.labels == GRAPHITE, cfg.uncertainty.edge_profile_px)
    delta = int(np.clip(np.round(blur / 2), 1, 4)) if np.isfinite(blur) else 1
    runs = [[k[c] for c in DECISION_NAMES]]
    for lab in perturbed_labels(res, delta, cfg.uncertainty, cfg.cleanup.si_min_area_px).values():
        kk = kpis_from_labels(lab, res.n, cfg.kpis, etd_n, cfg.multimodal.crack_k)[0]
        runs.append([kk.get(c, np.nan) for c in DECISION_NAMES])
    row = {"image_id": im.image_id, "batch": im.batch, "multimodal": res.mm is not None}
    row.update({c: k.get(c, np.nan) for c in DECISION_NAMES + list(CONTEXT_KPIS) + QC_COLUMNS})
    row["trust_flags"] = trust_flags(pd.DataFrame([row])).iloc[0]
    return {"row": row, "runs": np.asarray(runs, float), "thumbnail": _thumbnail(raw, res.labels)}


def reference_row(m: dict) -> dict:
    """One reference-table row: nominal KPIs plus the segmentation SD of each decision KPI."""
    sd = np.nanstd(m["runs"], axis=0, ddof=1)
    return {**m["row"], **{f"{c}_sd": float(s) for c, s in zip(DECISION_NAMES, sd, strict=True)}}


def collect_images(paths: list[Path]) -> list[BSEImage]:
    images = []
    for p in paths:
        if p.is_dir():
            images += find_bse_images(p)
        elif p.name.endswith(BSE_SUFFIX):
            images.append(BSEImage(image_id=p.name[: -len(BSE_SUFFIX)], batch="unknown", path=p))
        else:
            raise SystemExit(f"{p}: expected a *{BSE_SUFFIX} file or a folder containing them")
    return images


# ----------------------------------------------------------------------------- model


def load_reference(path: Path | None = None) -> pd.DataFrame:
    ref = pd.read_csv(path or REFERENCE_CSV)
    ref["trust_flags"] = ref["trust_flags"].fillna("").astype(str)
    missing = set(DECISION_NAMES) - set(ref.columns)
    if missing:
        raise ValueError(f"reference table is missing decision KPIs: {sorted(missing)}")
    return ref[ref.batch.isin(CLASSES)].reset_index(drop=True)


def _reference_measurements(ref: pd.DataFrame) -> list:
    c = _neura_uq()
    sd_cols = [f"{k}_sd" for k in DECISION_NAMES]
    if not set(sd_cols) <= set(ref.columns):
        return [c.Measurement.from_sd(x) for x in ref[DECISION_NAMES].values]
    return [
        c.Measurement.from_sd(x, s)
        for x, s in zip(ref[DECISION_NAMES].values, ref[sd_cols].values, strict=True)
    ]


def _clf_kwargs(n_draws: int = 2000) -> dict:
    return {
        "kpis": DECISION_NAMES,
        "transforms": [k.transform for k in DECISION_KPIS],
        "covariance": "diag",
        "prior_strength": 2.0,
        "n_draws": n_draws,
    }


def fit_classifier(ref: pd.DataFrame):
    c = _neura_uq()
    return c.BayesianStudentTClassifier(**_clf_kwargs()).fit(
        ref[DECISION_NAMES].values, ref.batch.values, reference_measurements=_reference_measurements(ref)
    )


def tier(p_top: float, typicality: float, flags: str = "") -> str:
    if p_top < TIER_LOW or typicality < TYPICALITY_ALPHA or flags:
        return "review"
    return "high" if p_top >= TIER_HIGH else "medium"


def validate(ref: pd.DataFrame) -> dict:
    """Leave-one-image-out reliability of the fixed classifier on the reference table."""
    c = _neura_uq()
    out = c.leave_one_out(
        ref[DECISION_NAMES].values, ref.batch.values, _reference_measurements(ref), **_clf_kwargs(500)
    )
    y = ref.batch.values
    pred = np.array([p.predicted for p in out["predictions"]])
    tiers = [
        tier(p.probabilities[p.predicted], p.typicality_pvalue[p.predicted], f)
        for p, f in zip(out["predictions"], ref.trust_flags, strict=True)
    ]
    per_tier = {}
    for t in ("high", "medium", "review"):
        m = np.array(tiers) == t
        per_tier[t] = {"n": int(m.sum()), "correct": int((pred[m] == y[m]).sum())}
    recall = {b: float(np.mean(pred[y == b] == b)) for b in CLASSES}
    return {
        "n_images": len(y),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.mean(list(recall.values()))),
        "recall": recall,
        "log_loss": out["log_loss"],
        "tiers": per_tier,
    }


@lru_cache(maxsize=4)
def _model(path: str | None):
    ref = load_reference(Path(path) if path else None)
    return ref, fit_classifier(ref), validate(ref)


# ----------------------------------------------------------------------------- assessment


@dataclass
class Shift:
    kpi: str
    label: str
    value: float
    sd: float
    baseline_median: float
    baseline_p5: float
    baseline_p95: float
    z: float
    percentile_in_baseline: float
    batch_medians: dict
    evidence: float
    status: str
    meaning: str
    unit: str
    scale: float


@dataclass
class Assessment:
    image_id: str
    batch: str
    probabilities: dict
    confidence: str
    verdict: str
    verdict_reason: str
    typicality: dict
    out_of_distribution: bool
    segmentation_stable: bool
    calls_under_segmentation_variants: list
    p_range_under_segmentation_variants: list
    shifts: list
    context: list
    trust_flags: str
    explanation: list
    validation: dict
    multimodal: bool = True
    thumbnail_png: bytes | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("thumbnail_png")
        return d


def baseline_shifts(x: dict, sd: np.ndarray, ref: pd.DataFrame, evidence: dict) -> list[Shift]:
    tf = _neura_uq().transform
    b3 = ref[ref.batch == BASELINE]
    out = []
    for i, k in enumerate(DECISION_KPIS):
        vals = b3[k.name].values.astype(float)
        t = tf(vals[:, None], [k.transform])[:, 0]
        z = float((tf(np.array([[x[k.name]]]), [k.transform])[0, 0] - t.mean()) / t.std(ddof=1))
        p5, p95 = np.percentile(vals, [5, 95])
        if abs(z) < Z_OUTSIDE and p5 <= x[k.name] <= p95:
            status, meaning = "within baseline", "typical of the approved baseline"
        elif abs(z) < Z_OUTSIDE:
            status = "edge of baseline"
            side = k.higher if x[k.name] > p95 else k.lower
            meaning = f"beyond the baseline 5–95 % range but within ±{Z_OUTSIDE:g} SD; leaning towards {side}"
        else:
            status = "above baseline" if z > 0 else "below baseline"
            meaning = k.higher if z > 0 else k.lower
        out.append(
            Shift(
                kpi=k.name,
                label=k.label,
                value=float(x[k.name]),
                sd=float(sd[i]),
                baseline_median=float(np.median(vals)),
                baseline_p5=float(p5),
                baseline_p95=float(p95),
                z=z,
                percentile_in_baseline=float(100 * np.mean(vals < x[k.name])),
                batch_medians={b: float(ref.loc[ref.batch == b, k.name].median()) for b in CLASSES},
                evidence=float(evidence.get(k.name, 0.0)),
                status=status,
                meaning=meaning,
                unit=k.unit,
                scale=k.scale,
            )
        )
    return out


def decide(p_base: float, typ_base: float, shifts: list[Shift], flags: str) -> tuple[str, str]:
    outside = [s.label for s in shifts if abs(s.z) >= Z_OUTSIDE]
    if typ_base < OOD_ALPHA or (p_base < 0.2 and (typ_base < TYPICALITY_ALPHA or outside)):
        why = f"differs from the {BASELINE} baseline (P({BASELINE}) = {p_base:.2f}, baseline typicality p = {typ_base:.3f}"
        why += f"; outside the baseline range: {', '.join(outside)})" if outside else ")"
        return "REJECT", why
    if p_base >= TIER_LOW and typ_base >= TYPICALITY_ALPHA and not outside:
        if flags:
            return (
                "INVESTIGATE",
                f"matches the baseline, but image-quality flags were raised ({flags}); re-image to confirm",
            )
        return (
            "ACCEPT",
            f"consistent with the {BASELINE} baseline on every decision KPI (P({BASELINE}) = {p_base:.2f})",
        )
    parts = [f"P({BASELINE}) = {p_base:.2f}"]
    if outside:
        parts.append(f"outside the baseline range: {', '.join(outside)}")
    if flags:
        parts.append(f"image-quality flags: {flags}")
    return "INVESTIGATE", "evidence is mixed (" + "; ".join(parts) + ")"


def _fmt(v: float, k_scale: float, unit: str) -> str:
    return f"{v * k_scale:.3g}{' ' + unit if unit == '%' else ''}" if np.isfinite(v) else "n/a"


def explain(a: Assessment) -> list[str]:
    p = a.probabilities[a.batch]
    lines = [
        f"Assigned to {a.batch} (P = {p:.2f}, {a.confidence} confidence). Verdict {a.verdict}: {a.verdict_reason}."
    ]
    for s in sorted(a.shifts, key=lambda s: -abs(s.z)):
        rng = f"{_fmt(s.baseline_p5, s.scale, s.unit)}–{_fmt(s.baseline_p95, s.scale, s.unit)}"
        lines.append(
            f"{s.label}: {_fmt(s.value, s.scale, s.unit)} vs baseline {_fmt(s.baseline_median, s.scale, s.unit)} "
            f"(baseline 5–95 %: {rng}; z = {s.z:+.1f}) — {s.meaning}."
        )
    ev = sorted(a.shifts, key=lambda s: -abs(s.evidence))
    second = sorted(a.probabilities, key=a.probabilities.get)[-2]
    lines.append(
        f"Strongest evidence for {a.batch} over {second}: "
        + ", ".join(f"{s.label} ({s.evidence:+.2f})" for s in ev)
        + " (log-evidence; + favours the assigned batch)."
    )
    if a.out_of_distribution:
        lines.append(
            "This sample is atypical for every known batch: possibly a new batch. The assignment above is the nearest known batch."
        )
    if not a.segmentation_stable:
        lines.append(
            "The call changes under plausible segmentation variants: "
            + ", ".join(a.calls_under_segmentation_variants)
            + "."
        )
    if a.trust_flags:
        lines.append(f"Image-quality flags: {a.trust_flags}. Treat Si-based numbers with caution.")
    return lines


def assess(m: dict, reference: Path | None = None) -> Assessment:
    c = _neura_uq()
    ref, clf, val = _model(str(reference) if reference else None)
    row, runs = m["row"], m["runs"]
    ok = np.isfinite(runs).all(axis=1)
    meas = c.Measurement.from_runs(runs[ok]) if ok.sum() > 1 else c.Measurement.from_sd(runs[0])
    meas = c.Measurement(x=runs[0], cov=meas.cov, runs=None)
    pred = clf.predict(meas)
    variants = [clf.predict(c.Measurement.from_sd(r)) for r in runs[ok]]
    calls = [v.predicted for v in variants]
    p_top = [v.probabilities[pred.predicted] for v in variants]
    flags = row.get("trust_flags", "") or ""
    sd = np.sqrt(np.diag(meas.cov)) if meas.cov is not None else np.zeros(len(DECISION_NAMES))
    shifts = baseline_shifts(row, sd, ref, pred.kpi_evidence)
    verdict, reason = decide(pred.probabilities[BASELINE], pred.typicality_pvalue[BASELINE], shifts, flags)
    context = []
    for name, (label, scale, unit, note) in CONTEXT_KPIS.items():
        if name in row and name in ref:
            context.append(
                {
                    "kpi": name,
                    "label": label,
                    "value": float(row[name]) * scale,
                    "baseline_median": float(ref.loc[ref.batch == BASELINE, name].median()) * scale,
                    "batch_medians": {
                        b: float(ref.loc[ref.batch == b, name].median()) * scale for b in CLASSES
                    },
                    "unit": unit,
                    "note": note,
                }
            )
    a = Assessment(
        image_id=row["image_id"],
        batch=pred.predicted,
        probabilities={b: float(pred.probabilities[b]) for b in CLASSES},
        confidence=tier(pred.probabilities[pred.predicted], pred.typicality_pvalue[pred.predicted], flags),
        verdict=verdict,
        verdict_reason=reason,
        typicality={b: float(pred.typicality_pvalue[b]) for b in CLASSES},
        out_of_distribution=bool(max(pred.typicality_pvalue.values()) < OOD_ALPHA),
        segmentation_stable=len(set(calls)) == 1 and calls[0] == pred.predicted,
        calls_under_segmentation_variants=calls,
        p_range_under_segmentation_variants=[float(min(p_top)), float(max(p_top))],
        shifts=shifts,
        context=context,
        trust_flags=flags,
        explanation=[],
        validation=val,
        multimodal=bool(row.get("multimodal", True)),
        thumbnail_png=m.get("thumbnail"),
    )
    a.explanation = explain(a)
    return a


# ----------------------------------------------------------------------------- report

_VERDICT_COLOUR = {"ACCEPT": "#1b7f3b", "INVESTIGATE": "#b26a00", "REJECT": "#b3261e"}


def render_html(a: Assessment) -> str:
    e = html.escape
    colour = _VERDICT_COLOUR[a.verdict]
    bars = "".join(
        f'<div class="bar"><span>{b}</span><div style="width:{100 * p:.0f}%;background:{"#555" if b != a.batch else colour}"></div>'
        f"<b>{p:.2f}</b></div>"
        for b, p in a.probabilities.items()
    )
    rows = "".join(
        f"<tr><td>{e(s.label)}</td><td>{_fmt(s.value, s.scale, s.unit)} ± {_fmt(s.sd, s.scale, s.unit)}</td>"
        f"<td>{_fmt(s.baseline_median, s.scale, s.unit)} [{_fmt(s.baseline_p5, s.scale, s.unit)}–{_fmt(s.baseline_p95, s.scale, s.unit)}]</td>"
        f"<td>{s.z:+.1f}</td><td>{e(s.status)}</td>"
        + "".join(f"<td>{_fmt(s.batch_medians[b], s.scale, s.unit)}</td>" for b in CLASSES)
        + f"<td>{s.evidence:+.2f}</td><td>{e(s.meaning)}</td></tr>"
        for s in a.shifts
    )
    ctx = "".join(
        f"<tr><td>{e(c['label'])}</td><td>{c['value']:.3g}</td>"
        + "".join(f"<td>{c['batch_medians'][b]:.3g}</td>" for b in CLASSES)
        + f"<td>{e(c['note'])}</td></tr>"
        for c in a.context
    )
    v = a.validation
    tiers = ", ".join(f"{t}: {d['correct']}/{d['n']}" for t, d in v["tiers"].items())
    img = (
        f'<img src="data:image/png;base64,{base64.b64encode(a.thumbnail_png).decode()}" alt="segmentation overlay">'
        '<p class="small">Overlay: orange = Si, blue = pore, magenta = gap/crack-like voids, green = binder (no ground truth); grey = graphite.</p>'
        if a.thumbnail_png
        else ""
    )
    expl = "".join(f"<li>{e(x)}</li>" for x in a.explanation)
    unc = [
        f"Segmentation: decision KPIs were recomputed on 4 perturbed label maps (confident / inclusive / shrunk / grown); "
        f"calls: {', '.join(a.calls_under_segmentation_variants)}; P({a.batch}) range "
        f"{a.p_range_under_segmentation_variants[0]:.2f}–{a.p_range_under_segmentation_variants[1]:.2f}. "
        "The segmentation SD is folded into the probabilities.",
        "Typicality (is the sample a plausible member of each batch?): "
        + ", ".join(f"{b} p = {p:.3f}" for b, p in a.typicality.items())
        + (" — atypical for all known batches (possible new batch)." if a.out_of_distribution else "."),
        f"Model reliability (leave-one-image-out on {v['n_images']} reference images): accuracy {v['accuracy']:.0%}, "
        f"balanced accuracy {v['balanced_accuracy']:.0%}; correct by confidence tier: {tiers}. "
        "The three KPIs were chosen after exploring the same images, so true accuracy on new batches is likely lower.",
    ]
    if not a.multimodal:
        unc.append("Only BSE was available: pores and cracks are less reliable without ETD/SE.")
    unc_html = "".join(f"<li>{e(x)}</li>" for x in unc)
    head = "".join(f"<th>{b} median</th>" for b in CLASSES)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>QC {e(a.image_id)}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;color:#222;padding:0 12px}}
.verdict{{background:{colour};color:#fff;padding:14px 18px;border-radius:8px;font-size:20px}}
.verdict small{{display:block;font-size:14px;opacity:.9;margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}td,th{{border:1px solid #ddd;padding:5px;text-align:left}}
th{{background:#f4f4f4}}.bar{{display:flex;align-items:center;gap:8px;margin:3px 0}}.bar span{{width:70px}}
.bar div{{height:14px;border-radius:3px}}img{{width:100%;border:1px solid #ccc}}.small{{font-size:12px;color:#666}}
</style></head><body>
<h2>Electrode QC: {e(a.image_id)}</h2>
<div class="verdict">{a.verdict} — closest batch {a.batch} ({a.confidence} confidence)<small>{e(a.verdict_reason)}</small></div>
<h3>Batch probabilities</h3>{bars}
<h3>Why</h3><ul>{expl}</ul>
<h3>Decision KPIs vs the {BASELINE} baseline</h3>
<table><tr><th>KPI</th><th>Sample ± segmentation SD</th><th>Baseline median [5–95 %]</th><th>z vs baseline</th><th>Status</th>{head}<th>Evidence</th><th>Materials meaning</th></tr>{rows}</table>
<h3>Uncertainty</h3><ul>{unc_html}</ul>
<h3>Context KPIs (not used for the decision)</h3>
<table><tr><th>KPI</th><th>Sample</th>{head}<th>Note</th></tr>{ctx}</table>
<h3>Segmentation</h3>{img}
<p class="small">Verdict rules: ACCEPT if P({BASELINE}) ≥ {TIER_LOW}, baseline typicality p ≥ {TYPICALITY_ALPHA}, every decision KPI within ±{Z_OUTSIDE:g} SD of the baseline and no image-quality flags; REJECT if baseline typicality p &lt; {OOD_ALPHA}, or P({BASELINE}) &lt; 0.2 with an atypical baseline fit or a KPI outside the baseline range; otherwise INVESTIGATE. Batches 1 and 2 are not necessarily defective: REJECT means the material does not match what the supplier promised.</p>
</body></html>"""


def write_outputs(a: Assessment, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.image_id}.json").write_text(json.dumps(a.to_dict(), indent=2, default=float))
    path = out_dir / f"{a.image_id}.html"
    path.write_text(render_html(a))
    return path


def summary_row(a: Assessment) -> dict:
    return {
        "image_id": a.image_id,
        "batch": a.batch,
        **{f"P_{b}": round(p, 3) for b, p in a.probabilities.items()},
        "confidence": a.confidence,
        "verdict": a.verdict,
        "out_of_distribution": a.out_of_distribution,
        "segmentation_stable": a.segmentation_stable,
        "trust_flags": a.trust_flags,
        "reason": a.verdict_reason,
    }
