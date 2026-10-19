"""Staged Stage-1 execution pipeline with live progress + email notifications.

Order (as requested):
    0. BASELINE   -- nested-CV baseline: ECFP2/Morgan drug + Attention cell + all omics
    1. FEATURE    -- omics ablation by noise substitution (all 15 subsets)
    2. ATTENTION  -- cell encoder: attention vs. MLP
    3. DRUG       -- drug representation: Morgan + 4 frozen pretrained

The baseline model recurs inside every later stage (feature=all, cell=attention,
drug=morgan are all the baseline). Because ``ExperimentConfig.tag()`` is the
experiment identity (independent of the stage label), those recurrences collapse
to the SAME tag and are DE-DUPLICATED here -> the baseline is trained once and
reused as each stage's reference row.

Progress is streamed to stdout (a progress bar + per-experiment lines) and to
``<out_root>/progress.md`` / ``progress.json`` so a detached run can be watched
with ``cat progress.md``. An email is sent after each stage (or each experiment).
"""
from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Dict, List, Optional

from .config import ExperimentConfig, OMICS_ORDER
from .ablations import reference_config, FEATURE_NOISE_SETS, ABLATION_DRUG_ENCODERS
from .experiment import run_experiment
from .notify import send_email


STAGE_ORDER = ["0_baseline", "1_feature", "2_attention", "3_drug"]


def _omics_label(omics) -> str:
    return "+".join(o for o in OMICS_ORDER if o in omics)


def config_label(c: ExperimentConfig) -> str:
    # use "·" (not "|") so the label is safe inside a markdown table cell
    noise = _omics_label(c.noise_omics) or "none"
    return f"omics={_omics_label(c.omics)} · noise={noise} · cell={c.cell_encoder} · drug={c.drug_encoder} · split={c.split_mode}"


def build_stage1_stages(**overrides):
    """Return (reference_config, OrderedDict[stage -> [unique configs]], baseline_tag).

    De-duplicates by tag across stages so the baseline is scheduled once.
    """
    ref = reference_config(name="s1", **overrides)  # all omics, attention, morgan, mixed
    raw_stages: "OrderedDict[str, List[ExperimentConfig]]" = OrderedDict()
    raw_stages["0_baseline"] = [ref]
    raw_stages["1_feature"] = [replace(ref, noise_omics=n) for n in FEATURE_NOISE_SETS]
    raw_stages["2_attention"] = [replace(ref, cell_encoder=e) for e in ("attention", "mlp")]
    raw_stages["3_drug"] = [replace(ref, drug_encoder=d) for d in ABLATION_DRUG_ENCODERS]

    seen = set()
    stages: "OrderedDict[str, List[ExperimentConfig]]" = OrderedDict()
    for sname, cfgs in raw_stages.items():
        uniq = []
        for c in cfgs:
            t = c.tag()
            if t in seen:
                continue
            seen.add(t)
            uniq.append(c)
        stages[sname] = uniq
    return ref, stages, ref.tag()


# --------------------------------------------------------------------------- #
# progress rendering
# --------------------------------------------------------------------------- #
def _bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / max(total, 1))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def _fmt_dur(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _metric(summary: dict, key: str):
    v = summary.get(key)
    return f"{v:.4f}" if isinstance(v, (int, float)) else "-"


def _write_progress(out_root: str, rows: List[dict], total: int, t0: float,
                    baseline_tag: str) -> None:
    done = sum(1 for r in rows if r["status"] in ("done", "cached"))
    elapsed = time.time() - t0
    ran = [r for r in rows if r["status"] == "done" and r["dur"] > 0]
    avg = sum(r["dur"] for r in ran) / len(ran) if ran else 0.0
    remaining = total - done
    eta = avg * remaining if avg else 0.0

    lines = [
        "# OmicsDRP Stage-1 sweep progress", "",
        f"`{_bar(done, total)}`  elapsed {_fmt_dur(elapsed)}"
        + (f", ETA ~{_fmt_dur(eta)}" if eta else ""), "",
        "| # | stage | config | status | RMSE | R2 | Pearson | folds | dur |",
        "|---|-------|--------|--------|------|----|---------|-------|-----|",
    ]
    for i, r in enumerate(rows, 1):
        s = r["summary"]
        star = " *(baseline)*" if r["tag"] == baseline_tag else ""
        rmse = (f"{_metric(s,'rmse_mean')}±{_metric(s,'rmse_std')}"
                if s.get("rmse_mean") is not None else "-")
        lines.append(
            f"| {i} | {r['stage']} | {r['label']}{star} | {r['status']} | "
            f"{rmse} | {_metric(s,'r2_mean')} | {_metric(s,'pearson_mean')} | "
            f"{s.get('n_folds','-')} | {_fmt_dur(r['dur']) if r['dur'] else '-'} |")
    with open(os.path.join(out_root, "progress.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out_root, "progress.json"), "w") as f:
        json.dump({"done": done, "total": total, "elapsed_s": elapsed,
                   "eta_s": eta, "rows": [{k: v for k, v in r.items() if k != "summary"} | r["summary"]
                                          for r in rows]}, f, indent=2, default=str)


def _stage_email_body(stage: str, rows: List[dict], baseline_tag: str) -> str:
    out = [f"Stage {stage} complete.", ""]
    for r in rows:
        s = r["summary"]
        star = " (baseline)" if r["tag"] == baseline_tag else ""
        out.append(f"- {r['label']}{star}: status={r['status']} "
                   f"RMSE={_metric(s,'rmse_mean')} R2={_metric(s,'r2_mean')} "
                   f"Pearson={_metric(s,'pearson_mean')} [{_fmt_dur(r['dur'])}]")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_pipeline(stages: "OrderedDict[str, List[ExperimentConfig]]",
                 baseline_tag: str, raw, out_root: str,
                 email_to: Optional[str] = None, email_per: str = "stage",
                 device: Optional[str] = None) -> List[dict]:
    os.makedirs(out_root, exist_ok=True)
    total = sum(len(v) for v in stages.values())
    rows: List[dict] = []
    t0 = time.time()
    done = 0

    plan = "\n".join(f"  {sname}: {len(cfgs)} experiment(s)" for sname, cfgs in stages.items())
    print(f"\n=== Stage-1 pipeline: {total} unique experiments ===\n{plan}\n")
    send_email(f"[OmicsDRP] Stage-1 sweep started ({total} experiments)",
               f"Plan (baseline trained once, reused):\n{plan}\n\nout_root={out_root}",
               email_to)

    for sname, cfgs in stages.items():
        print(f"\n----- STAGE {sname}  ({len(cfgs)} experiment(s)) -----")
        stage_rows: List[dict] = []
        for c in cfgs:
            done += 1
            label = config_label(c)
            print(f"\n{_bar(done - 1, total)}  ▶ [{sname}] {label}")
            et0 = time.time()
            summary = run_experiment(c, raw=raw, device=device)
            dur = time.time() - et0 if summary.get("status") != "cached" else 0.0
            row = {"stage": sname, "tag": c.tag(), "label": label,
                   "status": summary.get("status", "?"), "dur": dur, "summary": summary}
            rows.append(row)
            stage_rows.append(row)
            print(f"    {summary.get('status')}  RMSE={_metric(summary,'rmse_mean')} "
                  f"R2={_metric(summary,'r2_mean')} Pearson={_metric(summary,'pearson_mean')} "
                  f"[{_fmt_dur(dur) if dur else 'cached'}]")
            _write_progress(out_root, rows, total, t0, baseline_tag)
            if email_per == "experiment":
                send_email(f"[OmicsDRP] done: {label}",
                           _stage_email_body(sname, [row], baseline_tag), email_to,
                           attachments=[os.path.join(out_root, "progress.md")])

        if email_per == "stage":
            send_email(f"[OmicsDRP] stage {sname} done ({done}/{total})",
                       _stage_email_body(sname, stage_rows, baseline_tag), email_to,
                       attachments=[os.path.join(out_root, "progress.md")])

    print(f"\n=== pipeline complete: {done}/{total} in {_fmt_dur(time.time()-t0)} ===")
    send_email(f"[OmicsDRP] Stage-1 sweep COMPLETE ({total} experiments)",
               f"All done in {_fmt_dur(time.time()-t0)}.\nSee {out_root}/progress.md",
               email_to, attachments=[os.path.join(out_root, "progress.md")])
    return rows
