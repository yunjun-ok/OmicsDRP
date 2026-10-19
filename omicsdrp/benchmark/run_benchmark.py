"""Orchestrate the competitor benchmark: every model x split x regime.

Each adapter needs a different conda env, so every job runs as its own
``conda run`` subprocess. A crash in one job never takes the sweep down — the
failure is recorded and the next job starts. Jobs are fold-level resumable
(adapters skip folds whose outputs already exist), so re-running after an
interruption picks up where it stopped.

Progress lands in ``BenchmarkResults/progress.{md,json}`` and, if a recipient is
configured, an email goes out per job or once at the end.

Usage:
    python run_benchmark.py --smoke
    python run_benchmark.py --email-to you@example.com
    python run_benchmark.py --models deeptta,graphdrp --splits mixed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import common  # noqa: E402

HERE = Path(__file__).resolve().parent

# model -> (conda env, adapter script, extra CLI args)
MODELS = {
    "deeptta": ("benchmark_deeptta", "adapters/deeptta_adapter.py", []),
    "graphdrp": ("benchmark_graphdrp", "adapters/graphdrp_adapter.py", []),
    "graphdrp_gcn": ("benchmark_graphdrp", "adapters/graphdrp_adapter.py", ["--variant", "gcn"]),
    "graphdrp_gat": ("benchmark_graphdrp", "adapters/graphdrp_adapter.py", ["--variant", "gat"]),
    "graphdrp_gat_gcn": ("benchmark_graphdrp", "adapters/graphdrp_adapter.py", ["--variant", "gat_gcn"]),
    "paccmann": ("benchmark_paccmann", "adapters/paccmann_adapter.py", []),
}


def _notify(subject: str, body: str, to):
    if not to:
        return
    try:
        from omicsdrp.notify import send_email
        send_email(subject, body, to)
    except Exception as e:  # a mail problem must never stop training
        print(f"[email] unavailable: {e}")


def _jobs(models, splits, regimes):
    for m in models:
        for regime in regimes:
            for s in splits:
                # `ensemble` produces deployable weights; only the mixed split
                # is meaningful for it, matching inference_models.py.
                if regime == "ensemble" and s != "mixed":
                    continue
                yield m, s, regime


def _write_progress(out_root: Path, records):
    (out_root / "progress.json").write_text(json.dumps(records, indent=2))
    lines = ["# Benchmark progress", "",
             "| model | split | regime | status | seconds |",
             "|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['model']} | {r['split']} | {r['regime']} | "
                     f"{r['status']} | {r.get('seconds', '')} |")
    (out_root / "progress.md").write_text("\n".join(lines) + "\n")


def _summarise(out_root: Path, model, split, regime):
    """Average the per-fold metrics of a finished job, for the email body."""
    d = out_root / model / f"{regime}__{split}"
    metrics = [json.loads(p.read_text()) for p in sorted(d.glob("fold_*/metrics.json"))]
    if not metrics:
        return "no folds"
    keys = ("rmse", "pearson", "spearman", "r2", "n_params_total")
    parts = []
    for k in keys:
        vals = [m[k] for m in metrics if k in m]
        if vals:
            parts.append(f"{k}={sum(vals) / len(vals):.4f}")
    return f"{len(metrics)} folds | " + " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--splits", default=",".join(common.SPLIT_MODES))
    ap.add_argument("--regimes", default=",".join(common.REGIMES))
    ap.add_argument("--out-root", default=str(HERE / "BenchmarkResults"))
    ap.add_argument("--email-to", default=None)
    ap.add_argument("--email-per", choices=["job", "sweep", "none"], default="job")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(models) - set(MODELS)
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")

    for split in args.splits.split(","):
        if not (common.EXPORT_DIR / f"{split.strip()}.npz").exists():
            raise SystemExit(
                f"missing fold export for '{split.strip()}' — run "
                "`conda run -n omicsdrp python export_folds.py` first.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    jobs = list(_jobs(models,
                      [s.strip() for s in args.splits.split(",") if s.strip()],
                      [r.strip() for r in args.regimes.split(",") if r.strip()]))
    email_to = args.email_to
    records = []
    t_sweep = time.time()

    print(f"[sweep] {len(jobs)} jobs -> {out_root}", flush=True)
    _notify("[benchmark] sweep started",
            f"{len(jobs)} jobs\n" + "\n".join(f"{m} {s} {r}" for m, s, r in jobs),
            email_to if args.email_per != "none" else None)

    for i, (model, split, regime) in enumerate(jobs, 1):
        env, script, extra_args = MODELS[model]
        cmd = ["conda", "run", "--no-capture-output", "-n", env,
               "python", script,
               "--split", split, "--regime", regime,
               "--out-root", str(out_root), "--device", args.device] + extra_args
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.smoke:
            cmd.append("--smoke")

        print(f"\n[{i}/{len(jobs)}] {model} {regime}/{split} ({env})", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=HERE)
        dt = round(time.time() - t0, 1)

        rec = {"model": model, "split": split, "regime": regime,
               "status": "ok" if rc == 0 else f"FAILED(rc={rc})", "seconds": dt}
        if rc == 0:
            rec["summary"] = _summarise(out_root, model, split, regime)
        records.append(rec)
        _write_progress(out_root, records)

        if args.email_per == "job":
            _notify(f"[benchmark] {model} {regime}/{split} {rec['status']}",
                    json.dumps(rec, indent=2), email_to)

    total = round(time.time() - t_sweep, 1)
    failed = [r for r in records if r["status"] != "ok"]
    body = "\n".join(f"{r['model']:9s} {r['regime']:9s} {r['split']:12s} "
                     f"{r['status']:14s} {r.get('summary', '')}" for r in records)
    body += f"\n\ntotal {total}s | {len(failed)} failed"
    print("\n=== sweep complete ===\n" + body, flush=True)

    if args.email_per != "none":
        _notify(f"[benchmark] sweep complete ({len(failed)} failed)", body, email_to)

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
