"""Exhaustive experiment logging.

Design goal (user requirement): *nothing* produced by any evaluation step is
lost -- every per-epoch train/val loss and metric, every outer-fold test
metric, and every prediction is streamed to disk as it is produced.

Directory layout for one experiment (``root/<tag>/``)::

    config.json                 # exact ExperimentConfig
    events.jsonl                # append-only stream of EVERY logged event
    history.csv                 # tidy per-epoch train/val curve (all metrics)
    folds/
        fold_{k}_predictions.parquet   # per-sample test predictions
        fold_{k}_test_metrics.json
        fold_{k}_embeddings.pth        # optional latent embeddings
    summary.json                # aggregated mean/std across outer folds

``events.jsonl`` is the source of truth: it is flushed after every write, so a
crash mid-run still leaves a complete record of everything seen so far. The CSV
and summary files are conveniences derived from the same events.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class ExperimentRecorder:
    def __init__(self, root: str, tag: str, config: Dict[str, Any]):
        self.dir = os.path.join(root, tag)
        self.folds_dir = os.path.join(self.dir, "folds")
        os.makedirs(self.folds_dir, exist_ok=True)

        self.tag = tag
        self._events_path = os.path.join(self.dir, "events.jsonl")
        self._history: List[Dict[str, Any]] = []
        self._fold_test: List[Dict[str, Any]] = []

        # Fresh event stream per (re)run of this exact tag.
        self._events_fh = open(self._events_path, "a", buffering=1)  # line-buffered

        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=_to_jsonable)
        self.event("experiment_start", config=config)

    # -- low level: append one event and fsync-flush ----------------------
    def event(self, kind: str, **payload: Any) -> None:
        rec = {"t": time.time(), "kind": kind, **payload}
        self._events_fh.write(json.dumps(rec, default=_to_jsonable) + "\n")
        self._events_fh.flush()
        os.fsync(self._events_fh.fileno())

    # -- per-epoch training / validation curve ----------------------------
    def log_epoch(self, fold: int, epoch: int, split: str,
                  loss: float, metrics: Dict[str, float]) -> None:
        """Record one epoch's result for one split (train / val).

        Called every epoch for every split so the entire loss/metric trajectory
        is captured, not just the early-stopped best.
        """
        row = {"fold": fold, "epoch": epoch, "split": split,
               "loss": float(loss), **{k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                                        for k, v in metrics.items()}}
        self._history.append(row)
        self.event("epoch", **row)

    # -- outer-fold test evaluation (unbiased) ----------------------------
    def log_fold_test(self, fold: int, metrics: Dict[str, float],
                      best_epoch: Optional[int] = None) -> None:
        rec = {"fold": fold, "best_epoch": best_epoch, **metrics}
        self._fold_test.append(rec)
        with open(os.path.join(self.folds_dir, f"fold_{fold}_test_metrics.json"), "w") as f:
            json.dump(rec, f, indent=2, default=_to_jsonable)
        self.event("fold_test", **rec)

    def completed_folds(self) -> set:
        """Folds already finished on disk (test-metrics json present) -> skip on
        resume. Makes a long sweep restartable at fold granularity after a crash
        or power-off."""
        done = set()
        if os.path.isdir(self.folds_dir):
            for name in os.listdir(self.folds_dir):
                if name.startswith("fold_") and name.endswith("_test_metrics.json"):
                    try:
                        done.add(int(name.split("_")[1]))
                    except (IndexError, ValueError):
                        pass
        return done

    def save_predictions(self, fold: int, df: pd.DataFrame) -> None:
        path = os.path.join(self.folds_dir, f"fold_{fold}_predictions.parquet")
        df.to_parquet(path)  # needs pyarrow; no silent .pkl fallback
        self.event("predictions_saved", fold=fold, path=path, n=len(df))

    def save_embeddings(self, fold: int, embeddings: Dict[str, Any]) -> None:
        import torch
        path = os.path.join(self.folds_dir, f"fold_{fold}_embeddings.pth")
        torch.save(embeddings, path)
        self.event("embeddings_saved", fold=fold, path=path)

    # -- finalisation (rebuilt from the append-only event log) ------------
    def _read_events(self) -> list:
        rows = []
        if os.path.isfile(self._events_path):
            with open(self._events_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # tolerate a torn last line from a hard crash
        return rows

    def finalize(self) -> Dict[str, Any]:
        """Rebuild history.csv + summary.json from events.jsonl so the outputs are
        correct even after several resumed sessions (the event log is the
        append-only source of truth; duplicates from a re-run fold are de-duped)."""
        events = self._read_events()

        epochs = [e for e in events if e.get("kind") == "epoch"]
        if epochs:
            hist = pd.DataFrame(epochs).drop(columns=["kind", "t"], errors="ignore")
            hist = hist.drop_duplicates(subset=["fold", "epoch", "split"], keep="last")
            hist = hist.sort_values(["fold", "epoch", "split"])
            hist.to_csv(os.path.join(self.dir, "history.csv"), index=False)

        fold_tests = [e for e in events if e.get("kind") == "fold_test"]
        summary: Dict[str, Any] = {"tag": self.tag}
        if fold_tests:
            test_df = pd.DataFrame(fold_tests).drop(columns=["kind", "t"], errors="ignore")
            test_df = test_df.drop_duplicates(subset=["fold"], keep="last").sort_values("fold")
            summary["n_folds"] = int(len(test_df))
            metric_cols = [c for c in test_df.columns if c not in ("fold", "best_epoch")]
            for c in metric_cols:
                if pd.api.types.is_numeric_dtype(test_df[c]):
                    summary[f"{c}_mean"] = float(test_df[c].mean())
                    summary[f"{c}_std"] = float(test_df[c].std(ddof=0))
            test_df.to_csv(os.path.join(self.dir, "fold_test_metrics.csv"), index=False)
        else:
            summary["n_folds"] = 0

        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=_to_jsonable)
        self.event("experiment_end", summary=summary)
        self._events_fh.close()
        return summary


def is_experiment_complete(root: str, tag: str, outer_folds: int) -> bool:
    """True when all outer folds have a test-metrics json on disk -> skip on
    resume. (A finished experiment also has summary.json, but fold files are the
    authoritative per-fold completion markers.)"""
    folds_dir = os.path.join(root, tag, "folds")
    if not os.path.isdir(folds_dir):
        return False
    return all(
        os.path.isfile(os.path.join(folds_dir, f"fold_{k}_test_metrics.json"))
        for k in range(1, outer_folds + 1)
    )
