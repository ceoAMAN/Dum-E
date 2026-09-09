"""Persistence contract. One store, one writer (System.save), versioned.

  clock      monotonic token count — the time base every age is measured against,
             persisted so it never restarts at 0 (the old token_count bug)
  geometry   stamped with the gate hash that produced it; a mismatch at load is
             LOUD and forces re-formation — never a silent carry-over
  standing / reliability / chains
             cold = zero observations on the record, never a stored flag

on_missing: cold start with a printed line. on_corrupt: refuse the blob, print,
cold start — never crash boot, never half-load.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

from . import config as C

VERSION = 7   # v7: training sweeps the pool (curriculum); the router is deployment-only


def path() -> Path:
    return Path(C.STATE_DIR) / "state.pkl"


def save(blob: Dict[str, Any]) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"version": VERSION, **blob}, f)
    tmp.replace(p)


def load() -> Optional[Dict[str, Any]]:
    p = path()
    if not p.exists():
        print(f"[state] no state at {p} — cold start")
        return None
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        if not isinstance(blob, dict) or blob.get("version") != VERSION:
            print(f"[state] {p} has version {getattr(blob, 'get', lambda k: None)('version')} != {VERSION} — cold start")
            return None
        return blob
    except Exception as e:      # noqa: BLE001
        print(f"[state] could not read {p}: {e} — cold start")
        return None
