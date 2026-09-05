"""Samples and the dataset stream. Carried over from the audited data.py —
extract_pair is verified 8/8 datasets, 20/20 rows each. Only the dead tokenise
helpers and the calibration stream were dropped."""
from __future__ import annotations

import os
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional

from . import config as C


@dataclass
class Sample:
    source: str
    prompt: str
    answer: str
    verifiable: str = ""          # a STRING: the checkable final value, or "" — never a bool
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.source}::{self.prompt}"


def extract_pair(example: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """Split one dataset row into {prompt, answer, verifiable} against the REAL
    schema of each configured dataset. Returns None on no match."""
    def _s(key: str, default: str = "") -> str:
        v = example.get(key, default)
        return v.strip() if isinstance(v, str) else default

    if "completion" in example and "prompt" in example:                       # CodeAlpaca
        p, a = _s("prompt"), _s("completion")
        if p and a:
            return {"prompt": p, "answer": a, "verifiable": ""}
    if "instruction" in example and "output" in example:                      # alpaca
        instr, inp, out = _s("instruction"), _s("input") or _s("context"), _s("output")
        if instr and out:
            return {"prompt": f"{instr}\n{inp}".strip(), "answer": out, "verifiable": ""}
    ch = example.get("choices")                                                # ai2_arc
    if "answerKey" in example and isinstance(ch, dict) and "text" in ch and "label" in ch:
        q, key = _s("question"), _s("answerKey")
        texts, labels = list(ch["text"]), [str(l) for l in ch["label"]]
        if q and texts and key in labels:
            correct = texts[labels.index(key)]
            rendered = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            return {"prompt": f"{q}\n{rendered}", "answer": correct, "verifiable": correct}
    if "correct_answer" in example and "question" in example:                 # sciq
        q, correct = _s("question"), _s("correct_answer")
        if q and correct:
            return {"prompt": q, "answer": correct, "verifiable": correct}
    if "question" in example and "answer" in example:                         # gsm8k
        q, a = _s("question"), _s("answer")
        if q and a:
            final = a.split("####")[-1].strip() if "####" in a else ""
            return {"prompt": q, "answer": a, "verifiable": final}
    if "query" in example and "response" in example:                          # MetaMathQA
        q, a = _s("query"), _s("response")
        if q and a:
            marker = "The answer is:"
            final = a.split(marker)[-1].strip() if marker in a else ""
            return {"prompt": q, "answer": a, "verifiable": final}
    for field_, role_key, text_key, assistant in (                             # chat formats
        ("conversations", "from", "value", ("gpt", "assistant")),
        ("messages", "role", "content", ("assistant", "gpt")),
    ):
        conv = example.get(field_)
        if not isinstance(conv, list) or not conv:
            continue
        turns = [(str(m.get(role_key, "")), str(m.get(text_key, ""))) for m in conv if isinstance(m, dict)]
        last = next((i for i in range(len(turns) - 1, -1, -1)
                     if turns[i][0].lower() in assistant and turns[i][1].strip()), None)
        if last is None or last == 0:
            continue
        prompt = "\n".join(f"{r}: {t.strip()}" for r, t in turns[:last] if t.strip())
        if prompt and turns[last][1].strip():
            return {"prompt": prompt, "answer": turns[last][1].strip(), "verifiable": ""}
    return None


# ── streams ─────────────────────────────────────────────────────────────────
_CACHE: Dict[str, Any] = {}


def authenticate():
    token = C.HF_TOKEN or os.environ.get("HF_TOKEN", "")
    if token:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)


def _open(key: str):
    if key in _CACHE:
        return _CACHE[key]
    from datasets import load_dataset
    ds_id, cfg, split = C.DATASETS[key]
    kwargs: Dict[str, Any] = {"split": split, "streaming": True}
    if cfg:
        kwargs["name"] = cfg
    if C.HF_TOKEN:
        kwargs["token"] = C.HF_TOKEN
    print(f"[data] opening {key} ({ds_id})")
    ds = load_dataset(ds_id, **kwargs)
    _CACHE[key] = ds
    return ds


def iter_dataset(key: str, consumed: Optional[Dict[str, int]] = None) -> Iterator[Sample]:
    """`consumed[key]` counts RAW rows (what ds.skip() skips), including rows
    extract_pair rejects — so a resumed stream lands exactly where it stopped."""
    consumed = consumed if consumed is not None else {}
    ds = _open(key)
    skip = int(consumed.get(key, 0))
    if skip > 0:
        ds = ds.skip(skip)               # resume where the last process stopped
    for row in ds:
        consumed[key] = consumed.get(key, 0) + 1
        pair = extract_pair(row)
        if not pair:
            continue
        yield Sample(source=key, prompt=pair["prompt"], answer=pair["answer"],
                     verifiable=pair.get("verifiable") or "", raw=row)


def _next_with_timeout(stream: Iterator[Sample], key: str, timeout: Optional[float]) -> Sample:
    result: Dict[str, Any] = {}
    done = threading.Event()

    def runner():
        try:
            result["s"] = next(stream)
        except BaseException as e:      # noqa: BLE001 — propagated below
            result["e"] = e
        finally:
            done.set()

    threading.Thread(target=runner, daemon=True).start()
    if not done.wait(timeout if timeout is not None else C.DATASET_SAMPLE_TIMEOUT):
        raise TimeoutError(f"{key} timed out")
    if "e" in result:
        raise result["e"]
    return result["s"]


def iter_mixture(seed: int = 42, consumed: Optional[Dict[str, int]] = None) -> Iterator[Sample]:
    """Weighted round-robin over every configured dataset; every yielded Sample
    has a prompt AND an answer, so nothing downstream ever lacks ground truth.

    `consumed` (per-dataset rows already yielded) is MUTATED as rows are served
    and persisted by the caller, so a restarted process resumes instead of
    replaying the same rows from the top of every stream."""
    consumed = consumed if consumed is not None else {}
    rng = random.Random(seed + sum(consumed.values()))
    streams: Dict[str, Iterator[Sample]] = {}
    cold, failed = set(), set()
    for key, w in C.DATASET_WEIGHTS.items():
        if w <= 0:
            continue
        try:
            streams[key] = iter_dataset(key, consumed)
            cold.add(key)
        except Exception as e:      # noqa: BLE001
            print(f"[data] failed to open {key}: {e}")
            failed.add(key)
    while True:
        weights = {k: v for k, v in C.DATASET_WEIGHTS.items() if k not in failed and v > 0}
        if not weights:
            return
        total = sum(weights.values())
        r, acc, chosen = rng.random() * total, 0.0, next(iter(weights))
        for k, v in weights.items():
            acc += v
            if r <= acc:
                chosen = k
                break
        try:
            s = _next_with_timeout(streams[chosen], chosen, C.DATASET_BOOT_TIMEOUT if chosen in cold else None)
            cold.discard(chosen)
            yield s
        except StopIteration:
            try:
                consumed[chosen] = 0                  # wrapped around: start the dataset over
                streams[chosen] = iter_dataset(chosen, consumed)
                cold.add(chosen)
            except Exception:       # noqa: BLE001
                failed.add(chosen)
        except Exception as e:      # noqa: BLE001
            print(f"[data] {chosen}: {e}")
            failed.add(chosen)
