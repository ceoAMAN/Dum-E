# pyre-unsafe
"""Dataset streaming and tokenization pipeline."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

from datasets import load_dataset
from transformers import AutoTokenizer

import configs


_TEXT_KEYS = (
    "text",
    "content",
    "code",
    "prompt",
    "response",
    "instruction",
)


@dataclass
class Sample:
    source: str
    text: str
    raw: Dict[str, Any]


_DATASET_CACHE: Dict[str, Any] = {}
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _get_tokenizer(model_id: str):
    if model_id in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_id]
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _TOKENIZER_CACHE[model_id] = tok
    return tok


def _extract_text(example: Dict[str, Any]) -> str:
    for key in _TEXT_KEYS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
    if parts:
        return "\n".join(parts)
    return str(example)


def _load_stream(dataset_key: str):
    if dataset_key in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_key]
    dataset_id, dataset_cfg = configs.DATASET_IDS[dataset_key]
    kwargs: Dict[str, Any] = {
        "split": "train",
        "streaming": True,
    }
    if dataset_cfg:
        kwargs["name"] = dataset_cfg
    if configs.HF_TOKEN:
        kwargs["token"] = configs.HF_TOKEN
    ds = load_dataset(dataset_id, trust_remote_code=True, **kwargs)
    _DATASET_CACHE[dataset_key] = ds
    return ds


def _weighted_choice(rng: random.Random, weights: Dict[str, float]) -> str:
    r = rng.random()
    cumulative = 0.0
    for key, weight in weights.items():
        cumulative += weight
        if r <= cumulative:
            return key
    return list(weights.keys())[-1]


def iter_dataset_samples(dataset_key: str) -> Iterator[Sample]:
    if configs._env_bool("STURNUS_TINY", False):
        while True:
            yield Sample(source=dataset_key, text="Because quantum entanglement particles state bell... def merge sort list return... french revolution france 1789 republic monarchy... roses flowers conclude logic not... REST api endpoint get post user... entropy information probability shannon bits... gravity spacetime curve einstein mass... 2 hours meet distance... therefore first second in conclusion for example. This is high quality data.", raw={"text": "Dummy text"})

    ds = _load_stream(dataset_key)
    for row in ds:
        text = _extract_text(row)
        yield Sample(source=dataset_key, text=text, raw=row)


def iter_mixture_samples(seed: int = 42) -> Iterator[Sample]:
    rng = random.Random(seed)
    streams = {key: iter_dataset_samples(key) for key in configs.DATASET_WEIGHTS}
    while True:
        chosen = _weighted_choice(rng, configs.DATASET_WEIGHTS)
        try:
            yield next(streams[chosen])
        except StopIteration:
            streams[chosen] = iter_dataset_samples(chosen)
            yield next(streams[chosen])


def tokenize_texts(texts: List[str], model_id: str, max_length: int = configs.MAX_SEQ_LEN) -> Dict[str, Any]:
    tok = _get_tokenizer(model_id)
    return tok(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def iter_group_batches(
    group_name: str,
    batch_size: int = 4,
    max_length: int = configs.MAX_SEQ_LEN,
) -> Iterator[Dict[str, Any]]:
    group_to_dataset = {
        "A": "fineweb",
        "B": "fineweb",
        "C": "arxiv",
        "D": "arxiv",
        "E": "code_search_net",
        "F": "code_search_net",
        "G": "dolma",
        "H": "dolma",
        "I": "math",
        "J": "gsm8k",
        "K": "openhermes",
        "L": "openhermes",
        "M": "c4",
        "N": "c4",
        "O": "medqa",
        "P": "medqa",
    }
    dataset_key = group_to_dataset[group_name]
    stream = iter_dataset_samples(dataset_key)
    while True:
        texts = []
        for _ in range(batch_size):
            texts.append(next(stream).text)
        yield tokenize_texts(texts, configs.EXPERT_MODEL_ID, max_length=max_length)


def iter_token_batches_from_samples(
    samples: Iterator[Sample],
    model_id: str,
    batch_size: int,
    max_length: int,
) -> Iterator[Dict[str, Any]]:
    while True:
        texts = []
        for _ in range(batch_size):
            texts.append(next(samples).text)
        yield tokenize_texts(texts, model_id, max_length=max_length)


def iter_mixture_token_batches(
    model_id: str,
    batch_size: int = 4,
    max_length: int = configs.MAX_SEQ_LEN,
    seed: int = 42,
) -> Iterator[Dict[str, Any]]:
    return iter_token_batches_from_samples(
        iter_mixture_samples(seed=seed),
        model_id=model_id,
        batch_size=batch_size,
        max_length=max_length,
    )


def iter_group_token_batches(
    group_name: str,
    model_id: str,
    batch_size: int = 4,
    max_length: int = configs.MAX_SEQ_LEN,
) -> Iterator[Dict[str, Any]]:
    group_to_dataset = {
        "A": "fineweb",
        "B": "fineweb",
        "C": "arxiv",
        "D": "arxiv",
        "E": "code_search_net",
        "F": "code_search_net",
        "G": "dolma",
        "H": "dolma",
        "I": "math",
        "J": "gsm8k",
        "K": "openhermes",
        "L": "openhermes",
        "M": "c4",
        "N": "c4",
        "O": "medqa",
        "P": "medqa",
    }
    dataset_key = group_to_dataset[group_name]
    return iter_token_batches_from_samples(
        iter_dataset_samples(dataset_key),
        model_id=model_id,
        batch_size=batch_size,
        max_length=max_length,
    )


def self_test(live: bool = False) -> None:
    print("[data] self-test")
    if not live:
        print("[data] live=False: skipping network dataset fetch.")
        return

    for key in configs.DATASET_WEIGHTS:
        try:
            sample = next(iter_dataset_samples(key))
            print(f"[data] {key}: sample length={len(sample.text)}")
        except Exception as exc:
            print(f"[data] {key}: failed to stream ({exc})")

    try:
        encoded = tokenize_texts(["hello world"], configs.EXPERT_MODEL_ID)
        print(f"[data] tokenizer OK: {list(encoded.keys())}")
    except Exception as exc:
        print(f"[data] tokenizer failed: {exc}")


if __name__ == "__main__":
    live = configs._env_bool("STURNUS_LIVE_DATA", False)
    self_test(live=live)
