"""
Sturnus — Dataset Validation Script
Tests every dataset in configs.DATASET_IDS for:
  1. Can we open the stream?
  2. Can we pull the first 3 samples?
  3. Does _extract_text produce non-empty text?
  4. Can the gate tokenizer encode the text?
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import configs
from data import authenticate_huggingface, _extract_text, _load_stream, get_tokenizer

SAMPLE_COUNT = 3
TIMEOUT = 60  # seconds per dataset


def validate_one(key: str) -> dict:
    spec = configs.DATASET_IDS[key]
    dataset_id = spec[0]
    dataset_cfg = spec[1] if len(spec) >= 2 else None
    dataset_split = spec[2] if len(spec) >= 3 else "train"
    result = {
        "key": key,
        "dataset_id": dataset_id,
        "config": dataset_cfg,
        "split": dataset_split,
        "status": "unknown",
        "samples": 0,
        "errors": [],
        "sample_lengths": [],
        "elapsed": 0.0,
    }
    t0 = time.time()
    try:
        ds = _load_stream(key)
        result["status"] = "stream_ok"
    except Exception as e:
        result["status"] = "STREAM_FAIL"
        result["errors"].append(f"Failed to open stream: {e}")
        result["elapsed"] = time.time() - t0
        return result

    tokenizer = get_tokenizer(configs.GATE_MODEL_ID)
    count = 0
    try:
        for row in ds:
            if count >= SAMPLE_COUNT:
                break
            text = _extract_text(row)
            if not text or not text.strip():
                result["errors"].append(f"Sample {count}: _extract_text returned empty")
                count += 1
                continue
            ids = tokenizer.encode(text[:configs.MAX_SEQ_LEN * 6])[:configs.MAX_SEQ_LEN]
            if len(ids) < configs.FRAGMENT_MIN:
                result["errors"].append(
                    f"Sample {count}: tokenized to {len(ids)} tokens (< FRAGMENT_MIN={configs.FRAGMENT_MIN})"
                )
            result["sample_lengths"].append(len(ids))
            count += 1

            if time.time() - t0 > TIMEOUT:
                result["errors"].append(f"Timed out after {TIMEOUT}s")
                break
    except StopIteration:
        result["errors"].append(f"Stream exhausted after {count} samples (expected {SAMPLE_COUNT})")
    except Exception as e:
        result["errors"].append(f"Error reading samples: {e}")

    result["samples"] = count
    result["elapsed"] = time.time() - t0
    if count >= SAMPLE_COUNT and not result["errors"]:
        result["status"] = "OK"
    elif count > 0 and not result["errors"]:
        result["status"] = "PARTIAL"
    elif result["errors"]:
        result["status"] = "ERRORS"
    return result


def main():
    configs.validate_config()
    authenticate_huggingface()

    print("=" * 72)
    print("  STURNUS — Dataset Validation")
    print("=" * 72)
    print(f"  Datasets: {len(configs.DATASET_IDS)}")
    print(f"  Samples per dataset: {SAMPLE_COUNT}")
    print("=" * 72)
    print()

    results = []
    for key in configs.DATASET_IDS:
        weight = configs.DATASET_WEIGHTS.get(key, 0.0)
        print(f"[{key}] (weight={weight:.3f}) ... ", end="", flush=True)
        r = validate_one(key)
        results.append(r)
        status_emoji = "✅" if r["status"] == "OK" else "⚠️" if r["status"] in ("PARTIAL", "ERRORS") else "❌"
        print(
            f"{status_emoji} {r['status']} | "
            f"samples={r['samples']}/{SAMPLE_COUNT} | "
            f"lengths={r['sample_lengths']} | "
            f"elapsed={r['elapsed']:.1f}s"
        )
        if r["errors"]:
            for err in r["errors"]:
                print(f"    ⚠ {err}")
        print()

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    ok = [r for r in results if r["status"] == "OK"]
    partial = [r for r in results if r["status"] == "PARTIAL"]
    errors = [r for r in results if r["status"] == "ERRORS"]
    failed = [r for r in results if r["status"] == "STREAM_FAIL"]
    print(f"  ✅ OK:           {len(ok)}")
    print(f"  ⚠️  Partial:      {len(partial)}")
    print(f"  ⚠️  Errors:       {len(errors)}")
    print(f"  ❌ Stream Fail:  {len(failed)}")
    print()
    if failed:
        print("  FAILED DATASETS:")
        for r in failed:
            print(f"    - {r['key']} ({r['dataset_id']})")
            for err in r["errors"]:
                print(f"        {err}")
    if errors:
        print("  DATASETS WITH ERRORS:")
        for r in errors:
            print(f"    - {r['key']} ({r['dataset_id']})")
            for err in r["errors"]:
                print(f"        {err}")
    print("=" * 72)

    import os
    # Return non-zero if any critical failures
    if failed:
        os._exit(1)
    else:
        os._exit(0)


if __name__ == "__main__":
    main()
