"""Constants only.

Rule from the audit: nothing in this file may close an adaptive loop. Every
threshold that gates a decision is either DECIDED (evidence named beside it) or
derived at formation / calibration time and stored WITH the data it was
calibrated on. If you find yourself adding a number here that a running loop
compares against, stop — that number belongs in the store the loop reads.
"""
from __future__ import annotations

import math
import os

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# ── models (fixed; only the protocol between them changes) ──────────────────
GATE_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
EXPERT_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CENTRAL_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
GATE_D, EXPERT_D, CENTRAL_D = 896, 1536, 2560

E = 100                      # expert pool size. A VARIABLE: everything below is algebra over it.

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
LR = 2e-5
GRAD_CLIP = 1.0

# ── tokens ──────────────────────────────────────────────────────────────────
SPAN_MIN = 32                # smallest contiguous span an expert is handed
EXPERT_GEN_TOKENS = 32       # tokens an expert writes about its span
TARGET_MAX_TOKENS = 128      # y is truncated ONCE to min(this, limit//4) before any pass
SAMPLE_TEMP = 0.8            # temperature for the second candidate in self-imitation
G_CANDIDATES = 2

# ── sqrt brackets (DECIDED: c = experts) ────────────────────────────────────
GENERAL_EXPERTS = math.ceil(math.sqrt(E))                                     # 10
_NON_GENERAL = E - GENERAL_EXPERTS                                            # 90
CENTROID_EXPERTS = math.floor(math.sqrt(_NON_GENERAL)) if _NON_GENERAL > 0 else 0   # 9
MAX_CLUSTERS = (_NON_GENERAL // CENTROID_EXPERTS) if CENTROID_EXPERTS else 1        # 10


def k_upper(n_clusters: int) -> int:
    """The sqrt(C) admissibility band, upper edge. C a perfect square -> sqrt+1;
    otherwise ceil(sqrt). At C=10 -> 4."""
    r = math.sqrt(max(1, n_clusters))
    return int(r) + 1 if float(int(r)) == r else int(math.ceil(r))


# ── similarity bands (DECIDED with Aman: 0 errors on 190 pairs) ─────────────
SIM_MEMBER, SIM_NEIGHBOUR, SIM_FAR = 0.90, 0.70, 0.40
# Aman's 10/20/30/40 closeness tiers as allocation weights. Home dominates.
TIER_WEIGHT = {"member": 1.0, "neighbour": 0.5, "close": 0.25, "far": 0.0}

# ── chains ──────────────────────────────────────────────────────────────────
CHAIN_MEMORY = 500.0         # "remembers 500 transitions"
CHAIN_PRIOR = 1.0
CHAIN_SEED_STRENGTH = 8.0
TAU_STEP = 0.005
TAU_MAX = 0.97

# ── reliability (held out by construction) ──────────────────────────────────
HELDOUT_MOD = 4              # hash(sample) % 4 == 0 feeds reliability; the rest feed standing
RELIABILITY_MIN_OBS = 100    # below this a cluster pools to the global estimate
R_MIN = 0.05                 # admission: mean reliability over the target; exp(-3) = 3 nats mean CE

# ── standing ────────────────────────────────────────────────────────────────
STANDING_Z = 1.0             # one standard error discounted

# ── cadence ─────────────────────────────────────────────────────────────────
HEALTH_EVERY = 5
MIGRATE_EVERY = 20
SAVE_EVERY = 25

# ── paths ───────────────────────────────────────────────────────────────────
STATE_DIR = "state/dume"
CHECKPOINT_DIR = "state/dume/ckpt"
LEGACY_CENTRAL_CKPT = "state/checkpoints/central/weights.safetensors"   # grounded-CE trained; safe to inherit

# ── data ────────────────────────────────────────────────────────────────────
DATASET_BOOT_TIMEOUT = 60
DATASET_SAMPLE_TIMEOUT = 60
DATASETS = {
    "github_code":         ("HuggingFaceH4/CodeAlpaca_20K", None, "train"),
    "python_instructions": ("iamtarun/python_code_instructions_18k_alpaca", None, "train"),
    "gsm8k":               ("openai/gsm8k", "main", "train"),
    "metamath":            ("meta-math/MetaMathQA", None, "train"),
    "ai2_arc":             ("allenai/ai2_arc", "ARC-Challenge", "train"),
    "camel_science":       ("sciq", None, "train"),
    "slimorca":            ("Open-Orca/SlimOrca", None, "train"),
    "ultrachat":           ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
}
DATASET_WEIGHTS = {k: 1.0 / len(DATASETS) for k in DATASETS}
