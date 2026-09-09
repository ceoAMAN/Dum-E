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
# Gate and Central share one tokenizer (verified: 151,665 shared ids, identical
# encodings, both embedding tables 151,936). Target ids go through the gate as-is.

E = 100                      # expert pool size. A VARIABLE: everything below is algebra over it.

LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.05
LR = 2e-5
GRAD_CLIP = 1.0

# ── tokens ──────────────────────────────────────────────────────────────────
EXPERT_GEN_TOKENS = 32       # tokens an expert writes about its span; also the standing divisor floor
TARGET_MAX_TOKENS = 128      # y is truncated ONCE to min(this, limit//4) before any pass
# The delta is a MEAN over M target tokens and was the one statistic in the system
# with no minimum support (cf. MIN_MEMBERS, RELIABILITY_MIN_OBS). Measured over 400
# rows of the live mixture: M<=1 is 3.0%, M<=2 is 10.5%, M<=4 is 18.0%, M<=8 is
# 21.8% — almost all of it sciq and ai2_arc, whose answers are one word. Those rows
# produced every outlier delta in the b0-b31 run (+6.75, +6.69, +2.63, +2.29).
# VALUE NEEDS AMAN: 4 refuses 18% of the corpus, 8 refuses 22%, 16 refuses 29%.
TARGET_MIN_TOKENS = 4
SAMPLE_TEMP = 0.8            # CEILING on the second candidate's temperature, not the value.
                             # The value is temp = SAMPLE_TEMP * (1 - rho): when Central is
                             # reliable on this composition the experts should ACCEPT it and
                             # stop exploring; when it is not, exploration is all they have.
                             # A bound is safe here; a fixed 0.8 would have been a constant
                             # closing an adaptive loop, which is the failure mode of record.
WORKING_PROBE_TOKENS = 1024  # question length the working-memory reserve is measured at
EXPERT_PROBE_TOKENS = 256    # expert peak is measured at this. RAISING THIS CRASHED THE
                             # MACHINE: the probe runs with Central resident and a longer
                             # prefill through a 1.5B expert exceeds the Metal working set.

# ── sqrt brackets (DECIDED: c = experts) ────────────────────────────────────
GENERAL_EXPERTS = math.ceil(math.sqrt(E))                                     # 10
_NON_GENERAL = E - GENERAL_EXPERTS                                            # 90
CENTROID_EXPERTS = math.floor(math.sqrt(_NON_GENERAL)) if _NON_GENERAL > 0 else 0   # 9
MAX_CLUSTERS = (_NON_GENERAL // CENTROID_EXPERTS) if CENTROID_EXPERTS else 1        # 10



# ── similarity bands (DECIDED with Aman: 0 errors on 190 pairs) ─────────────
SIM_MEMBER, SIM_NEIGHBOUR, SIM_FAR = 0.90, 0.70, 0.40

# ── geometry formation ──────────────────────────────────────────────────────
TAU_PERCENTILE = 10                      # tau_c = this percentile of the members' similarity
MIN_MEMBERS = 100 // TAU_PERCENTILE      # a percentile is an order statistic only with this many points

# ── chains ──────────────────────────────────────────────────────────────────
CHAIN_MEMORY = 500.0         # "remembers 500 transitions"
CHAIN_PRIOR = 1.0
TAU_STEP = 0.005
TAU_MAX = 0.97
LOAD_WINDOW_PER_CLUSTER = 10 # presence is estimated over this many batches PER cluster (window = 10*C)

# ── reliability (held out by construction) ──────────────────────────────────
HELDOUT_MOD = 4              # hash(sample) % 4 == 0 feeds reliability; the rest feed standing
RELIABILITY_MIN_OBS = 100    # below this a cluster pools to the global estimate
R_MIN = 0.05                 # admission: mean reliability over the target; exp(-3) = 3 nats mean CE

# ── standing ────────────────────────────────────────────────────────────────
STANDING_A_WINDOW = 5 * HELDOUT_MOD   # canary A: P(no admitted batch in 20) = 0.25^20 if y arrives

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
