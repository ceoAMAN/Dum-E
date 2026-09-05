from __future__ import annotations
import math
import os
from pathlib import Path
def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env.local"
    if not env_path.exists():
        return
    comment_prefix = chr(35)
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line[:1] == comment_prefix or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
_load_local_env()
# Optional: only needed if the model checkpoints you point Dum-E at are gated on
# the HuggingFace Hub. The public mlx-community defaults below need no token.
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
DEPLOYMENT = os.getenv("DUME_DEPLOYMENT", "False").lower() in ("true", "1", "yes")
GATE_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
EXPERT_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CENTRAL_MODEL_ID = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
EXPERT_POOL_SIZE = 100
NUM_EXPERTS = EXPERT_POOL_SIZE
# Fallback RAM-per-expert estimate. Measured for real at boot via
# splitter.measure_expert_ram_mb() (loads one expert, reads the delta) and the
# measured value replaces this at runtime, so X/Y geometry uses the true cost on
# whatever hardware Dum-E runs on rather than a hardcoded guess.
# Cold-start floor estimate of per-expert RAM, used only until the live memory
# governor (diagnostics) measures the REAL marginal cost (weights + 7B-forward +
# generation spike) and takes over. No hard expert cap exists — concurrency is
# derived per batch from measured memory, thermal, and processing load.
EXPERT_RAM_MB = 850
# Cold-start estimate; splitter.measure_gate_ram_mb() replaces it at boot with the
# real load delta, exactly as EXPERT_RAM_MB is measured.
GATE_RAM_MB = 350
CENTRAL_RAM_MB = 2400
MIN_BOOT_RAM_MB = 4500
GATE_D_MODEL = 896
EXPERT_D_MODEL = 1536
# Central is the SYNTHESISER, not the knowledge store — the experts hold the
# knowledge. Sizing it as a composer (4B) rather than a knower (was 7B Mistral)
# frees ~1.5GB, and picking it from the SAME Qwen family as the gate/experts
# unifies the tokenizer across the whole stack (vocab 151936 everywhere) — the
# gate/expert/Central hidden states are finally same-lineage representations
# instead of cross-family ones. Still > EXPERT_D_MODEL, so the capacity
# hierarchy the architecture claims (small experts -> larger synthesiser) holds.
CENTRAL_D_MODEL = 2560
FRAGMENT_MIN = 32
OVERLAP_FRACTION = 0.175
K_MIN = 0
K_MAX = 6        # highest experts-per-token the gate may request
K_DEFAULT = 4
# Input-tokenisation safety ceiling ONLY — the longest token sequence we ever
# read in for a single sample. It is NOT the expert operating point: how many
# tokens each expert actually processes/generates is governed per-expert by the
# Apex-Nadir Convolution (R_out), bootstrapped from EXPERT_BOOTSTRAP_TOKENS until
# the curves have data. Keep generous; apex-nadir decides the real working size.
MAX_SEQ_LEN = 512   # Restored from the 128 emergency setting: that was forced by the
                    # 7B Central's activation spike on 16GB. With a 4B synthesiser
                    # (~1.5GB freed) the spike is far smaller. If Metal OOM ever
                    # returns, THIS is the first dial to turn back down.
# Cold-start expert fragment/generation size, used ONLY before the convolution has
# enough latency/quality data to produce an R_out for an expert (compute_r_out
# returns None until then). Once R_out exists it governs and this is ignored —
# the "first run gathers context, apex-nadir takes over" handshake.
EXPERT_BOOTSTRAP_TOKENS = 64
# Hard safety valve for expert generation length when R_out is unknown or the
# convolution call fails. Not the operating point — apex-nadir's R_out is. Kept
# small on 16GB: autoregressive expert generation holds a KV cache, a real chunk of
# the per-batch memory spike.
EXPERT_GEN_MAX_TOKENS = 16
TKL_FLOOR = 32
TKL_HISTORY_LEN = 10
MONOPOLY_THRESHOLD = 0.85
CALIBRATION_PATH = "state/calibration.npz"
LATENCY_STORE_PATH = "state/latency_store.npz"
VORONOI_ALPHA = 0.3
# Voronoi route-cache acceptance threshold (cosine distance). Measured on the
# gate's mean-pooled fingerprints (scripts/realistic_workload.py): paraphrases of
# the SAME query sit at cosine dist ~0.018 (p90 0.033); UNRELATED queries at
# ~0.136 (p10 0.063). So a tau in [0.033, 0.063] cleanly separates "same intent"
# from "different intent". The old cold-start fallback used VORONOI_ALPHA (0.30)
# directly when <2 clusters existed — ~7x too loose, so the first cluster
# swallowed everything before the cache could tighten (same-base hit accuracy
# was 47%, barely above chance). These bound the threshold to the measured band.
# Set near the within-paraphrase p90 (0.033): favours PRECISION because a false
# cache hit routes to the wrong experts, whereas a miss merely re-runs expert
# selection — which the workload harness showed is near-free (the gate pass, not
# selection, is the routing cost). Sweep (scripts/realistic_workload.py):
#   tau≈0.06  → 86% hit / 69% same-query precision
#   tau≈0.033 → 84% hit / ~80% precision   (this default)
#   tau≈0.020 → 75% hit / 91% precision    (tighter; risks missing real rewrites)
VORONOI_TAU_COLD = 0.030   # absolute tau when <2 clusters exist (cold cache)
VORONOI_TAU_CEIL = 0.040   # cap on the warm tau = ALPHA * mean_inter_centroid_dist

# DOMAIN MEMBERSHIP BANDS — a different question from tau above. Tau asks "is
# this the same QUERY I already cached" (measured query-to-query: paraphrases
# 0.018, unrelated 0.136). These ask "which DOMAIN does this belong to", and are
# measured centroid-to-centroid. Widths 10/20/30/40: finest resolution where the
# decision is hardest, coarsest where nothing is at stake.
#
# Validated on all 190 centroid pairs in state/routing_memory.pkl — ZERO errors:
#   member    21/21 same-domain     neighbour  23/23 same-domain
#   far         2/2 same-domain     corner    0/144 same-domain
# Cross-domain similarity tops out at 0.0878 and same-domain bottoms at 0.6600,
# so SIM_FAR sits inside the empirical dead gap where nothing can be misfiled.
# Merging at SIM_MEMBER collapses the 20 stored clusters to the correct 9. (An
# earlier note said 7: that used connected components, but merge_close_clusters
# merges pairwise and recomputes the centroid as it goes, so it does not chain.)
SIM_MEMBER    = 0.90   # cosine similarity — this IS the domain
SIM_NEIGHBOUR = 0.70   # immediate neighbour of the centroid
SIM_FAR       = 0.40   # on the map; below this, no association
CLUSTER_CAP_RATE = 50
CLUSTER_PRUNE_AGE = 10_000
CLUSTER_CONFIDENCE_FLOOR = 0.4

# ── sqrt bracketing: how many experts a tier holds ──────────────────────────
# c = EXPERTS. sqrt(c), rounding UP for the general pool (a grace period, so
# round toward keeping more) and DOWN per centroid (a selection rule, so round
# toward admitting fewer). Derived from EXPERT_POOL_SIZE rather than typed in,
# so the numbers follow the pool.
#
# These are structural CAPS and are meant to hold still — unlike compute_r_out,
# which was meant to vary per expert and did not. A cap that does not move is
# working.
#
# At N=100: general=10, per-centroid=9. Note per-centroid does not reference the
# cluster count, so the caps sum past the pool beyond 10 clusters (11 x 9 = 99 >
# 90). Re-formation is expected to yield ~9 — one cluster of headroom.
GENERAL_EXPERTS      = math.ceil(math.sqrt(EXPERT_POOL_SIZE))
_NON_GENERAL         = EXPERT_POOL_SIZE - GENERAL_EXPERTS
CENTROID_EXPERTS     = math.floor(math.sqrt(_NON_GENERAL)) if _NON_GENERAL > 0 else 0
MAX_CLUSTERS_BY_POOL = (_NON_GENERAL // CENTROID_EXPERTS) if CENTROID_EXPERTS else 0

# ── Markov chains (chain.py): expert migration, cluster territory ───────────
# CHAIN_MEMORY is the one number that decides how this system remembers. Counts
# accumulate freely up to it, then dilute. Asymptotically an EMA with
# lambda = 1 - 1/CHAIN_MEMORY, but stated as evidence rather than a decay rate:
# "the chain remembers 500 transitions." Unbounded counts would freeze the
# estimate (one new observation moves it by 1/n) with no symptom at all.
CHAIN_MEMORY = 500.0
# Prior mass per cell. A row with a few observations returns near-uniform — "no
# opinion" — which removes the need for an abstain-if-starved guard at every
# call site. A guard gets forgotten; a prior cannot.
CHAIN_PRIOR = 1.0
# How much inherited pool behaviour a fresh per-expert chain starts with, in
# pseudo-observations. A new expert is worth ~8 pool moves of prior belief and
# is outvoted by its own evidence soon after that — the pool is a starting
# point, not a verdict.
CHAIN_SEED_STRENGTH = 8.0
# tau_k is the membership cut, and it is now per-cluster state rather than a
# hand-set global — a fixed cut was an ungrounded constant closing an adaptive
# loop, which is this system's most-repeated bug. It starts at SIM_MEMBER and
# breathes with observed traffic, clamped into [SIM_NEIGHBOUR, TAU_MAX] so a
# mispredicting chain cannot make a cluster swallow the sphere or vanish.
TAU_STEP = 0.005
TAU_MAX  = 0.97
FAST_PATH_THRESHOLD = 0.70
THERMAL_SAMPLE_INTERVAL = 1
THERMAL_THROTTLE_TEMP = 85.0
# Concurrency back-off ratios for the live expert governor (diagnostics). These are
# RELATIVE to runtime metrics (throttle temp / best observed throughput), not
# absolute walls — back off above THERMAL_BACKOFF_FRAC of the throttle temp, or
# when throughput drops below THROUGHPUT_COLLAPSE_FRAC of the best seen this run.
THERMAL_BACKOFF_FRAC = 0.9
THROUGHPUT_COLLAPSE_FRAC = 0.5
# Expert-concurrency control on OBSERVED peak-memory utilization (peak / usable RAM).
# Grow x only when the last measured peak left >= (1 - GROW) headroom; back off when
# it exceeds BACKOFF. Empirical (no cost extrapolation) so it cannot overshoot into a
# Metal OOM. Relative to measured usable RAM, so still Zero-Constants-compliant.
MEM_GROW_HEADROOM_FRAC = 0.60   # grow x only if last peak used < this fraction
MEM_BACKOFF_FRAC = 0.80         # shrink x (toward 1) if last peak exceeded this
# Central-only fallback fires only when even ONE expert would risk a real OOM. Set
# ABOVE MEM_BACKOFF_FRAC: at 1 expert we still RUN (safe) up to here, we just don't
# grow — only skip experts entirely when peak is genuinely near the ceiling.
MEM_FALLBACK_FRAC = 0.92
DIAGNOSTICS_SAVE_PATH = "state/diagnostics.pkl"
X_MIN = 1
X_MAX = 6        # soft ceiling: most experts that may run concurrently. The live
                 # memory/thermal/throughput governor decides the ACTUAL count <= this
                 # each batch; this is just the highest it is ever allowed to reach.
LAMBDA_INIT = [0.25, 0.25, 0.25, 0.25]
ALPHA_LR = 1e-4
BETA_LR = 1e-5
# MAML lambda meta-update rate. SEPARATE from ALPHA/BETA (which govern the
# gate-parameter MAML inner/outer steps and keep their structural 10:1 ratio).
# At BETA_LR=1e-5 the loss-weight lambdas moved ~3e-6/step — effectively frozen,
# so the "emergence" loop was dead. This rate lets the lambdas adapt to per-domain
# training signal within a few thousand tokens. Paired with LAMBDA_FLOOR so the
# (linear) meta-loss can't collapse the weights onto a single objective.
LAMBDA_META_LR = 0.03
# Minimum weight every loss term keeps after the meta-update. Prevents degenerate
# collapse (e.g. all weight on l_eff, zeroing the l_dom routing loss). With 4
# lambdas and floor 0.05, each stays in [0.05, 0.85] and the sum stays 1.0.
LAMBDA_FLOOR = 0.05
T_CHECKPOINT = 5
L_EFF_EPS = 1e-8
L_REL_GAMMA = 0.95
L_REL_N_WINDOWS = 10
MASKING_STUCK_THRESHOLD = 0.9
ALPHA_PROTECTION_THRESHOLD = 0.5
EMA_DECAY = 0.99
STARVATION_MIN_ACTIVATIONS = 5   # expert must have this many activations in domain before eviction
OUTER_LOOP_TOKEN_INTERVAL = 500
ROUTING_MEMORY_PATH = "state/routing_memory.pkl"
# Expert history now SURVIVES a session — the old reset() wiped it every run,
# which was a training-era workaround from when experts were constantly
# retrained. Bounded the same way the Markov chains are bounded: keep the last
# N activations per expert so the record cannot grow without limit, and one new
# observation keeps a constant weight instead of decaying as 1/n.
SESSION_TRACKER_PATH = "state/session_tracker.pkl"
SESSION_HISTORY_CAP = 200
# How many experts per batch get the grounded leave-one-out delta instead of the
# cosine r_i. Leave-one-out costs one extra Central forward per expert, so this
# is a budget, not a quality dial — the grounded signal only has to be PERIODIC
# to keep the routing head anchored to real text rather than to agreement with
# Central's own hidden state.
GROUNDED_SAMPLE_K = 2
LAMBDA_SAVE_PATH = "state/lambdas.npz"
CHECKPOINT_DIR = Path("state/checkpoints/")
LOG_DIR = Path("logs/")
DEVICE = None

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
# Gradient-norm clip applied before every gate/expert optimizer step. The 3-term
# L_gate over the routing head can spike on real data; clipping bounds the update
# so a single bad batch can't blow weights to NaN (the finite guard then skips any
# residual non-finite step).
GRAD_CLIP_NORM = 1.0
# The gate's learned expert-routing head (gating.GateNet.route_head) emits a
# preference logit per expert. select_experts blends that learned preference with
# the Apex-Nadir distance-to-peak: final_rank = distance_to_peak - ROUTE_BIAS_W *
# softmax(route_logits). Apex-nadir keeps routing grounded while the head is still
# learning; raise this as the head matures to let the gate drive selection.
ROUTE_BIAS_W = 0.5
# Experts start UNASSIGNED. Domain membership is earned from measured
# performance (gating.DomainRegistry), never declared up front.
#
# This used to hard-partition the pool 25/25/25/25 by expert id before a single
# token had been seen — asserting that expert 7 is a "code" expert purely because
# of its index, and locking the split to a uniform prior that no real corpus
# matches. Leave empty; TripleKSelector then draws from the whole pool until
# measurement says otherwise, which is the correct cold start. A dict here still
# works as a manual override for reproducing a fixed assignment.
EXPERT_GROUPS: dict = {}
# ── TRAINING DATA (only used when a trainer is running; the engine never
# imports these). Lean, domain-balanced set covering all four DOMAINS so the
# curriculum can request a specific domain's tokens. DATASET_DOMAINS is the
# ground-truth label — dataset provenance, not keyword sniffing of the text.
DATASET_BOOT_TIMEOUT = 60
DATASET_SAMPLE_TIMEOUT = 60
DATASET_IDS = {
    "github_code":         ("HuggingFaceH4/CodeAlpaca_20K", None, "train"),
    "python_instructions": ("iamtarun/python_code_instructions_18k_alpaca", None, "train"),
    "gsm8k":               ("openai/gsm8k", "main", "train"),
    "metamath":            ("meta-math/MetaMathQA", None, "train"),
    "ai2_arc":             ("allenai/ai2_arc", "ARC-Challenge", "train"),
    "camel_science":       ("sciq", None, "train"),
    "slimorca":            ("Open-Orca/SlimOrca", None, "train"),
    "ultrachat":           ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
}
DATASET_DOMAINS = {
    "github_code": "code", "python_instructions": "code",
    "gsm8k": "reasoning", "metamath": "reasoning",
    "ai2_arc": "knowledge", "camel_science": "knowledge",
    "slimorca": "general", "ultrachat": "general",
}
DATASET_WEIGHTS = {k: 1.0 / len(DATASET_IDS) for k in DATASET_IDS}


def validate_config() -> None:
    # E is a VARIABLE. Everything downstream is algebra over it — sqrt(E) sample
    # sizes, ceil(sqrt(E/D)) domain floors, sqrt(E)/2 new-domain seeds — so the
    # pool can grow, shrink, or be re-partitioned across a different number of
    # domains without touching code. A hard `!= 100 -> raise` here made the
    # "horizontally scalable" pool a fixed-size one and blocked adding or
    # retiring experts outright. Only the genuinely impossible is rejected.
    if EXPERT_POOL_SIZE < 1:
        raise ValueError(f"EXPERT_POOL_SIZE must be >= 1, got {EXPERT_POOL_SIZE}.")
    if K_MAX > EXPERT_POOL_SIZE:
        raise ValueError(f"K_MAX ({K_MAX}) cannot exceed EXPERT_POOL_SIZE ({EXPERT_POOL_SIZE}).")
    if not (0 <= K_MIN <= K_DEFAULT <= K_MAX <= 20):
        raise ValueError("K bounds must be within [0, 20] and ordered.")
    if not (0.0 < FAST_PATH_THRESHOLD < 1.0):
        raise ValueError("FAST_PATH_THRESHOLD must be between 0 and 1.")
    if abs(BETA_LR - ALPHA_LR / 10) > 1e-12:
        raise ValueError(
            f"BETA_LR ({BETA_LR}) must equal ALPHA_LR / 10 ({ALPHA_LR / 10}). "
            "Structural constraint."
        )
    if TKL_FLOOR != FRAGMENT_MIN:
        raise ValueError("TKL_FLOOR must equal FRAGMENT_MIN (both = 32).")
    if FRAGMENT_MIN < 32:
        raise ValueError("FRAGMENT_MIN must be >= 32.")
if __name__ == "__main__":
    validate_config()
    print("Config OK")
