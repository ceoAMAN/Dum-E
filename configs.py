from __future__ import annotations
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
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
WOLFRAM_APP_ID = os.getenv("WOLFRAM_APP_ID", "").strip()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
OPENWEATHERMAP_KEY = os.getenv("OPENWEATHERMAP_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
DEPLOYMENT = os.getenv("STURNUS_DEPLOYMENT", "False").lower() in ("true", "1", "yes")
GATE_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
EXPERT_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CENTRAL_MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
EXPERT_POOL_SIZE = 100
NUM_EXPERTS = EXPERT_POOL_SIZE
# Fallback RAM-per-expert estimate. Measured for real at boot via
# splitter.measure_expert_ram_mb() (loads one expert, reads the delta) and the
# measured value replaces this at runtime, so X/Y geometry uses the true cost on
# whatever hardware Sturnus runs on rather than a hardcoded guess.
# Cold-start floor estimate of per-expert RAM, used only until the live memory
# governor (diagnostics) measures the REAL marginal cost (weights + 7B-forward +
# generation spike) and takes over. No hard expert cap exists — concurrency is
# derived per batch from measured memory, thermal, and processing load.
EXPERT_RAM_MB = 850
CENTRAL_RAM_MB = 4096
MIN_BOOT_RAM_MB = 6000
GATE_D_MODEL = 896
EXPERT_D_MODEL = 1536
CENTRAL_D_MODEL = 4096
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
MAX_SEQ_LEN = 256   # lowered from 512 for 16GB: halves the 7B-forward activation spike
# Cold-start expert fragment/generation size, used ONLY before the convolution has
# enough latency/quality data to produce an R_out for an expert (compute_r_out
# returns None until then). Once R_out exists it governs and this is ignored —
# the "first run gathers context, apex-nadir takes over" handshake.
EXPERT_BOOTSTRAP_TOKENS = 128
# Hard safety valve for expert generation length when R_out is unknown or the
# convolution call fails. Not the operating point — apex-nadir's R_out is.
EXPERT_GEN_MAX_TOKENS = 48
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
CLUSTER_CAP_RATE = 50
CLUSTER_PRUNE_AGE = 10_000
CLUSTER_CONFIDENCE_FLOOR = 0.4
FAST_PATH_THRESHOLD = 0.70
THERMAL_SAMPLE_INTERVAL = 1
THERMAL_THROTTLE_TEMP = 85.0
# Concurrency back-off ratios for the live expert governor (diagnostics). These are
# RELATIVE to runtime metrics (throttle temp / best observed throughput), not
# absolute walls — back off above THERMAL_BACKOFF_FRAC of the throttle temp, or
# when throughput drops below THROUGHPUT_COLLAPSE_FRAC of the best seen this run.
THERMAL_BACKOFF_FRAC = 0.9
THROUGHPUT_COLLAPSE_FRAC = 0.5
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
# LEAN 8-dataset mixture for 16GB stability: balanced across all 4 domains (each
# ~0.25) with LIGHT streams only. 19 concurrent HF streams' Arrow buffers were a
# multi-GB RAM hog that OOM'd the 7B+experts run; the heavy web crawls (c4/dolma,
# fineweb) and large sets (the_stack/CodeFeedback, tulu, ultrachat-200k) are dropped.
# Re-enable more once running on a bigger machine.
DATASET_WEIGHTS_MAC = {
    # code (0.25)
    "github_code": 0.15,            # CodeAlpaca_20K — light
    "python_instructions": 0.10,    # python_code_instructions_18k — light
    # reasoning (0.25)
    "gsm8k": 0.15,
    "metamath": 0.10,
    # knowledge (0.25)
    "ai2_arc": 0.15,
    "camel_science": 0.10,          # sciq
    # general (0.25)
    "slimorca": 0.15,
    "ultrachat": 0.10,
    # ── disabled for the lean 16GB run (0.0) ──
    "local_custom": 0.0, "xlam_function_calling": 0.0, "hermes_function_calling": 0.0,
    "glaive_function_calling": 0.0, "agent_flan": 0.0, "agentinstruct_zai": 0.0,
    "wizardlm_evol": 0.0, "tulu_v2": 0.0, "open_platypus": 0.0, "math": 0.0,
    "the_stack": 0.0, "redpajama": 0.0, "dolma": 0.0,
    "openhermes": 0.0, "wikipedia": 0.0, "openassistant": 0.0,
}

DATASET_WEIGHTS_TAB = {
    "local_custom": 0.0,             # SKIP: no data/custom_prompts.jsonl yet
    "xlam_function_calling": 0.0,    # SKIP: gated on HF (no access)
    "hermes_function_calling": 0.05,
    "glaive_function_calling": 0.03,
    "agent_flan": 0.03,
    "agentinstruct_zai": 0.02,
    "slimorca": 0.03,
    "wizardlm_evol": 0.03,
    "tulu_v2": 0.04,
    "open_platypus": 0.02,
    "ultrachat": 0.04,
    "metamath": 0.03,
    "math": 0.02,
    "gsm8k": 0.02,
    "the_stack": 0.03,
    "github_code": 0.03,
    "python_instructions": 0.01,
    "camel_science": 0.01,
    "ai2_arc": 0.01,
    "redpajama": 0.02,
    "dolma": 0.02,
    "openhermes": 0.0,      # SKIP: timeout/hang
    "wikipedia": 0.0,       # SKIP: timeout/hang
    "openassistant": 0.0,   # SKIP: stream deadlock
}

DATASET_WEIGHTS = DATASET_WEIGHTS_MAC

# Renormalise the positive weights to sum to 1.0. Lets us edit any single weight
# (e.g. drop local_custom to de-skew) without hand-balancing the other ~18 by
# arithmetic — relative proportions of the rest are preserved, the disabled
# (0.0) streams stay disabled, and validate_config's sum==1.0 check still holds.
_w_total = sum(w for w in DATASET_WEIGHTS.values() if w > 0.0)
if _w_total > 0:
    DATASET_WEIGHTS = {
        k: (w / _w_total if w > 0.0 else 0.0) for k, w in DATASET_WEIGHTS.items()
    }

# Local-only override: when STURNUS_LOCAL_ONLY=1, train purely on the local
# custom_prompts.jsonl (no network streams). Used for fast, deterministic
# engine-validation runs (clean convergence curves) when HF streaming is slow.
if os.getenv("STURNUS_LOCAL_ONLY", "").lower() in ("1", "true", "yes"):
    DATASET_WEIGHTS = {"local_custom": 1.0}

DATASET_IDS = {
    "local_custom": ("json", {"train": "data/custom_prompts.jsonl"}, "train"),
    # ── action / tool-calling layer ────────────────────────────────────────
    "xlam_function_calling": ("Salesforce/xlam-function-calling-60k", None, "train"),
    "hermes_function_calling": ("NousResearch/hermes-function-calling-v1", None, "train"),
    "glaive_function_calling": ("glaiveai/glaive-function-calling-v2", None, "train"),
    "agent_flan": ("internlm/Agent-FLAN", None, "agent_instruct_react"),
    "agentinstruct_zai": ("zai-org/AgentInstruct", None, "os"),
    # ── existing datasets ─────────────────────────────────────────────────
    "slimorca": ("Open-Orca/SlimOrca", None, "train"),
    "wizardlm_evol": ("WizardLM/WizardLM_evol_instruct_V2_196k", None, "train"),
    "tulu_v2": ("allenai/tulu-v2-sft-mixture", None, "train"),
    "open_platypus": ("garage-bAInd/Open-Platypus", None, "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft"),
    "metamath": ("meta-math/MetaMathQA", None, "train"),
    "math": ("HuggingFaceH4/MATH-500", "default", "test"),
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "the_stack": ("m-a-p/CodeFeedback-Filtered-Instruction", None, "train"),
    "github_code": ("HuggingFaceH4/CodeAlpaca_20K", None, "train"),
    "python_instructions": ("iamtarun/python_code_instructions_18k_alpaca", None, "train"),
    "camel_science": ("sciq", None, "train"),
    "ai2_arc": ("allenai/ai2_arc", "ARC-Challenge", "train"),
    "redpajama": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train"),
    "dolma": ("allenai/c4", "en", "train"),
    "openhermes": ("teknium/OpenHermes-2.5", None, "train"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "train"),
    "openassistant": ("OpenAssistant/oasst2", None, "train"),
}
# Ground-truth domain per dataset — the dataset's provenance IS its domain (gsm8k is
# reasoning, CodeFeedback is code), so the gate's L_dom target comes from here rather
# than keyword-sniffing the text. Anything unmapped falls back to general.
DATASET_DOMAINS = {
    "the_stack": "code", "github_code": "code", "python_instructions": "code",
    "gsm8k": "reasoning", "metamath": "reasoning", "math": "reasoning",
    "camel_science": "knowledge", "ai2_arc": "knowledge", "redpajama": "knowledge",
    "dolma": "knowledge", "wikipedia": "knowledge",
    "ultrachat": "general", "slimorca": "general", "wizardlm_evol": "general",
    "tulu_v2": "general", "open_platypus": "general", "openhermes": "general",
    "openassistant": "general", "local_custom": "general",
    "xlam_function_calling": "general", "hermes_function_calling": "general",
    "glaive_function_calling": "general", "agent_flan": "general", "agentinstruct_zai": "general",
}
DATASET_BOOT_TIMEOUT = 60.0
DATASET_SAMPLE_TIMEOUT = 5.0
ROUTING_MEMORY_PATH = "state/routing_memory.pkl"
LAMBDA_SAVE_PATH = "state/lambdas.npz"
CHECKPOINT_DIR = Path("state/checkpoints/")
LOG_DIR = Path("logs/")
DEVICE = None

CENTRAL_TRAIN_MODEL_ID = CENTRAL_MODEL_ID
GATE_TRAIN_MODEL_ID = GATE_MODEL_ID
EXPERT_TRAIN_MODEL_ID = EXPERT_MODEL_ID
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
EXPERT_GROUPS = {
    "code": list(range(0, 25)),
    "reasoning": list(range(25, 50)),
    "knowledge": list(range(50, 75)),
    "general": list(range(75, 100)),
}
def validate_config() -> None:
    if EXPERT_POOL_SIZE != 100:
        raise ValueError("EXPERT_POOL_SIZE must be 100.")
    if not (0 <= K_MIN <= K_DEFAULT <= K_MAX <= 20):
        raise ValueError("K bounds must be within [0, 20] and ordered.")
    if not (0.0 < FAST_PATH_THRESHOLD < 1.0):
        raise ValueError("FAST_PATH_THRESHOLD must be between 0 and 1.")
    if abs(sum(DATASET_WEIGHTS.values()) - 1.0) > 1e-6:
        raise ValueError("DATASET_WEIGHTS must sum to 1.0.")
    if abs(BETA_LR - ALPHA_LR / 10) > 1e-12:
        raise ValueError(
            f"BETA_LR ({BETA_LR}) must equal ALPHA_LR / 10 ({ALPHA_LR / 10}). "
            "Structural constraint."
        )
    if TKL_FLOOR != FRAGMENT_MIN:
        raise ValueError("TKL_FLOOR must equal FRAGMENT_MIN (both = 32).")
    if FRAGMENT_MIN < 32:
        raise ValueError("FRAGMENT_MIN must be >= 32.")
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is required. Set the HF_TOKEN environment variable before running."
        )
if __name__ == "__main__":
    validate_config()
    print("Config OK")
