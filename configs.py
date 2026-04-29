from __future__ import annotations
import os
from pathlib import Path
def _load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env.local"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
_load_local_env()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GATE_MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
EXPERT_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
CENTRAL_MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
EXPERT_POOL_SIZE = 100
NUM_EXPERTS = EXPERT_POOL_SIZE
EXPERT_RAM_MB = 850
CENTRAL_RAM_MB = 4096
MIN_BOOT_RAM_MB = 6000
GATE_D_MODEL = 896
EXPERT_D_MODEL = 1536
CENTRAL_D_MODEL = 4096
FRAGMENT_MIN = 32
OVERLAP_FRACTION = 0.175
K_MIN = 0
K_MAX = 20
K_DEFAULT = 4
MAX_SEQ_LEN = 512
TKL_FLOOR = 32
TKL_HISTORY_LEN = 10
MONOPOLY_THRESHOLD = 0.85
CALIBRATION_PATH = "state/calibration.npz"
LATENCY_STORE_PATH = "state/latency_store.npz"
VORONOI_ALPHA = 0.3
CLUSTER_CAP_RATE = 50
CLUSTER_PRUNE_AGE = 10_000
CLUSTER_CONFIDENCE_FLOOR = 0.4
FAST_PATH_THRESHOLD = 0.85
LAMBDA_INIT = [0.25, 0.25, 0.25, 0.25]
ALPHA_LR = 1e-4
BETA_LR = 1e-5
T_CHECKPOINT = 5
L_EFF_EPS = 1e-8
L_REL_GAMMA = 0.95
L_REL_N_WINDOWS = 10
MASKING_STUCK_THRESHOLD = 0.9
ALPHA_PROTECTION_THRESHOLD = 0.5
EMA_DECAY = 0.99
OUTER_LOOP_TOKEN_INTERVAL = 500
DATASET_WEIGHTS = {
    "slim_orca": 0.25,
    "red_pajama": 0.25,
    "starcoder": 0.25,
    "fineweb": 0.25,
}
DATASET_IDS = {
    "slim_orca": ("Open-Orca/SlimOrca", "default"),
    "red_pajama": ("wikitext", "wikitext-103-v1"),
    "starcoder": ("bigcode/starcoderdata", "default"),
    "fineweb": ("HuggingFaceFW/fineweb", "CC-MAIN-2024-10"),
}
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
