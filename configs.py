# pyre-unsafe
"""Sturnus configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _is_colab() -> bool:
    return bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("COLAB_GPU"))


HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

EXPERT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GATE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CENTRAL_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

EXPERT_TRAIN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
GATE_TRAIN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
CENTRAL_TRAIN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

NUM_EXPERTS = 240
EXPERT_GROUPS = {
    "A": list(range(0, 15)),
    "B": list(range(15, 30)),
    "C": list(range(30, 45)),
    "D": list(range(45, 60)),
    "E": list(range(60, 75)),
    "F": list(range(75, 90)),
    "G": list(range(90, 105)),
    "H": list(range(105, 120)),
    "I": list(range(120, 135)),
    "J": list(range(135, 150)),
    "K": list(range(150, 165)),
    "L": list(range(165, 180)),
    "M": list(range(180, 195)),
    "N": list(range(195, 210)),
    "O": list(range(210, 225)),
    "P": list(range(225, 240)),
}

# The MLX 4-bit config sizes
QUANT_BITS = 4
EXPERT_RAM_MB = 125

EXPERT_D_MODEL = 896
GATE_D_MODEL = 1536
CENTRAL_D_MODEL = 2048

NATIVE_MODE = os.environ.get('STURNUS_NATIVE', '0') == '1'

K_MIN = 0
K_MAX = 20
K_DEFAULT = 4
FAST_PATH_THRESHOLD = 0.85

X_MIN = 19
X_DEFAULT = 19
X_MAX = 104
Y_MAX = 25

TIMELINE_B_BUDGET_SECS = 10.0
DEAD_TIME_MIN_SECS = 3.0

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 512

DATASET_WEIGHTS = {
    "fineweb": 0.125,
    "arxiv": 0.125,
    "code_search_net": 0.125,
    "dolma": 0.125,
    "math": 0.0625,
    "gsm8k": 0.0625,
    "openhermes": 0.125,
    "c4": 0.125,
    "medqa": 0.125,
}

DATASET_IDS = {
    "fineweb": ("HuggingFaceFW/fineweb", "default"),
    "arxiv": ("Intelligent-Internet/arxiv", None),
    "code_search_net": ("code_search_net", "all"),
    "dolma": ("allenai/dolma", None),
    "math": ("lighteval/MATH", "all"),
    "gsm8k": ("gsm8k", "main"),
    "openhermes": ("teknium/OpenHermes-2.5", None),
    "c4": ("allenai/c4", "en"),
    "medqa": ("GBaker/MedQA-USMLE-4-options", None),
}

UTILIZATION_WINDOW = 1000
UNDERUSE_THRESHOLD = 0.02
MASKING_RATE = 0.10
EMA_DECAY = 0.99

DEVICE = "cuda" if _is_colab() else "cpu"
USE_FP16 = True if _is_colab() else False

BASE_PATH = "/content/drive/MyDrive/Sturnus" if _is_colab() else "./Sturnus"
BASE_DIR = Path(BASE_PATH).expanduser().resolve()
CHECKPOINT_DIR = Path(os.getenv("STURNUS_CHECKPOINT_DIR", "./checkpoints")).expanduser().resolve()

USE_MOCK_INFERENCE = _env_bool("STURNUS_MOCK_INFERENCE", not bool(HF_TOKEN))
GATE_MODE = os.getenv("STURNUS_GATE_MODE", "llm" if HF_TOKEN else "heuristic").strip().lower()
REQUEST_TIMEOUT_SECS = float(os.getenv("STURNUS_REQUEST_TIMEOUT", "60"))
DEBUG = _env_bool("STURNUS_DEBUG", False)
VECTOR_BACKEND = os.getenv("STURNUS_VECTOR_BACKEND", "hash").strip().lower()
VECTOR_MODEL_ID = os.getenv("STURNUS_VECTOR_MODEL_ID", CENTRAL_MODEL_ID).strip()

HF_CHAT_API_URL = "https://router.huggingface.co/v1/chat/completions"


@dataclass(frozen=True)
class ConfigSnapshot:
    hf_token_present: bool
    device: str
    use_fp16: bool
    base_path: str
    mock_inference: bool
    gate_mode: str
    vector_backend: str
    vector_model_id: str
    checkpoint_dir: str


CONFIG_SNAPSHOT = ConfigSnapshot(
    hf_token_present=bool(HF_TOKEN),
    device=DEVICE,
    use_fp16=USE_FP16,
    base_path=str(BASE_DIR),
    mock_inference=USE_MOCK_INFERENCE,
    gate_mode=GATE_MODE,
    vector_backend=VECTOR_BACKEND,
    vector_model_id=VECTOR_MODEL_ID,
    checkpoint_dir=str(CHECKPOINT_DIR),
)


def validate_config() -> None:
    if NUM_EXPERTS != 240:
        raise ValueError("NUM_EXPERTS must be 240.")
    total_groups = sum(len(v) for v in EXPERT_GROUPS.values())
    if total_groups != NUM_EXPERTS:
        raise ValueError("EXPERT_GROUPS must cover all experts exactly.")
    if not (0 <= K_MIN <= K_DEFAULT <= K_MAX <= 20):
        raise ValueError("K bounds must be within [0, 20] and ordered.")
    if not (0.0 < FAST_PATH_THRESHOLD < 1.0):
        raise ValueError("FAST_PATH_THRESHOLD must be between 0 and 1.")
    if abs(sum(DATASET_WEIGHTS.values()) - 1.0) > 1e-6:
        raise ValueError("DATASET_WEIGHTS must sum to 1.0.")
    if VECTOR_BACKEND not in {"hash", "hf", "hf_feature"}:
        raise ValueError("VECTOR_BACKEND must be one of: hash, hf, hf_feature.")


def _print_config() -> None:
    print("[Sturnus config]")
    print(f"  HF_TOKEN present: {bool(HF_TOKEN)}")
    print(f"  EXPERT_MODEL_ID: {EXPERT_MODEL_ID}")
    print(f"  GATE_MODEL_ID: {GATE_MODEL_ID}")
    print(f"  CENTRAL_MODEL_ID: {CENTRAL_MODEL_ID}")
    print(f"  NUM_EXPERTS: {NUM_EXPERTS}")
    print(f"  K_MIN/K_DEFAULT/K_MAX: {K_MIN}/{K_DEFAULT}/{K_MAX}")
    print(f"  X_DEFAULT/X_MAX/Y_MAX: {X_DEFAULT}/{X_MAX}/{Y_MAX}")
    print(f"  USE_MOCK_INFERENCE: {USE_MOCK_INFERENCE}")
    print(f"  GATE_MODE: {GATE_MODE}")
    print(f"  VECTOR_BACKEND: {VECTOR_BACKEND}")


if os.getenv("STURNUS_CONFIG_QUIET", "0") != "1":
    _print_config()

if __name__ == "__main__":
    validate_config()
    print("Config OK")
