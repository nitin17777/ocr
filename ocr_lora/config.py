"""Configuration and constants for OCR LoRA evaluation harness.

All paths, prompt templates, model identifiers, hyperparameters, and corruption rules
are defined here. Secrets are loaded dynamically via environment variables.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINT_INPUT_PATH = DATA_DIR / "ocr_eval_checkpoint.jsonl"
CHECKPOINT_OUTPUT_PATH = OUTPUTS_DIR / "ocr_eval_checkpoint.jsonl"
REPORT_OUTPUT_PATH = OUTPUTS_DIR / "report.md"

# Prompt template (exact - do not alter)
PROMPT_TEMPLATE = "### Instruction:\nFix the OCR errors: {input}\n\n### Response:\n"

# Models and adapters
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER_ID = "deep1104/ocr-lora-large"
SECONDARY_ADAPTER_IDS = [
    "deep1104/ocr-lora-big",
    "deep1104/ocr-lora-ablation-noclean",
]

# Eval configuration
EVAL_BATCH_SIZE = 8
EVAL_DO_SAMPLE = False
EVAL_MAX_NEW_TOKENS = 100
EVAL_MAX_LENGTH = 256
EVAL_PADDING_SIDE = "left"

# Training configuration (reference for retraining / verification)
TRAIN_LORA_R = 16
TRAIN_LORA_ALPHA = 32
TRAIN_LORA_DROPOUT = 0.05
TRAIN_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
TRAIN_GRAD_ACCUM = 2
TRAIN_LR = 2e-4
TRAIN_LR_SCHEDULER = "cosine"
TRAIN_WARMUP_RATIO = 0.1
TRAIN_MAX_SEQ_LEN = 256
TRAIN_QUANTIZATION = "4bit-nf4"

# Data generation configuration
RANDOM_SEED = 42
WIKITEXT_DATASET_ID = "Salesforce/wikitext"
WIKITEXT_PARQUET_FILE = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
MIN_SENTENCE_LEN = 40
MAX_SENTENCE_LEN = 110
NUM_MATH_SENTENCES = 5000

MATH_TEMPLATES = [
    "a + b = c.",
    "a × b = c.",
    "Add a and b to get c.",
    "Multiply a by b to get c.",
    "The sum of a and b is c.",
    "Subtract b from a to get c.",
    "a − b = c.",
    "a ÷ b = c.",
    "Divide a by b to get c.",
    "Solve: a ÷ b = c.",
]

CORRUPTION_LEVELS = [0.10, 0.25, 0.45]

TEXT_SWAPS = {
    "l": "1",
    "1": "l",
    "O": "0",
    "0": "O",
    "S": "5",
    "5": "S",
    "rn": "m",
}

MATH_SWAPS = {
    "×": "x",
    "÷": "+",
    "−": "=",
    "+": "÷",
    "=": "−",
}


def get_hf_token() -> str | None:
    """Retrieve Hugging Face token securely from environment.

    Never print, log, or hardcode this value.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        token = token.strip()
    return token
