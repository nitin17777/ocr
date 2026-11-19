"""Model and tokenizer loader for OCR LoRA evaluation harness.

Handles:
- AutoTokenizer loading with padding_side="left" and pad_token resolution.
- AutoModelForCausalLM loading in fp16 with device_map="auto" (or CPU/MPS fallback).
- Mounting PeftModel adapter (defaulting to deep1104/ocr-lora-large).
- Secure token handling via get_hf_token().
"""

from typing import Tuple, Optional
from pathlib import Path
import torch
try:
    import torch.distributed.tensor
except ImportError:
    pass
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from peft import PeftModel

from ocr_lora.config import (
    BASE_MODEL_ID,
    DEFAULT_ADAPTER_ID,
    EVAL_PADDING_SIDE,
    MODELS_DIR,
    get_hf_token,
)


def get_optimal_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Determine optimal device and torch_dtype based on system capabilities."""
    if torch.cuda.is_available():
        return "auto", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def load_tokenizer(
    model_id_or_path: str = BASE_MODEL_ID,
    token: Optional[str] = None,
) -> PreTrainedTokenizer:
    """Load AutoTokenizer configured with left padding."""
    if token is None:
        token = get_hf_token()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = EVAL_PADDING_SIDE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def load_model_and_adapter(
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = DEFAULT_ADAPTER_ID,
    local_models_dir: Optional[Path] = MODELS_DIR,
    token: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Load base model and mount LoRA adapter.
    
    Checks local_models_dir first if present, otherwise downloads/loads from HF Hub.
    """
    if token is None:
        token = get_hf_token()

    device_map, torch_dtype = get_optimal_device_and_dtype()

    # Determine paths: check local directory first
    base_target = base_model_id
    adapter_target = adapter_id

    if local_models_dir:
        local_base = local_models_dir / base_model_id.replace("/", "--")
        local_adapter = local_models_dir / adapter_id.replace("/", "--")
        if local_base.exists():
            base_target = str(local_base)
        if local_adapter.exists():
            adapter_target = str(local_adapter)

    tokenizer = load_tokenizer(base_target, token=token)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_target,
        token=token,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Attach PEFT adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_target,
        token=token,
    )
    model.eval()

    return model, tokenizer
