"""ModelManager singleton with lifecycle management, state machine, and adapter swapping."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import time
from typing import List, Optional, Tuple

import torch

# Essential patch for PyTorch 2.6 on Windows to prevent PEFT attribute error
try:
    import torch.distributed.tensor
except ImportError:
    pass

from transformers import AutoModelForCausalLM, PreTrainedTokenizer
from peft import PeftModel

from ocr_lora.model import load_model_and_adapter
from ocr_lora.infer import correct_text_batch
from webapp.config import settings
from webapp.schemas import AdapterInfo, CorrectResponse
from webapp.services.diff import compute_char_diff
from webapp.services.gpu_guard import empty_gpu_cache


class ModelManager:
    """Thread-safe singleton managing LLM weights and PEFT adapters in memory."""

    _instance: Optional["ModelManager"] = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self) -> None:
        self.state: str = "unloaded"  # unloaded | loading | ready | busy | error
        self.active_adapter: Optional[str] = None
        self.active_adapter_name: Optional[str] = None
        self.model: Optional[PeftModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu_worker")
        self.is_training: bool = False
        self.last_error: Optional[str] = None

    def get_adapter_path(self, adapter_id_or_name: str) -> Tuple[str, str]:
        """Resolve adapter identifier to a sanitized name and local/Hub target path."""
        models_dir = Path(settings.MODELS_DIR)
        
        # Check direct directory matches
        sanitized_dir_name = adapter_id_or_name.replace("/", "--")
        local_cand = models_dir / sanitized_dir_name
        if local_cand.exists() and (local_cand / "adapter_config.json").exists():
            return sanitized_dir_name, str(local_cand)

        for folder in models_dir.iterdir():
            if folder.is_dir() and (folder / "adapter_config.json").exists():
                if folder.name == adapter_id_or_name or folder.name.endswith(f"--{adapter_id_or_name}"):
                    return folder.name, str(folder)

        # Fallback to remote repo ID
        clean_name = adapter_id_or_name.split("/")[-1]
        return clean_name, adapter_id_or_name

    def scan_adapters(self) -> List[AdapterInfo]:
        """Scan models directory for valid PEFT adapters and report configuration."""
        adapters: List[AdapterInfo] = []
        models_dir = Path(settings.MODELS_DIR)
        if not models_dir.exists():
            return adapters

        for item in sorted(models_dir.iterdir()):
            if item.is_dir() and (item / "adapter_config.json").exists():
                cfg_path = item / "adapter_config.json"
                r = None
                alpha = None
                target_modules = None
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        r = cfg.get("r")
                        alpha = cfg.get("lora_alpha")
                        target_modules = cfg.get("target_modules")
                except Exception:
                    pass

                # Derive display name from folder name
                display_name = item.name.replace("--", "/")
                is_active = (self.active_adapter_name == item.name or self.active_adapter == display_name)

                adapters.append(
                    AdapterInfo(
                        name=display_name,
                        path=str(item),
                        is_active=is_active,
                        target_modules=target_modules,
                        r=r,
                        lora_alpha=alpha,
                    )
                )

        return adapters

    def _sync_unload(self) -> None:
        """Synchronously release model and CUDA VRAM allocations."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.active_adapter = None
        self.active_adapter_name = None
        self.state = "unloaded"
        empty_gpu_cache()

    async def unload(self) -> None:
        """Asynchronously unload model weights from GPU and system RAM."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._sync_unload)

    def _sync_load_adapter(self, adapter_id_or_name: str) -> None:
        """Load or switch PEFT adapter in single-worker thread."""
        adapter_name, adapter_path = self.get_adapter_path(adapter_id_or_name)

        # Case 1: Initial full load
        if self.model is None or self.tokenizer is None:
            self.state = "loading"
            try:
                model, tokenizer = load_model_and_adapter(
                    base_model_id=settings.BASE_MODEL_ID,
                    adapter_id=adapter_path,
                    token=settings.HF_TOKEN,
                )
                self.model = model
                self.tokenizer = tokenizer
                self.active_adapter = adapter_id_or_name
                self.active_adapter_name = adapter_name
                self.state = "ready"
                self.last_error = None
                return
            except Exception as e:
                self.state = "error"
                self.last_error = str(e)
                self._sync_unload()
                raise

        # Case 2: Adapter is already active
        if self.active_adapter_name == adapter_name and self.active_adapter == adapter_id_or_name:
            return

        # Case 3: In-place adapter swap with full reload fallback
        self.state = "loading"
        try:
            # Check if adapter is already mounted in PEFT
            if hasattr(self.model, "peft_config") and adapter_name in self.model.peft_config:
                self.model.set_adapter(adapter_name)
            else:
                self.model.load_adapter(adapter_path, adapter_name=adapter_name)
                self.model.set_adapter(adapter_name)

            self.active_adapter = adapter_id_or_name
            self.active_adapter_name = adapter_name
            self.state = "ready"
            self.last_error = None
        except Exception as in_place_err:
            # Fallback to clean reload (required when switching between 7-target and 4-target adapters)
            self._sync_unload()
            try:
                model, tokenizer = load_model_and_adapter(
                    base_model_id=settings.BASE_MODEL_ID,
                    adapter_id=adapter_path,
                    token=settings.HF_TOKEN,
                )
                self.model = model
                self.tokenizer = tokenizer
                self.active_adapter = adapter_id_or_name
                self.active_adapter_name = adapter_name
                self.state = "ready"
                self.last_error = None
            except Exception as reload_err:
                self.state = "error"
                self.last_error = f"In-place error ({in_place_err}); Reload error ({reload_err})"
                self._sync_unload()
                raise reload_err

    def _sync_correct(self, text: str, adapter_id_or_name: str) -> CorrectResponse:
        """Run inference and timing on the worker thread."""
        self._sync_load_adapter(adapter_id_or_name)

        self.state = "busy"
        t0 = time.perf_counter()
        try:
            predictions = correct_text_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                input_texts=[text],
                max_length=settings.MAX_INPUT_CHARS,
                max_new_tokens=settings.MAX_NEW_TOKENS,
            )
            t1 = time.perf_counter()
            corrected = predictions[0]

            # Measure performance metrics
            latency_ms = round((t1 - t0) * 1000.0, 2)
            token_ids = self.tokenizer.encode(corrected, add_special_tokens=False)
            num_tokens = len(token_ids)
            latency_sec = max(t1 - t0, 0.001)
            tokens_per_sec = round(num_tokens / latency_sec, 2)

            diff_ops = compute_char_diff(text, corrected)

            self.state = "ready"
            return CorrectResponse(
                corrected=corrected,
                diff=diff_ops,
                adapter=self.active_adapter or adapter_id_or_name,
                latency_ms=latency_ms,
                tokens=num_tokens,
                tokens_per_sec=tokens_per_sec,
            )
        except Exception as e:
            self.state = "error"
            self.last_error = str(e)
            raise

    async def correct(self, text: str, adapter: Optional[str] = None) -> CorrectResponse:
        """Asynchronously correct text with length validation and training locks."""
        if self.is_training:
            raise RuntimeError("training_in_progress")

        if len(text) > settings.MAX_INPUT_CHARS:
            raise ValueError(
                f"Input text exceeds maximum allowed length ({len(text)} > {settings.MAX_INPUT_CHARS} characters)."
            )

        target_adapter = adapter or settings.DEFAULT_ADAPTER

        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                self._executor,
                self._sync_correct,
                text,
                target_adapter,
            )


model_manager = ModelManager()
