"""Web application settings loaded via pydantic-settings from .env."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class WebSettings(BaseSettings):
    """Runtime configuration for FastAPI application."""

    HF_TOKEN: Optional[str] = None
    MODELS_DIR: Path = Path("models")
    BASE_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"
    DEFAULT_ADAPTER: str = "deep1104/ocr-lora-large"

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    MAX_INPUT_CHARS: int = 2048
    MAX_NEW_TOKENS: int = 100
    DEFAULT_BATCH_SIZE: int = 8

    # Batch file limits and job storage
    MAX_UPLOAD_MB: int = 50
    MAX_BATCH_ROWS: int = 50000
    JOBS_DIR: Path = Path("outputs/web/jobs")

    # Evaluation checkpoint path for analytics
    CHECKPOINT_PATH: Path = Path("data/ocr_eval_checkpoint.jsonl")

    # Training runs directory and minimum free VRAM requirement
    RUNS_DIR: Path = Path("outputs/web/runs")
    MIN_TRAIN_FREE_VRAM_MB: float = 2000.0


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = WebSettings()
