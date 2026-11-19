"""Pydantic v2 schemas for the OCR correction web application."""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class CorrectRequest(BaseModel):
    """Payload for correcting OCR errors in a single string."""
    text: str = Field(..., description="Corrupted OCR input text")
    adapter: Optional[str] = Field(None, description="Optional adapter ID or local name")


class DiffOp(BaseModel):
    """Single character-level diff operation from SequenceMatcher."""
    tag: Literal["equal", "replace", "delete", "insert"]
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    a_text: str
    b_text: str


class CorrectResponse(BaseModel):
    """Result of OCR correction with diff and generation performance metrics."""
    corrected: str
    diff: List[DiffOp]
    adapter: str
    latency_ms: float
    tokens: int
    tokens_per_sec: float


class GpuInfo(BaseModel):
    """VRAM statistics and device specifications."""
    available: bool
    device_name: Optional[str] = None
    total_mb: float = 0.0
    used_mb: float = 0.0
    free_mb: float = 0.0


class AdapterInfo(BaseModel):
    """Metadata for an available local PEFT adapter."""
    name: str
    path: str
    is_active: bool = False
    target_modules: Optional[List[str]] = None
    r: Optional[int] = None
    lora_alpha: Optional[int] = None


class HealthResponse(BaseModel):
    """Overall system health, state machine status, and active adapter."""
    status: str = "ok"
    model_state: Literal["unloaded", "loading", "ready", "busy", "error"]
    active_adapter: Optional[str] = None
    gpu: GpuInfo
