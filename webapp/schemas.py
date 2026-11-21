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


class BatchUploadResponse(BaseModel):
    """Response returned upon uploading a batch processing file."""
    job_id: str
    filename: str
    total_rows: int
    detected_columns: List[str] = []
    status: Literal["queued", "running", "needs_column", "completed", "cancelled", "error"]


class BatchProgressEvent(BaseModel):
    """Progress event payload streamed via Server-Sent Events (SSE)."""
    job_id: str
    status: Literal["queued", "running", "completed", "cancelled", "error"]
    done_rows: int
    total_rows: int
    percent: float
    rows_per_sec: float
    eta_seconds: Optional[float] = None
    error: Optional[str] = None


class BatchPreviewRow(BaseModel):
    """Single row representation in the batch results preview table."""
    index: int
    original: str
    corrected: str
    diff: List[DiffOp]


class BatchJobSummary(BaseModel):
    """Summary of a batch job for cataloging and historical status."""
    job_id: str
    filename: str
    adapter: str
    total_rows: int
    done_rows: int
    status: Literal["queued", "running", "needs_column", "completed", "cancelled", "error"]
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


class AnalyticsSummaryResponse(BaseModel):
    """Overall dataset metrics and domain-stratified evaluation accuracy."""
    total_rows: int
    exact_matches: int
    exact_match_acc: float
    math_total: int
    math_correct: int
    math_acc: float
    text_total: int
    text_correct: int
    text_acc: float
    failures_count: int
    checkpoint_file: str


class ConfusionPair(BaseModel):
    """Single character or string confusion pair."""
    expected: str
    predicted: str
    count: int
    domain: str = "all"


class ConfusionMatrixData(BaseModel):
    """Confusion matrix heatmap grid and top confusion pairs."""
    characters: List[str]
    grid: List[List[int]]
    max_count: int
    top_pairs: List[ConfusionPair]
    domain: str = "all"


class FailureItem(BaseModel):
    """Single failed evaluation row with side-by-side diff."""
    index: int
    input: str
    expected: str
    prediction: str
    domain: Literal["math", "text"]
    diff: List[DiffOp]


class FailuresResponse(BaseModel):
    """Paginated search response for failed predictions."""
    total: int
    page: int
    page_size: int
    total_pages: int
    failures: List[FailureItem]


