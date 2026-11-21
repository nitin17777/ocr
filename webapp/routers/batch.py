"""Batch file processing endpoints with SSE progress and export downloads."""

import asyncio
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from sse_starlette.sse import EventSourceResponse

from webapp.config import settings
from webapp.schemas import (
    BatchJobSummary,
    BatchPreviewRow,
    BatchUploadResponse,
)
from webapp.services.batch_jobs import batch_manager
from webapp.services.events import event_broker
from webapp.services.model_manager import model_manager

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.post("", response_model=BatchUploadResponse)
async def upload_batch(
    file: UploadFile = File(...),
    adapter: Optional[str] = Form(None),
    column_name: Optional[str] = Form(None),
) -> BatchUploadResponse:
    """Upload a .txt, .csv, or .jsonl batch file for offline correction."""
    if model_manager.is_training:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"reason": "training_in_progress"},
        )

    target_adapter = adapter or settings.DEFAULT_ADAPTER
    content = await file.read()

    try:
        meta = batch_manager.create_job(
            filename=file.filename or "batch.txt",
            content_bytes=content,
            adapter=target_adapter,
            column_or_field=column_name,
        )
    except ValueError as e:
        err_msg = str(e)
        if "exceeds maximum allowed limit" in err_msg or "exceed limit" in err_msg:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=err_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )

    return BatchUploadResponse(
        job_id=meta["job_id"],
        filename=meta["filename"],
        total_rows=meta.get("total_rows", 0),
        detected_columns=meta.get("detected_columns", []),
        status=meta["status"],
    )


@router.post("/{job_id}/configure", response_model=BatchUploadResponse)
async def configure_column(
    job_id: str,
    column_name: str = Form(...),
) -> BatchUploadResponse:
    """Configure target column/field for a job awaiting selection."""
    try:
        meta = batch_manager.configure_job_column(job_id, column_name)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return BatchUploadResponse(
        job_id=meta["job_id"],
        filename=meta["filename"],
        total_rows=meta.get("total_rows", 0),
        detected_columns=meta.get("detected_columns", []),
        status=meta["status"],
    )


@router.get("/{job_id}/events")
async def batch_events(job_id: str):
    """Server-Sent Events (SSE) streaming progress and status updates for a batch job."""
    meta = batch_manager.load_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    async def event_generator():
        # Yield initial state immediately
        initial_progress = {
            "job_id": job_id,
            "status": meta.get("status", "queued"),
            "done_rows": meta.get("done_rows", 0),
            "total_rows": meta.get("total_rows", 0),
            "percent": (
                round((meta.get("done_rows", 0) / meta.get("total_rows", 1)) * 100.0, 1)
                if meta.get("total_rows", 0) > 0
                else 0.0
            ),
            "rows_per_sec": 0.0,
            "eta_seconds": None,
        }
        yield {
            "event": "progress",
            "data": initial_progress,
        }

        # If already terminal, close stream
        if meta.get("status") in ("completed", "cancelled", "error"):
            if meta.get("status") == "completed":
                yield {"event": "done", "data": initial_progress}
            else:
                yield {"event": "error", "data": {"status": meta.get("status"), "message": meta.get("error")}}
            return

        queue = await event_broker.subscribe(job_id)
        try:
            while True:
                # Poll queue with heartbeat
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield payload
                    if payload.get("event") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    # Keep-alive heartbeat ping
                    yield {"event": "ping", "data": "ping"}
        finally:
            await event_broker.unsubscribe(job_id, queue)

    return EventSourceResponse(event_generator())


@router.get("/{job_id}/preview", response_model=List[BatchPreviewRow])
async def get_preview(job_id: str, limit: int = 10) -> List[BatchPreviewRow]:
    """Fetch first N processed rows with character diffs."""
    meta = batch_manager.load_meta(job_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return batch_manager.get_preview_rows(job_id, limit=limit)


@router.get("/{job_id}/download")
async def download_results(job_id: str, format: str = "csv") -> Response:
    """Download batch correction results as CSV (UTF-8 BOM) or JSONL."""
    fmt = format.lower()
    if fmt not in ("csv", "jsonl"):
        raise HTTPException(status_code=400, detail="Invalid format. Supported: 'csv', 'jsonl'.")

    try:
        content, filename, media_type = batch_manager.export_results(job_id, fmt=fmt)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No results found for job.")

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{job_id}/cancel")
async def cancel_batch(job_id: str):
    """Cancel an ongoing or queued batch processing job."""
    success = await batch_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return {"status": "ok", "message": f"Job {job_id} cancellation requested."}


@router.get("", response_model=List[BatchJobSummary])
async def list_recent_jobs(limit: int = 30) -> List[BatchJobSummary]:
    """List recently created and executed batch jobs."""
    return batch_manager.list_jobs(limit=limit)
