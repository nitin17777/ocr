"""Batch file processing service with resumable persistence and SSE progress."""

import asyncio
import csv
import io
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

from webapp.config import settings
from webapp.schemas import (
    BatchJobSummary,
    BatchPreviewRow,
    BatchProgressEvent,
    DiffOp,
)
from webapp.services.diff import compute_char_diff
from webapp.services.events import event_broker
from webapp.services.gpu_guard import execute_with_oom_retry
from webapp.services.model_manager import model_manager


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames to prevent directory traversal or invalid characters."""
    base = os.path.basename(filename)
    cleaned = re.sub(r"[^\w\s\.-]", "_", base)
    return cleaned.strip() or "batch_upload.txt"


def read_completed_indices_from_results(results_path: Path) -> Set[int]:
    """Inspect results.jsonl and collect indices already processed."""
    completed: Set[int] = set()
    if not results_path.exists():
        return completed

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "index" in rec:
                    completed.add(int(rec["index"]))
            except Exception:
                continue
    return completed


class BatchJobManager:
    """Manages batch job persistence, queueing, execution, and resumability."""

    def __init__(self):
        self.jobs_dir: Path = settings.JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._active_job_id: Optional[str] = None
        self._cancel_requested: Set[str] = set()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def get_job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def load_meta(self, job_id: str) -> Optional[Dict[str, Any]]:
        meta_file = self.get_job_dir(job_id) / "meta.json"
        if not meta_file.exists():
            return None
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_meta(self, job_id: str, meta: Dict[str, Any]) -> None:
        meta_file = self.get_job_dir(job_id) / "meta.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = meta_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, meta_file)

    def parse_input_file(
        self,
        content_bytes: bytes,
        filename: str,
        column_or_field: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[str], str]:
        """Parse .txt, .csv (BOM-aware), or .jsonl bytes into normalized rows."""
        ext = Path(filename).suffix.lower()
        rows: List[Dict[str, Any]] = []
        detected_columns: List[str] = []

        if ext == ".txt":
            detected_format = "txt"
            text_data = content_bytes.decode("utf-8", errors="replace")
            for idx, line in enumerate(text_data.splitlines()):
                line_str = line.strip()
                if line_str:
                    rows.append({"index": len(rows), "text": line_str, "raw": line_str})

        elif ext == ".csv":
            detected_format = "csv"
            # Decode using utf-8-sig to automatically strip BOM if present
            text_data = content_bytes.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(text_data))
            if reader.fieldnames:
                detected_columns = [c.strip() for c in reader.fieldnames if c and c.strip()]

            target_col = column_or_field or (detected_columns[0] if len(detected_columns) == 1 else None)

            if target_col and target_col in (reader.fieldnames or []):
                for idx, row in enumerate(reader):
                    val = row.get(target_col, "").strip()
                    rows.append({"index": idx, "text": val, "raw": row})

        elif ext == ".jsonl":
            detected_format = "jsonl"
            text_data = content_bytes.decode("utf-8-sig", errors="replace")
            parsed_json_rows: List[Dict[str, Any]] = []
            field_counts: Dict[str, int] = {}

            for line in text_data.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        parsed_json_rows.append(obj)
                        for k, v in obj.items():
                            if isinstance(v, str):
                                field_counts[k] = field_counts.get(k, 0) + 1
                except Exception:
                    continue

            detected_columns = sorted(list(field_counts.keys()))
            target_field = column_or_field or (detected_columns[0] if len(detected_columns) == 1 else None)

            if target_field:
                for idx, obj in enumerate(parsed_json_rows):
                    val = str(obj.get(target_field, "")).strip()
                    rows.append({"index": idx, "text": val, "raw": obj})
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported: .txt, .csv, .jsonl")

        return rows, detected_columns, detected_format

    def create_job(
        self,
        filename: str,
        content_bytes: bytes,
        adapter: str,
        column_or_field: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate, store, and create metadata for a new batch job."""
        # 1. Size checks
        mb_size = len(content_bytes) / (1024.0 * 1024.0)
        if mb_size > settings.MAX_UPLOAD_MB:
            raise ValueError(
                f"File size ({mb_size:.1f} MB) exceeds maximum allowed limit of {settings.MAX_UPLOAD_MB} MB."
            )

        sanitized = sanitize_filename(filename)
        rows, detected_columns, detected_format = self.parse_input_file(
            content_bytes, sanitized, column_or_field
        )

        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        job_dir = self.get_job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)

        # Save raw uploaded input
        raw_input_path = job_dir / "input.raw"
        with open(raw_input_path, "wb") as f:
            f.write(content_bytes)

        # Check if column selection is needed
        needs_column = (
            detected_format in ("csv", "jsonl")
            and not column_or_field
            and len(detected_columns) > 1
        )

        if needs_column:
            meta = {
                "job_id": job_id,
                "filename": sanitized,
                "format": detected_format,
                "column_or_field": None,
                "adapter": adapter,
                "total_rows": len(rows),
                "done_rows": 0,
                "status": "needs_column",
                "detected_columns": detected_columns,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "error": None,
            }
            self.save_meta(job_id, meta)
            return meta

        if len(rows) > settings.MAX_BATCH_ROWS:
            raise ValueError(
                f"File rows ({len(rows):,}) exceed maximum allowed limit of {settings.MAX_BATCH_ROWS:,} rows."
            )

        # Save normalized input records
        norm_input_path = job_dir / "input.jsonl"
        with open(norm_input_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        meta = {
            "job_id": job_id,
            "filename": sanitized,
            "format": detected_format,
            "column_or_field": column_or_field or (detected_columns[0] if detected_columns else None),
            "adapter": adapter,
            "total_rows": len(rows),
            "done_rows": 0,
            "status": "queued",
            "detected_columns": detected_columns,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "error": None,
        }
        self.save_meta(job_id, meta)
        self._queue.put_nowait(job_id)
        return meta

    def configure_job_column(self, job_id: str, column_or_field: str) -> Dict[str, Any]:
        """Configure target column/field for a job in 'needs_column' status and queue it."""
        meta = self.load_meta(job_id)
        if not meta or meta.get("status") != "needs_column":
            raise ValueError(f"Job {job_id} is not awaiting column configuration.")

        job_dir = self.get_job_dir(job_id)
        raw_input_path = job_dir / "input.raw"
        with open(raw_input_path, "rb") as f:
            content_bytes = f.read()

        rows, detected_columns, _ = self.parse_input_file(
            content_bytes, meta["filename"], column_or_field
        )

        if len(rows) > settings.MAX_BATCH_ROWS:
            raise ValueError(f"Rows ({len(rows):,}) exceed limit ({settings.MAX_BATCH_ROWS:,}).")

        # Save normalized rows
        norm_input_path = job_dir / "input.jsonl"
        with open(norm_input_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        meta["column_or_field"] = column_or_field
        meta["total_rows"] = len(rows)
        meta["status"] = "queued"
        self.save_meta(job_id, meta)
        self._queue.put_nowait(job_id)
        return meta

    async def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of an active or queued job."""
        meta = self.load_meta(job_id)
        if not meta:
            return False

        self._cancel_requested.add(job_id)
        if meta["status"] in ("queued", "needs_column"):
            meta["status"] = "cancelled"
            meta["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.save_meta(job_id, meta)
            await event_broker.publish(job_id, "error", {"status": "cancelled", "message": "Job cancelled by user"})
            return True

        return True

    def list_jobs(self, limit: int = 50) -> List[BatchJobSummary]:
        """List recent jobs sorted by creation timestamp."""
        summaries: List[BatchJobSummary] = []
        if not self.jobs_dir.exists():
            return summaries

        for folder in sorted(self.jobs_dir.iterdir(), reverse=True):
            if folder.is_dir() and (folder / "meta.json").exists():
                try:
                    with open(folder / "meta.json", "r", encoding="utf-8") as f:
                        data = json.load(f)
                    summaries.append(
                        BatchJobSummary(
                            job_id=data["job_id"],
                            filename=data["filename"],
                            adapter=data.get("adapter", settings.DEFAULT_ADAPTER),
                            total_rows=data.get("total_rows", 0),
                            done_rows=data.get("done_rows", 0),
                            status=data.get("status", "error"),
                            created_at=data.get("created_at", ""),
                            completed_at=data.get("completed_at"),
                            error=data.get("error"),
                        )
                    )
                    if len(summaries) >= limit:
                        break
                except Exception:
                    continue
        return summaries

    def get_preview_rows(self, job_id: str, limit: int = 10) -> List[BatchPreviewRow]:
        """Fetch the first N results with character diffs for previewing."""
        results_file = self.get_job_dir(job_id) / "results.jsonl"
        rows: List[BatchPreviewRow] = []
        if not results_file.exists():
            return rows

        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    orig = data.get("original", "")
                    corr = data.get("corrected", "")
                    diff_ops = compute_char_diff(orig, corr)
                    rows.append(
                        BatchPreviewRow(
                            index=data.get("index", len(rows)),
                            original=orig,
                            corrected=corr,
                            diff=diff_ops,
                        )
                    )
                    if len(rows) >= limit:
                        break
                except Exception:
                    continue

        return rows

    def export_results(self, job_id: str, fmt: str = "jsonl") -> Tuple[bytes, str, str]:
        """Export completed job results preserving original row order."""
        job_dir = self.get_job_dir(job_id)
        results_file = job_dir / "results.jsonl"
        meta = self.load_meta(job_id) or {}
        orig_filename = Path(meta.get("filename", "batch")).stem

        if not results_file.exists():
            raise FileNotFoundError(f"No results found for job {job_id}")

        all_records: List[Dict[str, Any]] = []
        with open(results_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except Exception:
                        pass

        # Sort by original input index to guarantee original order
        all_records.sort(key=lambda r: r.get("index", 0))

        if fmt == "csv":
            out_filename = f"{orig_filename}_corrected.csv"
            media_type = "text/csv; charset=utf-8"
            
            # UTF-8 with BOM for Excel compatibility
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(["index", "original", "corrected"])
            for r in all_records:
                writer.writerow([r.get("index"), r.get("original", ""), r.get("corrected", "")])
            
            content = buffer.getvalue().encode("utf-8-sig")
            return content, out_filename, media_type

        else:
            # JSONL
            out_filename = f"{orig_filename}_corrected.jsonl"
            media_type = "application/x-ndjson; charset=utf-8"
            buffer = io.BytesIO()
            for r in all_records:
                buffer.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
            return buffer.getvalue(), out_filename, media_type

    async def _execute_job(self, job_id: str) -> None:
        """Core batch execution runner."""
        meta = self.load_meta(job_id)
        if not meta:
            return

        job_dir = self.get_job_dir(job_id)
        norm_input_path = job_dir / "input.jsonl"
        results_path = job_dir / "results.jsonl"

        if not norm_input_path.exists():
            meta["status"] = "error"
            meta["error"] = "Input normalized records not found."
            self.save_meta(job_id, meta)
            return

        # 1. Read input records
        input_records: List[Dict[str, Any]] = []
        with open(norm_input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    input_records.append(json.loads(line))

        total_rows = len(input_records)
        meta["total_rows"] = total_rows
        meta["status"] = "running"
        self.save_meta(job_id, meta)

        # 2. Check completed records for resume capability
        completed_indices = read_completed_indices_from_results(results_path)
        pending_records = [r for r in input_records if r["index"] not in completed_indices]

        # 3. Sort pending rows by length to optimize GPU batch padding
        pending_records.sort(key=lambda r: len(r.get("text", "")))

        done_rows = len(completed_indices)
        start_time = time.time()
        batch_size = settings.DEFAULT_BATCH_SIZE

        # Open results.jsonl in append mode
        with open(results_path, "a", encoding="utf-8") as out_f:
            for i in range(0, len(pending_records), batch_size):
                # Check for cancellation
                if job_id in self._cancel_requested:
                    meta["status"] = "cancelled"
                    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
                    self.save_meta(job_id, meta)
                    await event_broker.publish(
                        job_id,
                        "error",
                        {"status": "cancelled", "message": "Job cancelled by user."},
                    )
                    return

                # Check if training took over
                if model_manager.is_training:
                    meta["status"] = "error"
                    meta["error"] = "Training started during batch run. Job paused."
                    self.save_meta(job_id, meta)
                    await event_broker.publish(
                        job_id,
                        "error",
                        {"status": "error", "message": "Training in progress. Batch halted."},
                    )
                    return

                chunk = pending_records[i : i + batch_size]
                chunk_texts = [r.get("text", "") for r in chunk]

                # Run GPU inference with OOM halving retry
                try:
                    # Async lock around model inference
                    async with model_manager._lock:
                        # Synchronously load adapter and run batched inference in worker thread
                        loop = asyncio.get_running_loop()
                        predictions = await loop.run_in_executor(
                            model_manager._executor,
                            self._sync_infer_batch,
                            chunk_texts,
                            meta.get("adapter", settings.DEFAULT_ADAPTER),
                        )
                except Exception as infer_err:
                    meta["status"] = "error"
                    meta["error"] = str(infer_err)
                    self.save_meta(job_id, meta)
                    await event_broker.publish(job_id, "error", {"status": "error", "message": str(infer_err)})
                    return

                # Write results with atomic flush + fsync
                for rec, pred in zip(chunk, predictions):
                    out_record = {
                        "index": rec["index"],
                        "original": rec.get("text", ""),
                        "corrected": pred,
                        "raw": rec.get("raw"),
                    }
                    out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

                out_f.flush()
                os.fsync(out_f.fileno())

                done_rows += len(chunk)
                elapsed = max(time.time() - start_time, 0.001)
                rows_per_sec = round((done_rows - len(completed_indices)) / elapsed, 2)
                remaining_rows = total_rows - done_rows
                eta_sec = round(remaining_rows / rows_per_sec, 1) if rows_per_sec > 0 else 0.0
                percent = round((done_rows / total_rows) * 100.0, 1) if total_rows > 0 else 100.0

                meta["done_rows"] = done_rows
                self.save_meta(job_id, meta)

                progress_data = {
                    "job_id": job_id,
                    "status": "running",
                    "done_rows": done_rows,
                    "total_rows": total_rows,
                    "percent": percent,
                    "rows_per_sec": rows_per_sec,
                    "eta_seconds": eta_sec,
                }
                await event_broker.publish(job_id, "progress", progress_data)

        # Job completed
        meta["status"] = "completed"
        meta["done_rows"] = total_rows
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.save_meta(job_id, meta)

        await event_broker.publish(
            job_id,
            "done",
            {
                "job_id": job_id,
                "status": "completed",
                "done_rows": total_rows,
                "total_rows": total_rows,
                "percent": 100.0,
            },
        )

    def _sync_infer_batch(self, texts: List[str], adapter: str) -> List[str]:
        """Synchronously execute inference on the model thread with OOM retry."""
        model_manager._sync_load_adapter(adapter)
        
        def _call_chunk(chunk_texts: List[str]) -> List[str]:
            from ocr_lora.infer import correct_text_batch
            return correct_text_batch(
                model=model_manager.model,
                tokenizer=model_manager.tokenizer,
                input_texts=chunk_texts,
                max_length=settings.MAX_INPUT_CHARS,
                max_new_tokens=settings.MAX_NEW_TOKENS,
            )

        return execute_with_oom_retry(
            fn=_call_chunk,
            texts=texts,
            initial_batch_size=settings.DEFAULT_BATCH_SIZE,
        )

    async def _queue_worker(self) -> None:
        """Background worker consuming jobs one at a time."""
        while True:
            job_id = await self._queue.get()
            self._active_job_id = job_id
            try:
                await self._execute_job(job_id)
            except Exception as e:
                meta = self.load_meta(job_id)
                if meta:
                    meta["status"] = "error"
                    meta["error"] = str(e)
                    self.save_meta(job_id, meta)
                await event_broker.publish(job_id, "error", {"status": "error", "message": str(e)})
            finally:
                self._active_job_id = None
                self._queue.task_done()

    def start_worker(self) -> None:
        """Start background queue worker and queue unfinished jobs from disk."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._queue_worker())

        # Scan for unfinished jobs to resume seamlessly on restart
        if self.jobs_dir.exists():
            for folder in sorted(self.jobs_dir.iterdir()):
                if folder.is_dir() and (folder / "meta.json").exists():
                    try:
                        with open(folder / "meta.json", "r", encoding="utf-8") as f:
                            m = json.load(f)
                        if m.get("status") in ("queued", "running"):
                            # Re-queue for continuation
                            self._queue.put_nowait(m["job_id"])
                    except Exception:
                        pass


batch_manager = BatchJobManager()
