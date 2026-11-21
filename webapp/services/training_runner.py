"""Training runner and process manager for fine-tuning OCR LoRA models.

Manages subprocess execution of train.py with bounded ring buffer logging,
real-time SSE streaming, progress parsing, and GPU safety preflight checks.
"""

from typing import Optional, Dict, Any, List
import os
import sys
import re
import json
import uuid
import time
import asyncio
import subprocess
import threading
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
import ast

import torch
from webapp.config import settings
from webapp.schemas import TrainRequest, TrainRunStatus, TrainProgressEvent
from webapp.services.events import event_broker
from webapp.services.gpu_guard import get_gpu_info, empty_gpu_cache
from webapp.services.model_manager import model_manager


ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
OUTPUT_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PROGRESS_STEP_RE = re.compile(r"(\d+)/(\d+)\s+\[.*?\]")


def get_venv_python() -> str:
    """Resolve Python executable path within the local virtual environment."""
    candidates = [
        Path("e:/ocr/venv/Scripts/python.exe"),
        Path(settings.MODELS_DIR).parent / "venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return sys.executable


class TrainingRunner:
    """Singleton service to supervise training subprocesses and stream metrics."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._current_run: Optional[Dict[str, Any]] = None
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._log_buffer: deque = deque(maxlen=1000)
        self._cancelled: bool = False
        self._log_file = None

    @property
    def is_running(self) -> bool:
        """Check if a training process is actively running."""
        return self._proc is not None and self._proc.poll() is None

    def get_runs_dir(self) -> Path:
        """Ensure and return the base runs directory."""
        runs_dir = Path(settings.RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)
        return runs_dir

    def get_run_dir(self, run_id: str) -> Path:
        """Get directory for a specific training run."""
        d = self.get_runs_dir() / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_meta(self, run_id: str, meta: Dict[str, Any]) -> None:
        """Save training run metadata atomically."""
        meta_path = self.get_run_dir(run_id) / "meta.json"
        temp_path = meta_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        temp_path.replace(meta_path)

    def load_meta(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load training run metadata from disk."""
        meta_path = self.get_run_dir(run_id) / "meta.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all past training runs ordered by start time descending."""
        runs = []
        runs_dir = self.get_runs_dir()
        for item in runs_dir.iterdir():
            if item.is_dir():
                meta = self.load_meta(item.name)
                if meta:
                    runs.append(meta)
        runs.sort(key=lambda r: r.get("started_at", ""), reverse=True)
        return runs

    async def preflight_check(self, req: TrainRequest) -> Path:
        """Perform GPU safety, VRAM, parameter bounds, and path conflict checks."""
        # 1. Parameter Bounds Enforcement
        if not OUTPUT_NAME_RE.match(req.output_name):
            raise ValueError(
                f"Invalid output name '{req.output_name}'. Must match ^[A-Za-z0-9_-]{{1,64}}$"
            )
        if not (1 <= req.epochs <= 10):
            raise ValueError(f"Epochs must be between 1 and 10 (got {req.epochs})")
        if not (1 <= req.batch_size <= 4):
            raise ValueError(f"Batch size must be between 1 and 4 (got {req.batch_size})")
        if not (1e-6 <= req.learning_rate <= 1e-3):
            raise ValueError(
                f"Learning rate must be between 1e-6 and 1e-3 (got {req.learning_rate})"
            )

        # 2. Path Conflict Check: Prevent overwriting existing adapters
        target_adapter_dir = Path(settings.MODELS_DIR) / req.output_name
        if target_adapter_dir.exists() and any(target_adapter_dir.iterdir()):
            raise FileExistsError(
                f"Adapter directory '{target_adapter_dir}' already exists in models/. Choose a different name."
            )

        # 3. Singleton Run Check
        if self.is_running:
            raise RuntimeError("training_in_progress")

        # 4. Unload inference model to free GPU memory
        print("[TrainingRunner] Unloading inference model to free VRAM for training...")
        await model_manager.unload()
        empty_gpu_cache()

        # 5. Check Free VRAM
        gpu_info = get_gpu_info()
        if gpu_info.available:
            print(f"[TrainingRunner] Current Free VRAM: {gpu_info.free_mb:.1f} MB (Total: {gpu_info.total_mb:.1f} MB)")
            if gpu_info.free_mb < settings.MIN_TRAIN_FREE_VRAM_MB:
                raise RuntimeError(
                    f"Insufficient free VRAM for training ({gpu_info.free_mb:.1f} MB free, minimum required is {settings.MIN_TRAIN_FREE_VRAM_MB:.1f} MB). Another process may be holding the GPU."
                )

        return target_adapter_dir

    async def start_training(self, req: TrainRequest) -> TrainRunStatus:
        """Validate, prepare environment, and launch train.py subprocess."""
        async with self._lock:
            target_adapter_dir = await self.preflight_check(req)

            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            run_dir = self.get_run_dir(run_id)
            log_path = run_dir / "train.log"

            # Set model manager state so inference & batch return 503
            model_manager.is_training = True
            self._cancelled = False
            self._log_buffer.clear()

            meta = {
                "run_id": run_id,
                "output_name": req.output_name,
                "output_dir": str(target_adapter_dir),
                "epochs": req.epochs,
                "batch_size": req.batch_size,
                "learning_rate": req.learning_rate,
                "grad_accum": req.grad_accum,
                "data_path": req.data_path,
                "status": "running",
                "current_epoch": 0.0,
                "current_step": 0,
                "total_steps": None,
                "current_loss": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "exit_code": None,
                "error": None,
            }
            self.save_meta(run_id, meta)
            self._current_run = meta

            # Build subprocess command
            python_bin = get_venv_python()
            cmd = [
                python_bin,
                "train.py",
                "--output-dir", str(target_adapter_dir),
                "--epochs", str(req.epochs),
                "--batch-size", str(req.batch_size),
                "--learning-rate", str(req.learning_rate),
                "--grad-accum", str(req.grad_accum),
            ]
            if req.data_path:
                cmd.extend(["--data", str(req.data_path)])

            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            self._log_file = open(log_path, "w", encoding="utf-8")

            # Launch process
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )

            # Start background reader thread
            loop = asyncio.get_running_loop()
            self._reader_thread = threading.Thread(
                target=self._pump_logs,
                args=(self._proc, run_id, loop),
                daemon=True,
                name=f"train_reader_{run_id}",
            )
            self._reader_thread.start()

            return self.get_status()

    def _pump_logs(self, proc: subprocess.Popen, run_id: str, loop: asyncio.AbstractEventLoop) -> None:
        """Reader thread: pumps stdout/stderr into buffer, file, and SSE events."""
        try:
            assert proc.stdout is not None
            for raw_line in iter(proc.stdout.readline, ""):
                if not raw_line:
                    break

                clean_line = ANSI_ESCAPE_RE.sub("", raw_line).rstrip()
                if not clean_line:
                    continue

                # 1. Append to ring buffer & log file
                self._log_buffer.append(clean_line)
                if self._log_file and not self._log_file.closed:
                    self._log_file.write(clean_line + "\n")
                    self._log_file.flush()

                # 2. Parse loss / epoch / step progress from trainer logs
                parsed_metrics = self._parse_progress(clean_line)
                if parsed_metrics and self._current_run:
                    self._current_run.update(parsed_metrics)

                # 3. Broadcast SSE log event
                event_data = {
                    "run_id": run_id,
                    "type": "log",
                    "status": "running" if not self._cancelled else "cancelling",
                    "line": clean_line,
                    "epoch": self._current_run.get("current_epoch") if self._current_run else None,
                    "step": self._current_run.get("current_step") if self._current_run else None,
                    "total_steps": self._current_run.get("total_steps") if self._current_run else None,
                    "loss": self._current_run.get("current_loss") if self._current_run else None,
                }
                asyncio.run_coroutine_threadsafe(
                    event_broker.publish("training", event_data),
                    loop,
                )

        except Exception as e:
            print(f"[TrainingRunner] Reader exception: {e}")
        finally:
            proc.wait()
            exit_code = proc.returncode

            if self._log_file and not self._log_file.closed:
                self._log_file.close()

            # Finalize run state
            if self._cancelled:
                final_status = "cancelled"
            elif exit_code == 0:
                final_status = "completed"
            else:
                final_status = "failed"

            if self._current_run:
                self._current_run["status"] = final_status
                self._current_run["exit_code"] = exit_code
                self._current_run["completed_at"] = datetime.now(timezone.utc).isoformat()
                self.save_meta(run_id, self._current_run)

            # Re-enable inference
            model_manager.is_training = False
            empty_gpu_cache()

            # Broadcast final status
            final_event = {
                "run_id": run_id,
                "type": "status",
                "status": final_status,
                "line": f"[Process exited with code {exit_code}] Status: {final_status}",
            }
            asyncio.run_coroutine_threadsafe(
                event_broker.publish("training", final_event),
                loop,
            )

    def _parse_progress(self, line: str) -> Optional[Dict[str, Any]]:
        """Extract epoch, step, and loss from SFTTrainer logs."""
        metrics: Dict[str, Any] = {}

        # Pattern 1: Dict-like trainer logs (e.g. {'loss': 0.234, 'learning_rate': ..., 'epoch': 1.25})
        if line.startswith("{'") and "'loss'" in line:
            try:
                parsed = ast.literal_eval(line)
                if isinstance(parsed, dict):
                    if "loss" in parsed:
                        metrics["current_loss"] = float(parsed["loss"])
                    if "epoch" in parsed:
                        metrics["current_epoch"] = float(parsed["epoch"])
                    if "step" in parsed:
                        metrics["current_step"] = int(parsed["step"])
            except Exception:
                pass

        # Pattern 2: Progress bar steps (e.g. 50/100 [00:15<00:15, ...])
        step_match = PROGRESS_STEP_RE.search(line)
        if step_match:
            try:
                cur = int(step_match.group(1))
                tot = int(step_match.group(2))
                metrics["current_step"] = cur
                metrics["total_steps"] = tot
            except Exception:
                pass

        return metrics if metrics else None

    async def cancel_training(self) -> Dict[str, Any]:
        """Cancel the active training process tree using taskkill."""
        if not self.is_running or self._proc is None:
            return {"status": "not_running", "message": "No active training run to cancel."}

        self._cancelled = True
        pid = self._proc.pid
        print(f"[TrainingRunner] Cancelling process tree for PID {pid}...")

        try:
            if sys.platform == "win32":
                # Kill entire process tree cleanly on Windows
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            else:
                self._proc.terminate()
        except Exception as e:
            print(f"[TrainingRunner] Error terminating process: {e}")

        # Wait briefly for process to exit
        for _ in range(20):
            if self._proc.poll() is not None:
                break
            await asyncio.sleep(0.2)

        model_manager.is_training = False
        empty_gpu_cache()

        if self._current_run:
            self._current_run["status"] = "cancelled"
            self._current_run["completed_at"] = datetime.now(timezone.utc).isoformat()
            self.save_meta(self._current_run["run_id"], self._current_run)

        return {"status": "cancelled", "pid": pid}

    def get_status(self) -> TrainRunStatus:
        """Get live status and recent log lines of current or last run."""
        if self._current_run is None:
            # Check most recent run from disk
            recent = self.list_runs()
            if recent:
                r = recent[0]
                # Load last 100 log lines from file
                log_file = self.get_run_dir(r["run_id"]) / "train.log"
                lines = []
                if log_file.exists():
                    try:
                        with open(log_file, "r", encoding="utf-8") as f:
                            lines = [ln.rstrip() for ln in f.readlines()[-200:]]
                    except Exception:
                        pass
                return TrainRunStatus(
                    run_id=r["run_id"],
                    output_name=r.get("output_name", "retrained"),
                    status=r.get("status", "idle"),
                    epochs=r.get("epochs", 3),
                    batch_size=r.get("batch_size", 4),
                    learning_rate=r.get("learning_rate", 2e-4),
                    grad_accum=r.get("grad_accum", 2),
                    current_epoch=r.get("current_epoch"),
                    current_step=r.get("current_step"),
                    total_steps=r.get("total_steps"),
                    current_loss=r.get("current_loss"),
                    started_at=r.get("started_at"),
                    completed_at=r.get("completed_at"),
                    exit_code=r.get("exit_code"),
                    error=r.get("error"),
                    output_dir=r.get("output_dir", ""),
                    log_lines=lines,
                )

            return TrainRunStatus(
                run_id="none",
                output_name="retrained",
                status="idle",
                epochs=3,
                batch_size=4,
                learning_rate=2e-4,
                grad_accum=2,
                output_dir="",
                log_lines=[],
            )

        r = self._current_run
        return TrainRunStatus(
            run_id=r["run_id"],
            output_name=r.get("output_name", "retrained"),
            status="running" if self.is_running else r.get("status", "completed"),
            epochs=r.get("epochs", 3),
            batch_size=r.get("batch_size", 4),
            learning_rate=r.get("learning_rate", 2e-4),
            grad_accum=r.get("grad_accum", 2),
            current_epoch=r.get("current_epoch"),
            current_step=r.get("current_step"),
            total_steps=r.get("total_steps"),
            current_loss=r.get("current_loss"),
            started_at=r.get("started_at"),
            completed_at=r.get("completed_at"),
            exit_code=r.get("exit_code"),
            error=r.get("error"),
            output_dir=r.get("output_dir", ""),
            log_lines=list(self._log_buffer),
        )


training_runner = TrainingRunner()
