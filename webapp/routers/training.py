"""FastAPI router for training runner management, SSE metrics, and process control."""

from typing import List, Dict, Any
import json
import asyncio
from fastapi import APIRouter, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from webapp.schemas import TrainRequest, TrainRunStatus
from webapp.services.training_runner import training_runner
from webapp.services.events import event_broker

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("", response_model=TrainRunStatus)
async def start_training(req: TrainRequest):
    """Initiate a fine-tuning run with preflight validation and GPU guards."""
    try:
        run_status = await training_runner.start_training(req)
        return run_status
    except FileExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except RuntimeError as e:
        err_msg = str(e)
        if err_msg == "training_in_progress":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"reason": "training_in_progress"},
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to launch training process: {str(e)}",
        )


@router.get("/status", response_model=TrainRunStatus)
async def get_training_status():
    """Retrieve current status, metrics, and recent log buffer of training."""
    return training_runner.get_status()


@router.get("/events")
async def stream_training_events():
    """Stream live training stdout/stderr logs and progress updates via Server-Sent Events."""
    async def event_generator():
        # First send the current status snapshot
        initial_status = training_runner.get_status()
        yield {
            "event": "message",
            "data": json.dumps({
                "run_id": initial_status.run_id,
                "type": "status",
                "status": initial_status.status,
                "line": f"Connected to training event stream (Status: {initial_status.status})",
            }),
        }

        # Subscribe to training event channel
        sub = await event_broker.subscribe("training")
        try:
            while True:
                data = await sub.get()
                yield {
                    "event": "message",
                    "data": json.dumps(data),
                }
        except asyncio.CancelledError:
            pass
        finally:
            await event_broker.unsubscribe("training", sub)

    return EventSourceResponse(event_generator())


@router.post("/cancel")
async def cancel_training():
    """Terminate the active training process tree."""
    res = await training_runner.cancel_training()
    return res


@router.get("/runs", response_model=List[Dict[str, Any]])
async def list_training_runs():
    """Retrieve historical metadata for all recorded training runs."""
    return training_runner.list_runs()
