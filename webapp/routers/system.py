"""System endpoints: health, GPU monitoring, and adapter catalog."""

from typing import List
from fastapi import APIRouter
from webapp.schemas import HealthResponse, GpuInfo, AdapterInfo
from webapp.services.gpu_guard import get_gpu_info
from webapp.services.model_manager import model_manager

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Return current server status, model lifecycle state, and active adapter."""
    return HealthResponse(
        status="ok",
        model_state=model_manager.state,
        active_adapter=model_manager.active_adapter,
        gpu=get_gpu_info(),
    )


@router.get("/gpu", response_model=GpuInfo)
async def get_gpu() -> GpuInfo:
    """Return real-time VRAM allocation, free space, and device specs."""
    return get_gpu_info()


@router.get("/adapters", response_model=List[AdapterInfo])
async def list_adapters() -> List[AdapterInfo]:
    """Scan local models directory and list all available PEFT adapters."""
    return model_manager.scan_adapters()
