"""GPU monitoring and OOM recovery utilities."""

import gc
from typing import Callable, List, TypeVar
import torch
from webapp.schemas import GpuInfo

T = TypeVar("T")


def get_gpu_info() -> GpuInfo:
    """Read VRAM statistics via torch.cuda.mem_get_info."""
    if not torch.cuda.is_available():
        return GpuInfo(
            available=False,
            device_name=None,
            total_mb=0.0,
            used_mb=0.0,
            free_mb=0.0,
        )

    try:
        free_b, total_b = torch.cuda.mem_get_info()
        used_b = total_b - free_b
        mb = 1024.0 * 1024.0
        return GpuInfo(
            available=True,
            device_name=torch.cuda.get_device_name(0),
            total_mb=round(total_b / mb, 2),
            used_mb=round(used_b / mb, 2),
            free_mb=round(free_b / mb, 2),
        )
    except Exception:
        name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        return GpuInfo(
            available=True,
            device_name=name,
            total_mb=0.0,
            used_mb=0.0,
            free_mb=0.0,
        )


def empty_gpu_cache() -> None:
    """Force garbage collection and release cached CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_with_oom_retry(
    fn: Callable[[List[str]], List[str]],
    texts: List[str],
    initial_batch_size: int = 8,
) -> List[str]:
    """Execute batched inference function with dynamic halving upon CUDA OOM."""
    batch_size = max(1, initial_batch_size)
    
    while batch_size >= 1:
        try:
            results: List[str] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                out = fn(chunk)
                results.extend(out)
            return results
        except torch.cuda.OutOfMemoryError:
            empty_gpu_cache()
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)

    raise RuntimeError("Batch execution failed with unrecoverable CUDA OOM.")
